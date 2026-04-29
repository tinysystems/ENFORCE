#include "spectrogram.h"
#include <stdio.h>
#include <math.h>
#include <string.h>

// Function to save debug output to a file
void save_debug_output(const char* filename, const char* message, float* data, int rows, int cols) {
    FILE* file = fopen(filename, "a");
    if (!file) {
        printf("Error: Could not open debug file for writing\n");
        return;
    }
    fprintf(file, "%s\n", message);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fprintf(file, "%.6f ", data[i * cols + j]);
        }
        fprintf(file, "\n");
    }
    fprintf(file, "\n");
    fclose(file);
}

void pre_emphasis(short* audio, float pre_emphasis_array[], unsigned long num_samples) {
    float norm[num_samples];
    for (int i = 0; i < num_samples; i++) {
        norm[i] = audio[i] / 32768.0f;
    }
    pre_emphasis_array[0] = norm[0];
    for (int i = 1; i < num_samples; i++) {
        pre_emphasis_array[i] = norm[i] - COEFFICIENT * norm[i - 1];
    }
    // Save debug output
    // save_debug_output("debug.txt", "Pre-emphasis output:", pre_emphasis_array, 1, num_samples);
}

void apply_windowing(float* frame, int size) {
    for (int i = 0; i < size; i++) {
        float window = 0.54f - 0.46f * cos(2 * M_PI * i / (size - 1));
        frame[i] *= window;
    }
    // Save debug output
    float windowed_frame[size];
    for (int i = 0; i < size; i++) {
        windowed_frame[i] = frame[i];  // Save real part for debugging
    }
    //save_debug_output("debug.txt", "Windowed frame output:", windowed_frame, 1, size);
}

void spectrogram_population(kiss_fft_cpx fft_out[], float spectrogram[], int frame) {
    for (int i = 0; i < NUM_BINS; i++) {
        spectrogram[frame * NUM_BINS + i] = sqrt(fft_out[i].r * fft_out[i].r + fft_out[i].i * fft_out[i].i);
    }
    // Save debug output
    //save_debug_output("debug.txt", "Spectrogram population output:", spectrogram + frame * NUM_BINS, 1, NUM_BINS);
}

void framing_operation(float* pre_emphasis_audio, float spectrogram[], unsigned long num_samples) {
    kiss_fftr_cfg cfg = kiss_fftr_alloc(FRAME_SIZE, 0, NULL, NULL);
    kiss_fft_cpx fft_out[FRAME_SIZE/2+1];
    float fft_in[FRAME_SIZE];

    for (int frame = 0; frame < NUM_FRAMES(num_samples); frame++) {
        int start = frame * FRAME_STRIDE;
        for (int i = 0; i < FRAME_SIZE; i++) {
            fft_in[i] = (i < FRAME_SIZE) ? pre_emphasis_audio[start + i] : 0.0f;
        }
        apply_windowing(fft_in, FRAME_SIZE);
        kiss_fftr(cfg, fft_in, fft_out);
        spectrogram_population(fft_out, spectrogram, frame);
    }
    free(cfg);
    // Save debug output
    //save_debug_output("debug.txt", "Framing operation output:", spectrogram, NUM_FRAMES(num_samples), NUM_BINS);
}

float hz_to_mel(float hz) {
    return 1127.0f * log10(1+hz/700+1e-20);
}

float mel_to_hz(float mel) {
    return 700 * (pow(10,mel/1127.0f) - 1);
}

void triangle(float* out, float* x, int size, int left, int middle, int right) {
    for(int i=0; i<size; i++) {
        if(x[i]<=left || x[i]>=right) {
            out[i]=0.0f;
        }
        else if (x[i]<=middle) {
            out[i]=(x[i]-left)/(middle-left);
        }
        else {
            out[i]=(right-x[i])/(right-middle);
        }
        if (!isfinite(out[i])) out[i] = 0.0f;
        //printf("%.6f\t", out[i]);
    }
}

void create_mel_filterbank(float mel_filterbank[FILTER_NUMBER][NUM_BINS]) {
    float min_mel = hz_to_mel(MIN_FREQ); //0
    float max_mel = hz_to_mel(MAX_FREQ); //1233,4127
    float mel_spacing=(max_mel-min_mel)/(FILTER_NUMBER+1);
    float mel_points[FILTER_NUMBER + 2];
    for (int i = 0; i < FILTER_NUMBER + 2; i++) {
        float value=min_mel+i*mel_spacing;
        if(value>max_mel) {
            value=max_mel;
        }
        mel_points[i]= mel_to_hz(value);
        if(mel_points[i]>MAX_FREQ) {
            mel_points[i]=MAX_FREQ;
        }
    }
    for (int i=0; i < FILTER_NUMBER; i++) {
        for(int j=0; j<NUM_BINS; j++) {
            mel_filterbank[i][j]=0.0f;
        }
        float left=mel_points[i];
        float middle=mel_points[i+1];
        float right=mel_points[i+2];

        int left_bin = (int)(left * (NUM_BINS - 1) / (SAMPLE_RATE / 2));
        int middle_bin = (int)(middle * (NUM_BINS - 1) / (SAMPLE_RATE / 2));
        int right_bin = (int)(right * (NUM_BINS - 1) / (SAMPLE_RATE / 2));

        left_bin = max(0, min(NUM_BINS-1, left_bin));
        middle_bin = max(0, min(NUM_BINS-1, middle_bin));
        right_bin = max(0, min(NUM_BINS-1, right_bin));

        if (left_bin==middle_bin) middle_bin = min(left_bin+1, NUM_BINS-1);
        if (middle_bin==right_bin) right_bin = min(middle_bin+1, NUM_BINS-1);

        int size = right_bin - left_bin + 1;
        float z[size];
        for(int j=0; j<size; j++) {
            z[j]=left_bin+j;
        }
        float out[size];
        triangle(out, z, size, left_bin, middle_bin, right_bin);
        for(int j=0; j<size; j++) {
            mel_filterbank[i][left_bin+j]=out[j];
        }
    }
    // Save debug output
    //save_debug_output("debug.txt", "Mel filterbank output:", (float*)mel_filterbank, FILTER_NUMBER, NUM_BINS);
}

void apply_mel_filterbank(float spectrogram[], float mel_filterbank[FILTER_NUMBER][NUM_BINS], float log_mel_spectrogram[], int num_frames) {
    for (int i = 0; i < num_frames; i++) {
        for (int j = 0; j < FILTER_NUMBER; j++) {
            float sum = 0.0f;
            for (int k = 0; k < NUM_BINS; k++) {
                sum += spectrogram[i * NUM_BINS + k] * mel_filterbank[j][k];
            }
            log_mel_spectrogram[i * FILTER_NUMBER + j] = 10*log10(sum+1e-20);
        }
    }
    // Save debug output
    //save_debug_output("debug.txt", "Log Mel spectrogram output:", log_mel_spectrogram, num_frames, FILTER_NUMBER);
}

void save_spectrogram(float log_mel_spectrogram[], int num_frames, const char* filename) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        printf("Error: Could not open file for writing\n");
        return;
    }
    for (int frame = 0; frame < num_frames; frame++) {
        for (int i = 0; i < FILTER_NUMBER; i++) {
            fprintf(file, "%.6f ", log_mel_spectrogram[frame * FILTER_NUMBER + i]);
        }
        fprintf(file, "\n");
    }
   /*for(int i=0; i<num_frames; i++) {
        for(int j=0; j<FILTER_NUMBER; j++) {
            fprintf(file, "%.6f\t%.6f\n", log_mel_spectrogram[i * FILTER_NUMBER + j], spectrogram_sample[i * FILTER_NUMBER + j]);
        }
    }*/
    fclose(file);
}

void mean_filter(float log_mel_spectrogram[], int num_frames) {
    for (int i = 0; i < num_frames; i++) {
        for (int j = 0; j < FILTER_NUMBER; j++) {
            float sum = 0.0f;
            int count = 0;
            for (int di = -1; di <= 1; di++) {
                for (int dj = -1; dj <= 1; dj++) {
                    int ni = i + di;
                    int nj = j + dj;
                    if (ni >= 0 && ni < num_frames && nj >= 0 && nj < FILTER_NUMBER) {
                        sum += log_mel_spectrogram[ni * FILTER_NUMBER + nj];
                        count++;
                    }
                }
            }
            log_mel_spectrogram[i * FILTER_NUMBER + j] = sum / count;
        }
    }
    // Save debug output
    //save_debug_output("debug.txt", "Mean filter output:", log_mel_spectrogram, num_frames, FILTER_NUMBER);
}

void apply_noise_floor(float log_mel_spectrogram[], int num_frames) {
    for (int i = 0; i < num_frames; i++) {
        for (int j = 0; j < FILTER_NUMBER; j++) {
            float value = log_mel_spectrogram[i * FILTER_NUMBER + j];
            value = (value - NOISE_FLOOR) / ((-1 * NOISE_FLOOR) + 12);
            if (value < 0) value = 0;
            if (value > 1) value = 1;
            int quantized = (int)round(value * 256);
            if (quantized < 0) quantized = 0;
            if (quantized > 255) quantized = 255;
            log_mel_spectrogram[i * FILTER_NUMBER + j] = (float)quantized / 256.0f;
            log_mel_spectrogram[i*FILTER_NUMBER+j]=(log_mel_spectrogram[i*FILTER_NUMBER+j]>=0.65) ? log_mel_spectrogram[i*FILTER_NUMBER+j] : 0.0f;
        }
    }
    // Save debug output
    //save_debug_output("debug.txt", "Noise floor output:", log_mel_spectrogram, num_frames, FILTER_NUMBER);
}

void compute_spectrogram(short* audio, float log_mel_spectrogram[], unsigned long num_samples) {
    float pre_emphasis_array[num_samples];
    pre_emphasis(audio, pre_emphasis_array, num_samples);

    float spectrogram[NUM_FRAMES(num_samples) * NUM_BINS];
    framing_operation(pre_emphasis_array, spectrogram, num_samples);

    float mel_filterbank[FILTER_NUMBER][NUM_BINS];
    create_mel_filterbank(mel_filterbank);

    apply_mel_filterbank(spectrogram, mel_filterbank, log_mel_spectrogram, NUM_FRAMES(num_samples));

    // Uncomment to apply mean filter
    //mean_filter(log_mel_spectrogram, NUM_FRAMES(num_samples));

    apply_noise_floor(log_mel_spectrogram, NUM_FRAMES(num_samples));

    save_spectrogram(log_mel_spectrogram, NUM_FRAMES(num_samples), SPECTROGRAM_FILENAME);
}
