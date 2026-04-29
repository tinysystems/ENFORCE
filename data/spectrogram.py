import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.fftpack import dct
from scipy.signal import get_window

SAMPLE_RATE = 16000
FRAME_DUR = 0.032
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DUR)
FRAME_STRIDE_DUR = 0.024
FRAME_STRIDE = int(SAMPLE_RATE * FRAME_STRIDE_DUR)
NUM_BINS = FRAME_SIZE // 2
FILTER_NUMBER = 40
MIN_FREQ = 0
MAX_FREQ = SAMPLE_RATE // 2
COEFFICIENT = 0.96875
NOISE_FLOOR = -40.0

def pre_emphasis(audio):
    emphasized = np.zeros_like(audio, dtype=np.float32)
    emphasized[0] = audio[0] / 32768.0
    for i in range(1, len(audio)):
        emphasized[i] = (audio[i] / 32768.0) - COEFFICIENT * (audio[i-1] / 32768.0)
    return emphasized

def apply_windowing(frame):
    window = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(len(frame)) / (len(frame) - 1))
    return frame * window

def hz_to_mel(hz):
    return 1127.0 * np.log10(1 + hz / 700.0)

def mel_to_hz(mel):
    return 700 * (10 ** (mel / 1127.0) - 1)

def create_mel_filterbank():
    min_mel = hz_to_mel(MIN_FREQ)
    max_mel = hz_to_mel(MAX_FREQ)
    #mel_points = np.linspace(min_mel, max_mel, FILTER_NUMBER + 2)
    #hz_points = mel_to_hz(mel_points)
    mel_points = np.zeros(FILTER_NUMBER + 2)
    mel_spacing = (max_mel - min_mel) / (FILTER_NUMBER + 1)
    for i in range(FILTER_NUMBER + 2):
        mel_points[i] = mel_to_hz(min_mel + i * mel_spacing)
        if mel_points[i] > MAX_FREQ:
            mel_points[i] = MAX_FREQ

    #bin_indices = np.floor((NUM_BINS) * hz_points / (SAMPLE_RATE / 2)).astype(int)
    #bin_indices = np.clip(bin_indices, 0, NUM_BINS - 1)
    bin_indices = np.zeros(FILTER_NUMBER + 2, dtype=int)
    for i in range(FILTER_NUMBER + 2):
        bin_indices[i] = int(mel_points[i] * (NUM_BINS - 1) / (SAMPLE_RATE / 2.0))
        bin_indices[i] = max(0, min(NUM_BINS - 1, bin_indices[i]))

    filterbank = np.zeros((FILTER_NUMBER, NUM_BINS))

    for i in range(FILTER_NUMBER):
        left = bin_indices[i]
        middle = bin_indices[i+1]
        right = bin_indices[i+2]

        if left == middle:
            middle = min(left + 1, NUM_BINS - 1)
        if middle == right:
            right = min(middle + 1, NUM_BINS - 1)

        #filterbank[i, left:middle] = np.linspace(0, 1, middle - left)
        for j in range(left, middle):
            filterbank[i, j] = (j - left) / (middle - left)

        #filterbank[i, middle:right] = np.linspace(1, 0, right - middle)
        for j in range(middle, right):
            filterbank[i, j] = 1.0 - (j - middle) / (right - middle)
    return filterbank

def compute_spectrogram(audio, show_plot=True):
    num_samples = len(audio)

    total_duration = num_samples / SAMPLE_RATE
    num_frames_full_second = int((total_duration - FRAME_DUR) / FRAME_STRIDE_DUR) + 1
    num_frames = min(num_frames_full_second, 40)
    pre_emphasis_array = pre_emphasis(audio)
    spectrogram = np.zeros((num_frames, NUM_BINS))

    for frame in range(num_frames):
        start = frame * FRAME_STRIDE
        end = start + FRAME_SIZE
        segment = pre_emphasis_array[start:end]
        if len(segment) < FRAME_SIZE:
            segment = np.pad(segment, (0, FRAME_SIZE - len(segment)))

        windowed = apply_windowing(segment)
        fft = np.fft.rfft(windowed, n=FRAME_SIZE)
        magnitude = np.abs(fft)
        spectrogram[frame] = magnitude[:NUM_BINS]

    mel_filterbank = create_mel_filterbank()
    mel_spectrogram = np.dot(spectrogram, mel_filterbank.T)
    log_mel_spectrogram = 10 * np.log10(mel_spectrogram + 1e-20)

    log_mel_spectrogram = (log_mel_spectrogram - NOISE_FLOOR) / (-NOISE_FLOOR + 12)
    log_mel_spectrogram = np.clip(log_mel_spectrogram, 0, 1)
    quantized = np.round(log_mel_spectrogram * 256) / 256.0
    quantized = np.where(quantized >= 0.65, quantized, 0)
    quantized = quantized[:40]

    if show_plot:
        plt.figure(figsize=(10, 6))
        time_axis = np.linspace(0, 0.968, 40)
        plt.imshow(quantized.T, aspect='auto', origin='lower',
                  extent=[0, 0.968, 0, FILTER_NUMBER])
        plt.colorbar(label='Magnitude')
        plt.xlabel('Time (s)')
        plt.ylabel('Mel filter index')
        plt.title('40x40 Mel Spectrogram (0.968s duration)')
        plt.show()

    return quantized

if __name__ == "__main__":
    audio_file = <Insert WAV FILE>
    audio, sr = librosa.load(audio_file, sr=SAMPLE_RATE, mono=True)

    audio_int16 = (audio * 32768).astype(np.int16)

    required_samples = int(0.968 * SAMPLE_RATE)
    audio_int16 = audio_int16[:required_samples]

    mel_spectrogram = compute_spectrogram(audio_int16)
    print(f"Spectrogram shape: {mel_spectrogram.shape}")

    with open('mfe_features_formatted.txt', 'w') as f:
      for row in mel_spectrogram:
          # Write each row with 6 decimal places, space-separated
          line = ' '.join([f'{x:.6f}' for x in row])
          f.write(line + '\n')
