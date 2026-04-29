# Adapting-Pretrained-Models-for-TinyML-Acceleration

The current project aims to take a general Convolution Neural Network for audio applications that contains Conv2D, MaxPooling2D and Dense Layers and convert them into a fixed dimensional model composed by 3 Fully Connected layers of 256 neurons and an output layer. 

# Structure
```bash
Adalting-Pretrained-Models-for-TinyML-Acceleration/
│── README.md
│── requirements.txt                 #Required python libraries
│── images/                          #Images of the results or models and displayed in the README.md
│── data/                            #Dataset
│── models/                          #Models created with the codes in scripts folder, details inside
└── scripts/                         #Code running, details inside
```

## Project's Aim

The primary objective of this work is to:
- Transfer knowledge from a large CNN teacher model to a smaller dense student model
- Reduce computational cost while maintaining accuracy
- Preserve directional (cosine similarity) and magnitude features of representations across layers (Log-Magnitude)
- Achieve a balanced task performance with multi-loss optimization

## Environment
To setup the environment, run this bash code:
```bash
gh repo clone tinysystems/ENFORCE
chmod +x setup.sh
./setup.sh
```
