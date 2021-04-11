# iTest


## Description

Implement Classifier model with Auto Encoder.


### Architecture

1. Auto Encoder

Encoder -> Hidden Layer -> Decoder

2. Fully Connected Layer

Hidden Layer(from Auto Encoder) -> NN -> 10 dims outputs


Implement 2 parts, AutoEncoder part and Classification part.


### Data

CIFAR-10 Dataset

Max Size 2500 images

2. Bird
4. Deer
9. Truck

Max Size 5000 images

- Others


### Evaluation

1000 for each class.

### Results

| Algorithm | Mean | Std |
| :---: | :---: | :---: |
| Combine | 31.6 | 10.818 |
| CAE(Over Sample) | 72.8 | 0.4 |
| CAE(Under Sample) | 69.6 | 0.489 |
| CAE(2ensemble) | 75.0 | 0.632 |
| CAE(3ensemble) | 77.6 | 0.489 | 

### Pretrained Model

Pretrained model is available [here](https://drive.google.com/drive/folders/1B4mSPVkEN2pWp2hAWq9eMPJnIkr4zE9i?usp=sharing)

