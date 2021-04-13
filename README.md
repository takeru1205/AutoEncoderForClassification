# iTest


## Description

Implement Classifier model with Auto Encoder.

## Usage
```Shell
$sudo docker build -t ae_classify .

$sudo docker run -it --gpus all ae_classify /bin/bash

# in the container below

# Train and evaluate
python3 main.py

# If want to save model weights, use save option
python3 main.py --save

# Only test with trained model
python3 test.py
```


### Architecture

1. Auto Encoder

Encoder -> Hidden Layer -> Decoder

2. Fully Connected Layer

Hidden Layer(from Auto Encoder) -> NN -> 10 dims outputs


Implement 2 parts, AutoEncoder part and Classification part.


### Data

CIFAR-10 Dataset

Max Size 2500 images

  2.Bird
  4.Deer
  9.Truck

Max Size 5000 images

- Others


### Evaluation

1000 for each class.

### Results

| Algorithm | Mean | Std |
| :---: | :---: | :---: |
| Simple CAE | 31.6 | 10.818 |
| Over Sampled CAE | 72.8 | 0.4 |
| Under Sampled CAE | 69.6 | 0.489 |
| 2 Ensemble Under Sampled CAE | 75.0 | 0.632 |
| 3 Ensemble Over and Under Sampled CAE | 77.6 | 0.489 | 

### Pretrained Model

Pretrained model is available [here](https://drive.google.com/drive/folders/1B4mSPVkEN2pWp2hAWq9eMPJnIkr4zE9i?usp=sharing)

