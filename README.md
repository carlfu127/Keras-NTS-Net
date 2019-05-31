# Keras-NTS-Net
This is an implementation of NTS-Net(https://arxiv.org/pdf/1809.00287.pdf) on Python 3, Keras, and TensorFlow.

## Requirements
- python 3+
- keras 2.2.4+
- tensorflow-gpu 1.9+
- numpy
- opencv-python 3.4+
- datetime

## Datasets
Download the [CUB-200-2011](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz) datasets and put it in the dataset directory named **CUB_200_2011**, You can also try other fine-grained datasets.

## Train the model
You may need to change the configurations in config.py. 

## Acknowledgement
Original implementation
[NTS-Net](https://github.com/yangze0930/NTS-Net)

Third Party Libs
[NTS-Net-keras](https://github.com/He-Jian/NTS-Net-keras)
