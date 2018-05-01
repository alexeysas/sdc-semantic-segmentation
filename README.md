# Semantic Segmentation
### Introduction
The goal of this project is to label the pixels of a road in images using a Fully Convolutional Network (FCN).

[//]: # (Image References)

[image1]: ./images/architecture.png "Architecture"

### Setup

##### System

FCN requires GPU to train efficiently. Following system setup used:

 - Physical GPU (device: 0, name: GeForce GTX 1080 Ti with 8723 MB memory, compute capability: 6.1)
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)

##### Dataset
[Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip) is used for training.

### Architecture

The project replicates FCN8 architecture described in the https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf

![alt text][image1]

