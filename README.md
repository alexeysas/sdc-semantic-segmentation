# Semantic Segmentation
### Introduction
The goal of this project is to label the pixels of a road in images using a Fully Convolutional Network (FCN).

[//]: # (Image References)

[image1]: ./images/architecture.png "Architecture"
[image2]: ./runs/1525041123.206333/um_000060.png "Image1"
[image3]: ./runs/1525041123.206333/um_000039.png "Image2"
[image4]: ./runs/1525041123.206333/um_000093.png "Image3"

6

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


### Training

Training is a standard Tensorflow training cycle with 50 epochs and batch size 4.  Setting bigger batch size causes memory issues with GPU.
Use Adam optimizer (tried RMSPROP as well - ADAM provides better results). Additionally, used L2 regularizer as it provided better results as well.

Epoch 47
Loss:     0.0106 IOU: 0.457153
Epoch 48
Loss:     0.0194 IOU: 0.460766
Epoch 49
Loss:     0.0296 IOU: 0.464164
Epoch 50
Loss:     0.0109 IOU: 0.467661
Training Completed.
Training Finished. Saving test images to: ./runs\1525041123.2063336

### Inference samples

![alt text][image2]
![alt text][image3]
![alt text][image4]
