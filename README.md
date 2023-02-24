# Semantic Segmentation for Hyperspectral Images using Tridimensional ResNet


### Abstract

Convolutional Neural Networks (CNNs) are widely used in hyperspectral images (HSIs) processing, as well as in image classification, remote sensing data processing and object detection. ResNet is an existing and efficient CNN model that will be used in this paper for experimentation. When it comes to HSI classification, not only 2D-CNNs are used, but it results in a good idea to use 3D-CNNs that evaluate the third dimension alongside the first and the second ones, in order to obtain a better performance in training and a higher accuracy in validation of hyperspectral images. In this paper, we propose a change of the current networks that evaluate the AeroRIT dataset. Such change consists in the transformation of the 2D networks in 3D networks, and a further evaluation of the results obtained. However, in general terms, CNN networks have problems when it comes to processing images with low resolution or that may contain small objects. This results in a significant information loss. To approach this problem, Sunkara and Luo proposed a new convolutional block that replaces the strided and pooling layers inside a network, so that information loss is not as remarkable as previously. We also propose an adaptation of that idea that can be implemented inside tridimensional neural networks. This new block will replace convolutional layers with any stride value, including those strides that have distinct values in the three different dimensions.


## Dataset

The AeroRIT repository can be found [here](https://github.com/aneesh3108/AeroRIT).
The scene images can be found [here](https://drive.google.com/drive/folders/1yCMqa9uDC_CEGtbnxeWEQCTb-odC2r4c?usp=sharing).
The directory contains four files: 
1. image_rgb - The RGB rectified hyperspectral scene.
2. image_hsi_radiance - Radiance calibrated hyperspectral scene.
3. image_hsi_reflectance - Reflectance calibrated hyperspectral scene.
4. image_labels - Semantic labels for the entire scene.


## Executions

To obtain train, validation and test splits with 64 x 64 image chips, execute the [sampling_data.py](/sampling_data.py/) file:

```
  python sampling_data.py
 ```

To execute the training and validation of a network, run the [train](/train.py/) file as it follows:

```
  python3 train.py --network_arch SSP --batch-size 64 --ngf 8
```
Main arguments used in [train](/train.py/):

| Argument | Details |
| -- | -- |
| network_arch | Network architecture to use: resnet, SSP, NoSSP |
| batch-size | Number of images sampled per minibatch |
| ngf | Number of filters: 4, 8, 16, 32, 64 |
