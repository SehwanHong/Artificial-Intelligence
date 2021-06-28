# [MobileNets:Efficient Convolutional Neural Networks for Mobile Vision Application](https://arxiv.org/pdf/1704.04861.pdf)

MobileNets are based on streamlined architecutre that uses depthwise separable convolutions to build light wieght deep neural networks. In this paper, author introudce two global hyper parameters that efficiently trade off between latency and accuracy


# Introduction

Convolutional neural networks have become ubiquitous in computer vision after AlexNet became famous by winning the ImageNet Challenge: ILSVRC 2012. The general trend of the neural network ever since was to make deeper and more complicated networks to achive higher accuracy. However, these advances to improve accuracy had a trade off of having higher computational complexity. In real world applications, such as robotics, self-driving car and augmented reality, the recongintion tasks need to be carried out in a timely fashion of computiation.

In this paper author discribes an efficinet network architecture and a set of two hyper-parameters in order to build very small, low latency models that can be easily matched to the design requirement for mobile and embedded vision applications.

# Prior Work

There are many ways to build small and efficient neural networks. These different approaches could be generallized into two big categories:

* Compressing pretrained networks
* Training small networks

In this paper, author proposes a class of network architectures that allows model developer to specifically choose a small network that matches the resource restriction for their applications. MobileNets primarily focus on optimizing for latency but also yield small network

## Training small networks

MobileNets are build primarily from depthwise separable convolutions which is subsequently used in Inception models to reduce the computation in the first few layers. Flatten networks build a netwrok out of fully factorized convolutions and showed the potentional of extremely factorized networks. Factorized Networks introduces a small factorized convolutions as well as the use of topological connections. Xception network demonstrated how to scale up depthwise separable filters to out perform Inception V3 networks. Squeezenet uses a bottlenect approach to design a very small network. Other reduced computation networks are structured transform networks and deep fired convnets.

## Obtaining small networks by factorizing or compressing pretrained networks.

Compression based on product quantization, hashing, and pruning, vector quantization and Huffman coding have been proposed in the literature. Moreover, Various Factorization have been proposed to speed up pretrained netowrks.

Another method for traing small networks is distillations which uses a larger network to teach a smaller network.

# MobileNet Architecture

## Depthwise separable Convolution

MobileNet model is based on depthwise separable convolutions which is a form of factorized convolutions which factorize a standard convolution into a depthwise convolution and ![1 \times 1](https://latex.codecogs.com/svg.image?1\times1) convolution called pointwise convolution.

### Standard Convolution

Standard convolution both filters and combies inputs into a new set of outpus in one step. If we denote the input feature map as ![D_F \times D_F \times M](https://latex.codecogs.com/svg.image?D_F&space;\times&space;D_F&space;\times&space;M) and output feature map as ![D_F \times D_F \times N](https://latex.codecogs.com/svg.image?D_F&space;\times&space;D_F&space;\times&space;N), then we could calculate the kernal size for standard convolutional layers.

![Standard Convolutional Filters](.\standardConvFilter.png)

As shown in the image above, Standard convolutional layer have the filter size of ![D_K \times D_K \times M \times N](https://latex.codecogs.com/svg.image?D_K&space;\times&space;D_K&space;\times&space;M&space;\times&space;N). ![D_K](https://latex.codecogs.com/svg.image?D_K) is the size of the filter kernel  ![M](https://latex.codecogs.com/svg.image?M) is the size of input Channel. ![N](https://latex.codecogs.com/svg.image?N) is the size of output Channel.

Using the size of the Kernel and the size of input, we could calculate the computational cost of Standard convolutional layers.

![D_K \times D_K \times M \times N \times D_F \times D_F](https://latex.codecogs.com/svg.image?D_K&space;\times&space;D_K&space;\times&space;M&space;\times&space;N&space;\times&space;D_F&space;\times&space;D_F)

### Depthwise Separable convolution

On the other hand the depthwise separable convolution splits this into two layers:
* Depthwise convolution for filtering
* Pointwise convolution for combining

Depthwise convolution is the filtering step in depthwise separable convolution. In this step, a single filter is applied per each input channel. As shown in the image below, kernel would be ![D_K \times D_K \times M \times 1](https://latex.codecogs.com/svg.image?D_K&space;\times&space;D_K&space;\times&space;M&space;\times&space;N).

![Depthwise convolution filter](.\depthwiseConvFilter.png)

Computational cost of depthwise convolution filter is extremely small compared to standard convolution because it does not need extra parameters for filtering all input channels.

![D_K \times D_K \times M \times D_F \times D_F](https://latex.codecogs.com/svg.image?D_K&space;\times&space;D_K&space;\times&space;M&space;\times&space;N&space;\times&space;D_F&space;\times&space;D_F)

Pointwise convolution is the combining step in depthwise separable convolution. In this step, ![1 \times 1](https://latex.codecogs.com/svg.image?1\times1) convolution is applied to combine the result of depthwise convolutional layer. The kernel size of the this would be ![1 \times 1 \times M \times N](https://latex.codecogs.com/svg.image?1\times1\times&space;M&space;\times&space;N).

![Pointwise convolution filter](.\pointwiseConvFilter.png)

Computational cost of pointwise convolution filter is dependent on the input size and the output size but not dependent on the kernel size.

![M \times N \times D_F \times D_F](https://latex.codecogs.com/svg.image?M&space;\times&space;N&space;\times&space;D_F&space;\times&space;D_F)

Total Computational cost of depthwise separable convolutional layer is sum of depthwise convolutional layer and pointwise convolutional layer. Thus it would be:

![D_K \times D_K \times M \times D_F \times D_F + M \times N \times D_F \times D_F](https://latex.codecogs.com/svg.image?D_K&space;\times&space;D_K&space;\times&space;M&space;\times&space;N&space;\times&space;D_F&space;\times&space;D_F+M&space;\times&space;N&space;\times&space;D_F&space;\times&space;D_F)

### Reduction in Computation.

From both computational cost equation, we are able to find the reduction ratio in computation when using depthwise separable convolution compared to standard convolutional layer.

![computaion ratio](https://latex.codecogs.com/svg.image?\frac{D_K&space;\times&space;D_K&space;\times&space;M&space;\times&space;D_F&space;\times&space;D_F&space;&plus;&space;M&space;\times&space;N&space;\times&space;D_F&space;\times&space;D_F}{D_K&space;\times&space;D_K&space;\times&space;M&space;\times&space;N&space;\times&space;D_F&space;\times&space;D_F}&space;=&space;\frac{1}{N}&space;&plus;&space;\frac{1}{D_K^2})

Since MobileNet is uses kernel size of 3, (![3\times3](https://latex.codecogs.com/svg.image?3\times3)), MobileNet uses 8 to 9 times less computaion than using standard convolutions.

## Network Structure and Training

MobileNet structure is built on depthwise separable convolutions except for the first layer which is a full convolution. All layers are followed by a batch normalization and ReLU non-lineality with the exception of the final fully convolutional layer which has no nonlinearity and feeds into a softmax layer for classification. Downsampling is handled with strided convolution in the depthwise convolutions as well as in the first layer. A final average pooling reduces spatial resolution into 1 before the fully connected layer. Counting depthwise and pointwise convolutions as separate layers, MobileNet has 28 layers.

Standard Convolutional layer | Depthwise Separable Convolutional Layer
-----------|-----------
![standard convolutional layer](.\standardConvLayer.png) | ![depth wise separable convolutional layer](.\depthwiseConvLayer.png)

The image above represent how the layers in the standard convolution and depthwise separable convolutional layer is different. Standard convolution, as described in the Depthwise separable convolution section, uses one large convolutional filter for all output dimensions. However, depthwise separable convolution uses depthwise convolution for filtering the input image by channel wise, then use pointwise convoluiton for combining the layers.


