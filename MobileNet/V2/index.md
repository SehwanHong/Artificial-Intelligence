# [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/pdf/1801.04381.pdf)

In this paper, author describes a new mobile architecture, MobileNetV2, that improves the performance of mobile models on multiple tasks and benchmarks.

# Introduction

Neural networks have revolutionized many areas of machine intelligences, enabling superhuman accuracy for challenging image recognition tasks. The drive to improve accuracy often comes at a cost: modern state of art networks require high computational resources beyond the capabilities of many mobile and embedded applications.

This paper introduces a new neural network architecture that is specifically tailored for mobile and resource constrained environments. To retain the same accuracy while decreasing the number of operations and memory usage, author introcues a novel layer module: inverted residual with linear bottleneck.

# Related Work

There has been a lot of progress in algorithmic architecture exploration including hyperparameter optimization, various methods of network pruning, and connectivity learning. There are different work to change the connetivity structure of the internal convolutional blocks.

The new direction of bringing optimization methods include genetic algorithms and reinforcement learning to architectural search. However, drawback of these methodes is that the resulting network end up very complex.

The network presented in this paper is designed based on [MobileNetV1](../). 

# Preliminaries, discussion and intuition

## Depthwise separable convolution

Depthwise Separable Convolutions are a key building block for many efficient neural network architectures. The basic idea is to replace a full convolutional operator with a factorized version that split convolution into two separate layers.

  1. Depthwise convolution
  2. pointwise convolution

For more detail, follow [this link](../)

## Linear Bottlenecks

Consider a deep neural network consisting of n layers ![L_i](https://latex.codecogs.com/svg.image?L_i) each of which has an activation tensor of dimensions ![tensor dimension](https://latex.codecogs.com/svg.image?h_i\times&space;w_i\times&space;d_i). The set of layer activations(for any layer ![L_i](https://latex.codecogs.com/svg.image?L_i)) forms a "manifold of interest" which could be embedded in low-dimensional subspaces. In other words, the information encoded in d-channel pixels of a deep convolutional layer actually lie in some manifold, which in turn is embeddable into a low-dimensional subspace.

In general if a result of a layer transformation ![ReLU(Bx)](https://latex.codecogs.com/svg.image?ReLU(Bx)) has a non-zero volume ![S](https://latex.codecogs.com/svg.image?S), the points mapped to interior ![S](https://latex.codecogs.com/svg.image?S) are obtained via a linear transformation ![B](https://latex.codecogs.com/svg.image?B) of the input is limited to a linear transformation. In other words, deep networks only have the power of a linear classifier on the non-zero volume part of the output domain.

On the other hand, when ReLU collapses the channel, it is inevitabley loses information in that channel. However, if we have lots of channels, there is a structure in the activation manifold that information might still be preserved in the other channel. The bottom image represent this example.

![ReLU transformations of low-dimensional manifold embedded in higher-dimensional spaces](./ReLUtransformation.png)

In these example, the initial spiral is embedded into an n-dimensional space using random matrix ![T](https://latex.codecogs.com/svg.image?T) followed by ReLU, and then projected back to the 2d space using ![inverse of T](https://latex.codecogs.com/svg.image?T^{-1}). When ![n = 2,3](https://latex.codecogs.com/svg.image?n=2,3), there is an information loss where certain point of the manifold collaps into each other. While for ![n = 15](https://latex.codecogs.com/svg.image?n=15) to 30, the transformation is highly non-convex.

To summarize, there are two properties that are indicative of the requirement that the manifold of interest should lie in a low-dimensional subspace of the higher-dimensional activation space:

1. If the manifold of interest remains non-zero volume after ReLU transformation, it corresponds to a linear transformation.
2. ReLU is capable of preserving complete information about the input manifold, but only if the input manifold lies in a low-dimensional subspace of the input space.

Assuming the manifold of interest is low-dimensional, by inserting linear bottleneck layer into the convolutional blocks. Through experiemtn, using linear layer is crucial as it prevents non-linearities from desctorying too much information.

## Inverted residuals

Inspired by the intuition that the bottlenecks actually contain all the necessary information, while an expansion layer acts merely as an implementation detail that accompanies a non-linear trasnformation of the tensor, uses shortcuts directly between the bottlenecks.

Residual block | Inverted Residual Block
--------------|---------------
![Residual Block](./residualBlock.png) | ![Inverted Residual Block](./invertedResidualBlock.png)

Residual block is normally represented as the left image. It is represented with wide -> narrow -> wide, creating a bottleneck structure. However, in this paper, author presents inverted residual where structure is narrow -> wide -> narrow. The diagonally hatched layer do not use non-linearlities to reserve the information loss by using non-linearlity.

The use of shortcut in the inverted residual block is same as the [ResNet](../ResNet/) to improve the ability of a gradient to propagate across multiplier layers.

Inverted Residual block uses less memory as well as having improved performance.

### Running time and parameter count for bottleneck convolution

![bottleneck residual block](./bottleneckResidualBlock.png)

Above table represent the basic implementation structure of inverse Residual Block. For block size ![h w](https://latex.codecogs.com/svg.image?h\times&space;w), expansion factor ![t](https://latex.codecogs.com/svg.image?t) and kernel size ![k](https://latex.codecogs.com/svg.image?k) with ![d'](https://latex.codecogs.com/svg.image?d') input channels and ![d''](https://latex.codecogs.com/svg.image?d''), total number of multiply add required is

![complexity for bottlencck residual block](https://latex.codecogs.com/svg.image?h&space;\times&space;w&space;\times&space;t&space;\times&space;d'&space;\times&space;d'&space;&plus;&space;h&space;\times&space;w&space;\times&space;t&space;\times&space;d'&space;\times&space;k&space;\times&space;k&space;&plus;h&space;\times&space;w&space;\times&space;t&space;\times&space;d'&space;\times&space;d''&space;=&space;h&space;\times&space;w&space;\times&space;t&space;\times&space;d'&space;\times&space;(d'&space;&plus;&space;k^2&space;&plus;&space;d''))

This number is higher than Depthwise Separable Convolution([described in this page](../)) because of extra layer of ![1 by 1](https://latex.codecogs.com/svg.image?1\times1) convolution. However, using bottleneck residual block is more compact because of smaller input and output dimensions.


# Model Architecture

# Implementation Notes

## Memory efficient inference

# Experiments

## ImageNet Classification

## Object Detection

## Semantic Segmentation

## Ablation Study

## [Link to Neural Net](../../)
## [Link to MobileNet](../)