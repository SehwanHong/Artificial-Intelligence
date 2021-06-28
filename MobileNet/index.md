# [MobileNets:Efficient Convolutional Neural Networks for Mobile Vision Application](https://arxiv.org/pdf/1704.04861.pdf)

MobileNets are based on streamlined architecutre that uses depthwise separable convolutions to build light wieght deep neural networks. In this paper, author introudce two global hyper parameters that efficiently trade off between latency and accuracy


# Introduction

Convolutional neural networks have become ubiquitous in computer vision after AlexNet became famous by winning the ImageNet Challenge: ILSVRC 2012. The general trend of the neural network ever since was to make deeper and more complicated networks to achive higher accuracy. However, these advances to improve accuracy had a trade off of having higher computational complexity. In real world applications, such as robotics, self-driving car and augmented reality, the recongintion tasks need to be carried out in a timely fashion of computiation.

In this paper author discribes an efficinet network architecture and a set of two hyper-parameters in order to build very small, low latency models that can be easily matched to the design requirement for mobile and embedded vision applications.