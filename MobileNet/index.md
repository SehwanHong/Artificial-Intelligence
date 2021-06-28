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

Another method for traing small networks is distillations which uses a larger network to teach a smaller netwrok.