# [Aggregated Residual Transformations for Deep Neural Network](https://arxiv.org/pdf/1611.05431.pdf)

Author of this paper wants to talk about redesigning residual block using multi-branch architecture so called aggregated Residual Block. This strategy creates a new dimension which the authors call as "cardinality".

# Introduction

Current research on visual recognition is transitioning from feature engineering to network engineering, where features are learned by neural networks from large-scale data.

Designing better network architecture becomes increasingly difficult with growing number of hyper-parameters(such as width, filter sizes, strides, ets) with larger numbers of layers. Like VGG nets, which use simple but efficient strategy, ResNets inherit their techniques of stacking building blocks of same shapes.

This paper adopts VGG/ResNet's strategy of repeating layers, while implimenting new techniques of split-transfrom-merge strategy. This strategy appears similar to the Inception-ResNet Module in concatenating multiple paths, but differes from all existing inception modules that this architecture, named ResNeXt, shares the same topology and thus easily isolated as a factor to be investigated.

Author emphasice the new method called cardinality which is the size of the set of trasnformations. Through experiment described below, authors indiation that increasing cardinality is a more effective way of gaining accuracy than going deeper or wider.

# Related work
## Multibranch convolutional networks
## Grouped convolutions
## Compressing convolutional networks
## Ensembling

# Method
## Tempalate

Design of ResNeXt is adopted from highly modularized design following VGG/ResNets. This network consist of a stack of residual blocks, which have same topology and are subject to two simple rules.

 1. For the same output feature map size, the layers have the same number of filters.
 2. If the feature map size is halved, the number of filter is doubled so as to preserve the time complexity per layer

With these two rules, authors designed template modules and all modules in a newtork can be determined accordingly.

## Revisiting Simple Neurons.

The simplest neurons in artificial neural networks perform inner product(weighted sum), which is the elementary transformation done by fully connected nad convolutional layers.

![Inner Product](.\InnerProduct.png)

Inner product equation, presented above, is similar to aggregated transformation shown below.

![Aggregated Transformation](.\AggregatedTrasnformation.png)

Where ![x=[x_1,x_2,...,x_D]](https://latex.codecogs.com/svg.image?x=[x_1,x_2,...,x_D]) is a D-Channel input vector to the neuron and ![w_i](https://latex.codecogs.com/svg.image?w_i) is a filter's weight for the i-th Channel. This equation could be represented as a figure below

![neuron](.\neuron.png)

The operations to build a neuron could be splitted in to three operations:

1. *Spliting*: the vector x is sliced as a low-dimensional subspace ![x_i](https://latex.codecogs.com/svg.image?x_i)
2. *Transforming* : the low-dimensional representation is transformed, ![w_i x_i](https://latex.codecogs.com/svg.image?w_ix_i)
3. *Aggregating* : the transformations in all embedding are aggregated by ![Aggregation](https://latex.codecogs.com/svg.image?\inline\sum_{i=1}^{D})

## Aggregated Transformations

Giving above analysis of simple neuron, Aggregated Transformation will be formally presented as equation below.

![Formal aggregated transformation equation](.\formalAggregatedTransformEquation.png)

# Reference

https://takenotesforvision.tistory.com/12
