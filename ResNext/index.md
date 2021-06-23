# [Aggregated Residual Transformations for Deep Neural Network](https://arxiv.org/pdf/1611.05431.pdf)

Author of this paper wants to talk about redesigning residual block using multi-branch architecture so called aggregated Residual Block. This strategy creates a new dimension which the authors call as "cardinality".

# Introduction

Current research on visual recognition is transitioning from feature engineering to network engineering, where features are learned by neural networks from large-scale data.

Designing better network architecture becomes increasingly difficult with growing number of hyper-parameters(such as width, filter sizes, strides, ets) with larger numbers of layers. Like VGG nets, which use simple but efficient strategy, ResNets inherit their techniques of stacking building blocks of same shapes.

This paper adopts VGG/ResNet's strategy of repeating layers, while implimenting new techniques of split-transfrom-merge strategy. This strategy appears similar to the Inception-ResNet Module in concatenating multiple paths, but differes from all existing inception modules that this architecture share the same topology and thus easily isolated as a factor to be investigated.

Author emphasice the new method called cardinality which is the size of the set of trasnformations. Through experiment described below, authors indiation that increasing cardinality is a more effective way of gaining accuracy than going deeper or wider.

# Related work
## Multibranch convolutional networks
## Grouped convolutions
## Compressing convolutional networks
## Ensembling

# Method
## Tempalate



# Reference

https://takenotesforvision.tistory.com/12
