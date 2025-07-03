---
layout: default
title: Resnet V2
tags:
  - ToNN
---
# [Identity Mapping in Deep Residual Networks](https://arxiv.org/pdf/1603.05027.pdf)

In this paper, author analyzes the propagation formulation behind the residual building blocks. Also author talks about different types of skip connections including identity mapping and after-addition activation.

# Analysis of Deep Residual Networks.

The original Residual Unit in [\[1\]](https://arxiv.org/pdf/1512.03385.pdf) performs following computation

![Residual Block Equation](/assets/images/ToNN/ResNet/V2/residualblockequation.PNG)

Here ![x_l](https://latex.codecogs.com/svg.image?x_l) represents the input feature to the l-th Residual Unit. Also ![W_l](https://latex.codecogs.com/svg.image?W_l) is a set of weights and biases associated with the l-th Residual Unit, where it could have up to K number of layers.

If we set the function h as an identity mapping and function f also as an identity mapping, then we could assume following equation.

![nextLayer](/assets/images/ToNN/ResNet/V2/nextLayer.png)

Recusively implimenting following equation from k = 1 to L, we will have:

![recursiveEquation](/assets/images/ToNN/ResNet/V2/recursiveEquation.png)

This equation have nice properties in understanding residual networks.

1. For any feature in deeper unit L, could be preseneted as the feature of shallower unit plus the resdual functions in form or summations
2. The feature of any deeper unit L, is the summation of the outputs of all preceding residual functions.

This recursing equation leads to nice backward propagation properties. The gradient of the equation above would be following.

![gradient](/assets/images/ToNN/ResNet/V2/gradient.png)

This equation implies that the gradient of a layer does not vanish even when the weights are arbitrarily small.

# On the Importance of Identity Skip Connections

Let's modify the function h, ![function h](https://latex.codecogs.com/svg.image?h(x_{l&plus;1})=\lambda_lx_l) to break the identity shortcuts.

![equation6](/assets/images/ToNN/ResNet/V2/equation6.png)

The value ![lambda](https://latex.codecogs.com/svg.image?%5Clambda_l) is a modulating scalar.

Recursively implimenting this fomular re could get equation similar to equation 4 presented above.

![equation7](/assets/images/ToNN/ResNet/V2/equation7.png)

Similar to equation 5, we could get back propagation of following:

![equation8](/assets/images/ToNN/ResNet/V2/equation8.png)

Unlike Identity(equation5), equation 8 have a term ![product term](https://latex.codecogs.com/svg.image?\prod_{i=l}^{L-1}{\lambda_i}). If ![greather than](https://latex.codecogs.com/svg.image?%5Clambda_i%3E1), the product term would have exponentially large value. If ![less then](https://latex.codecogs.com/svg.image?\lambda_i<1), then the product term is exponentially small and vanish.

Thus if the layer is large, then using weighted value for shortcut would cause information propagation and impede the training procedure.

## Experiment on Skip Connections

Looking into above equations, as the layer increases, using skip connection that is not identity matrics would suffer decrease in the training error rate. Thus, author of the paper presents different shortcut variations.

![Differnt types of shortcuts](/assets/images/ToNN/ResNet/V2/DifferentTypeOfShortcut.png)

Using different shortcut methods, Author examines difference between the skip connection methods and test error. In this skip connections, a) original is from the paper Deep Residual Learning for Image Recongition, where using identity shortcut and zero padding for extanded dimensions. b) constant scaling is multiplying floats between 0 to 1 for skip connection and for residual functions. c) is exclusive gating which is adopted from Highway Net where gating funciton is ![1x1](https://latex.codecogs.com/svg.image?1\times1) convolutional layer and applying sigmoid as activation function. In this exclusive gating, we use ![1-g(x)](https://latex.codecogs.com/svg.image?1-g(x)) for shortcut connections and ![g(x)](https://latex.codecogs.com/svg.image?g(x)) for residual function. d) shortcut only gating is similar to exclusive gating but only eliminating gating function on the residual function. e) is using ![1x1](https://latex.codecogs.com/svg.image?1\times1) convolutional layer as a shortcut. Final, for f) adding dropout layer for shortcut connection.

![Skip conncetion and their result](/assets/images/ToNN/ResNet/V2/SkipConnectionAndResult.png)

As shown in this Table, addind different layers to Skip Connection reports higher error rate compared to the identity mapping. Therefore, identity mapping is the best way to use the skip connections.

# On the Usage of Activation Function.

Experiment above talks about different techniques in Skip Connection. In this section, author moves attention from skip Connection to activation function. Finding the best order for activation function.

Author want to kind different arrangement of activation function that would increase the accuracy.

## Experiment on activation

![Experiment On Activation](/assets/images/ToNN/ResNet/V2/ExperimentOnActivation.png)

Above image present different activation methods and their error rate using CIFAR 10 data and ResNet 110 and ResNet 164. ResNet 110 uses two ![3 x 3](https://latex.codecogs.com/svg.image?3\times3) convolutional layers. On the other hand, ResNet 164 substitute two ![3 x 3](https://latex.codecogs.com/svg.image?3\times3) convolutional layers with ![1 x 1](https://latex.codecogs.com/svg.image?1\times1) convolutional layer and ![3 x 3](https://latex.codecogs.com/svg.image?3\times3) convolutional layer and ![1 x 1](https://latex.codecogs.com/svg.image?1\times1) convolutional layer.

ResNet 110 Residual Unit | ResNet 164 Residual Unit
:-----------------------:|:---------------------------:
![ResNet110](/assets/images/ToNN/ResNet/V2/ResNet110.png) | ![ResNet164](/assets/images/ToNN/ResNet/V2/ResNet164.png)

Left image present the Residual Unti of ResNet 110, and Right image represent the Residual Unit of ResNet 164. Both ResNet110 and ResNet 164 have same 18 blocks of residual Unit but ResNet 164 have more layers since ResNet 164 have 3 Layers inside Residual Unit while Resnet 110 have only 1 layers.

As above image displays, using preactivation have marginal increase in Test error.

Since the experiment result for preactivation was good, thus increaseing the layers over 1000 to find the benefit of using preactivation.

![Table 3](/assets/images/ToNN/ResNet/V2/Table3.png)

Looking at the result of using preactivation residual unit, we could find that using preactivation unit gives marginal benefit in lowering classification error.

## Analysis

Author have found two beneifts when applying preactivation.

### Ease of optimization

The first benefit is ease of optimization. This effect was largely visible when using ResNet 1001. As table 3 present, using original residual unit validation classificaion error is higher than ResNet that have smaller layers. However, when using preactivation unit, we could see the benefit of using more layers and gives better result than using less layers.

### Reducing overfitting.

Another impact of using preactivation unit is on regularization. Using original have problem when normalziation. After adding to the shortcut, the result is not normalized. On the contrary, preactivation version, inputs to all weight layers have been normalized.

## [Link to Neural Net](../../)
## [Link to Original Version](../)
## [Link to ResNext](../ResNext/)
## [Link to Korean Version](../V2)