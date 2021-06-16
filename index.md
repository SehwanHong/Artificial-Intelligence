# Residual neural network

## [Deep residual learning for image recognition](https://arxiv.org/pdf/1512.03385.pdf)

Author of the paper [Deep residual learning for image recognition](https://arxiv.org/pdf/1512.03385.pdf) have introduced Residual Unit because deeper neural networks are more difficult to train. Thus presents a new framwork that makes training easier which is Residual Unit.

### Why residual Network

When deeper network converges, degradtion problem is emarged. In other word, gradient vanishing problem occurs. As layer increases the derivitive of the value decreases significantly, thus output has less or sometime have no effect on the weights.

Degradation is not caused by overfitting, and using more layers leads to higher training error.

### What is Deep Residual Learning.

#### Residual Learning.

Let us consider $H(x)$ as underlying mapping to be fit by a few stacked layers. $x$ is 
