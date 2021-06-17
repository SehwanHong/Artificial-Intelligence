# [Identity Mapping in Deep Residual Networks](https://arxiv.org/pdf/1603.05027.pdf)

In this paper, author analyzes the propagation formulation behind the residual building blocks. Also author talks about different types of skip connections including identity mapping and after-addition activation.

# Analysis of Deep Residual Networks.

The original Residual Unit in [\[1\]](https://arxiv.org/pdf/1512.03385.pdf) performs following computation

![Residual Block Equation](residualblockequation.png)

Here ![x_l](https://latex.codecogs.com/svg.image?x_l) represents the input feature to the l-th Residual Unit. Also ![W_l](https://latex.codecogs.com/svg.image?W_l) is a set of weights and biases associated with the l-th Residual Unit, where it could have up to K number of layers.
