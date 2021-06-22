# [Identity Mapping in Deep Residual Networks](https://arxiv.org/pdf/1603.05027.pdf)

이 논문에서 저자는 Residual block의 propagation formulation을 분석했습니다. 또한 Skip connection을 하는 다양한 방식들을 분석했습다. 또한 다양한 activation 방식을 통해서 차이점들을 확인했습니다.

# Analysis of Deep Residual Networks.

논문 [\[1\]](https://arxiv.org/pdf/1512.03385.pdf)에서 표현한 Residual Unit은 아래와 같은 수식을 표현할 수 있습니다.

![Residual Block Equation](../../V2/residualblockequation.PNG)

여기서 ![x_l](https://latex.codecogs.com/svg.image?x_l)은 Residual Unit에서 l번째 입력값입니다. ![W_l](https://latex.codecogs.com/svg.image?W_l)는  l번째 Residual Unit와 연관된 모든 weights와 bias값의 총 집합입니다. 이때 하나의 Residual Unit은 K 겟수의 layer들을 가질수 있습니다.

만약 우리가 function h를 identity mapping 이라고 가정하고 function f 또한 identity mapping 가정한다면, 아래와 같은 수식으로 변환할 수 있습니다.

![nextLayer](../../V2/nextLayer.png)

이 수식을 k = 1 부터 L 제귀적으로 합친다면, 우리는 아래와 같은 수식으로 바꿀 수 있습니다:

![recursiveEquation](../../V2/recursiveEquation.png)

이 수식을 통해서 Residual Network에 관한 여러가지 특성들을 알 수 있습니다.

1. 깊은 Residual Unit L은 얕은 Residual unit들과 Residual Functions들의 합으로 표현될 수 있다.
2. 어떠한 깊은 unit L은  unit L 전에 있는 모든 residual function의 출력 값들의 합으로 표현할 수 있다.

위의 수식 4 번을 통해서 backpropagation 특성들을 확인 할 수 있습니다. Back propagation 하기 전에 graident equation을 살펴보면 아래와 같습니다.

![gradient](../../V2/gradient.png)

이 수식을 통해서 function F가 -1이 되지 않는 이상, graident는 항상 1 이상을 유지합니다. 그럼으로 gradient vanishing 효과가 나타나지 않습니다.

# On the Importance of Identity Skip Connections

Let's modify the function h, ![function h](https://latex.codecogs.com/svg.image?h(x_{l&plus;1})=\lambda_lx_l) to break the identity shortcuts.

![equation6](../../V2/equation6.png)

The value ![lambda](https://latex.codecogs.com/svg.image?%5Clambda_l) is a modulating scalar.

Recursively implimenting this fomular re could get equation similar to equation 4 presented above.

![equation7](../../V2/equation7.png)

Similar to equation 5, we could get back propagation of following:

![equation8](../../V2/equation8.png)

Unlike Identity(equation5), equation 8 have a term ![product term](https://latex.codecogs.com/svg.image?\prod_{i=l}^{L-1}{\lambda_i}). If ![greather than](https://latex.codecogs.com/svg.image?%5Clambda_i%3E1), the product term would have exponentially large value. If ![less then](https://latex.codecogs.com/svg.image?\lambda_i<1), then the product term is exponentially small and vanish.

Thus if the layer is large, then using weighted value for shortcut would cause information propagation and impede the training procedure.

## Experiment on Skip Connections

Looking into above equations, as the layer increases, using skip connection that is not identity matrics would suffer decrease in the training error rate. Thus, author of the paper presents different shortcut variations.

![Differnt types of shortcuts](../../V2/DifferentTypeOfShortcut.png)

Using different shortcut methods, Author examines difference between the skip connection methods and test error.

![Skip conncetion and their result](../../V2/SkipConnectionAndResult.png)

As shown in this Table, addind different layers to Skip Connection reports higher error rate compared to the identity mapping. Therefore, identity mapping is the best way to use the skip connections.

# On the Usage of Activation Function.

Experiment above talks about different techniques in Skip Connection. In this section, author moves attention from skip Connection to activation function. Finding the best order for activation function.

Author want to kind different arrangement of activation function that would increase the accuracy.

## Experiment on activation

![Experiment On Activation](../../V2/ExperimentOnActivation.png)

Above image present different activation methods and their error rate using CIFAR 10 data and ResNet 110 and ResNet 164. ResNet 110 uses two ![3 x 3](https://latex.codecogs.com/svg.image?3\times3) convolutional layers. On the other hand, ResNet 164 substitute two ![3 x 3](https://latex.codecogs.com/svg.image?3\times3) convolutional layers with ![1 x 1](https://latex.codecogs.com/svg.image?1\times1) convolutional layer and ![3 x 3](https://latex.codecogs.com/svg.image?3\times3) convolutional layer and ![1 x 1](https://latex.codecogs.com/svg.image?1\times1) convolutional layer.

ResNet 110 Residual Unit | ResNet 164 Residual Unit
:-----------------------:|:---------------------------:
![ResNet110](../../V2/ResNet110.png) | ![ResNet164](../../V2/ResNet164.png)

Left image present the Residual Unti of ResNet 110, and Right image represent the Residual Unit of ResNet 164. Both ResNet110 and ResNet 164 have same 18 blocks of residual Unit but ResNet 164 have more layers since ResNet 164 have 3 Layers inside Residual Unit while Resnet 110 ahve only 1 layers.

As above image displays, using preactivation have marginal increase in Test error.
