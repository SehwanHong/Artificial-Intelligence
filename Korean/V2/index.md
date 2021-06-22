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

위의 분석에서는 function H 값을 identity로 추정한 상태에서의 해석입니다. function h를 ![function h](https://latex.codecogs.com/svg.image?h(x_{l&plus;1})=\lambda_lx_l)  로 바꾸어서 identity shortcut을 없엤다면 아래와 같은 수식이 나옵니다.

![equation6](../../V2/equation6.png)

여기서 ![lambda](https://latex.codecogs.com/svg.image?%5Clambda_l)은 Shortcut을 변환하는 Scalar 값입니다.

이 함수는 수식 4번과 같이 제귀적으로 표현한다면, 아래와 같은 수식이 나오게됩니다.

![equation7](../../V2/equation7.png)

수식 5 번을 구한것처럼 backpropagation equation을 찾아보면 아래와 같이 나옵니다.

![equation8](../../V2/equation8.png)

Identity를 사용한 수식 5와는 다르게, 수식 8은 ![product term](https://latex.codecogs.com/svg.image?\prod_{i=l}^{L-1}{\lambda_i})항이 존재합니다. 이때 ![greather than](https://latex.codecogs.com/svg.image?%5Clambda_i%3E1)라면, 저 곱셈항이 기하급수적으로 증가하게 됩니다. 그와 반대로 ![less then](https://latex.codecogs.com/svg.image?\lambda_i<1)라면, 기하급수적으로 작아지게 되어 graident vanishing효과가 나타나게 됩니다.

그럼으로 만약 레이어의 수가 증가한다면, shortcut을 사용할 때 추가적은 weight를 집어넣는 것이 weight 값들에게 information propagation을 막고 training을 방해합니다.

## Experiment on Skip Connections

위의 수식을 통해서, layer의 수가 증가할수록, identity를 제외한 skip connection 사용시에 training error의 상승하는 문제점이 생깁니다. 그럼으로, 이 논문의 저자는 다양한 shortcut을 통해서 이를 증명하려고 합니다.

![Differnt types of shortcuts](../../V2/DifferentTypeOfShortcut.png)

위에 보시는 이미지에서 처럼 다양한 shortcut을 실험합니다. 여기서 a) original은 [이전](../) 논문에서 사용한 identity를 의미합니다. b) constant Scaling은 shortcut 이나 Residual function 값에 실수 값을 곱해주는 것을 의미합니다. c) exclusive gating은 Highway Network에서 사용한 gating mechanism을 따른 것입니다. Convolutional network에서 gating function은 ![1x1](https://latex.codecogs.com/svg.image?1\times1) convolutional layer에 sigmoid 함수를 적용한 것입니다. 여기서, Shortcut에는 ![1-g(x)](https://latex.codecogs.com/svg.image?1-g(x))를 residual function에는 ![g(x)](https://latex.codecogs.com/svg.image?g(x))를 적용한 것을 의미합니다. d) shortcut-only gating은 c)의 exclusive gating에서 residual function에 적용한 g(x)를 없엔 것입니다. e)의 convolutional shortcut은 [\[1\]](../) 논문에서 적용해보았던 option C를 크게 적용해본것 입니다. f)의 dropout shortcut은 shortcut에 dropout layer를 적용해본 것입니다.

![Skip conncetion and their result](../../V2/SkipConnectionAndResult.png)

위의 Table 1에서의 결과를 보면, 가장 기본적인 orignal layer가 가장 좋은 결과를 자지고 왔고, 나머지는 이론적으로 계산한것과 비슷하게 shortcut에 weight를 적용할 경우 더 안좋은 결과를 냈습니다.

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

## [Link to English Version](../../V2/)
## [Link to Original Version](../)