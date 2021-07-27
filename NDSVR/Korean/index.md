# [On Network Design Spaces for Visual Recognition](https://arxiv.org/pdf/1905.13214.pdf)

이 논문에서 저자는 인공신경망의 성능을 비교하는 새로운 방식을 소개합니다. 전에 사용하던 point와 curve estimate 와는 다르게, distribution estimate은 model family의 설계에 대한 더 완전한 모습을 보여준다.

# Introduction

최근 논문들은 실증적 조사를 통해서 인공신경망의 구조에 대한 연구를 합니다. 이들은 더욱 더 나은 관찰 방식을 찾기 위해서 노력합니다.

![Comparing Network using different estimations](../ComparingNetworks.png)

초기 연구단계에서는 간단한 방식을 사용했습니다. 인공신경망의 성능을 간단한 point estimate로 계산했습니다. 만약 새로운 인공신경망의 error가 다른 benchmark dataset에서 더 낮게 나오면, 이 인공신경망의 성능이 인공신경망의 복잡도에 관계없이 더 좋다고 여겼습니다.

최근에는 더 향상된 방식인 Curve Estimate를 사용합니다. 이 방식은 다양한 design tradeoff에 대해서 탐구합니다. 이때 다양한 handful of models from a loosely defined model familes를 예를 들어 설명합니다. 그리고 error와 model complexity를 사용해서 선을 그립니다. 이를 통해서 하나의 model family가 다른 model family와 비교해서 전체적으로 낮은 error를 가지고 있다면 더 우수하다고 생각합니다. 위의 그래프를 확인하면, ResNeXt가 ResNet보다 더 좋다고 여겨집니다. 왜냐하면, ResNeXt가 전체적으로 낮은 error를 가지고 있기 때문입니다.

하지만 이러한 방식에도 단점은 있습니다. Curve estimate는 다른 confounding factor를 고려하지 않습니다. Confounding factor가 model family 마다 다를 수도 있도, 어떠한 model family에게는 차선의 선택일 때도 있습니다.

다른 변수들을 고정시키고 하나의 network hyperparameter를 변화시키는 것 보다, 모든 network hyperparameter를 변화시키면 어떻겠습니까? 이러한 방식은 confounding factor의 영향을 없에줍지만, 무한대에 가까운 model을 만듭니다. 그렇기에 저자는 새로운 방식의 비교방식을 소개합니다: distribution estimates.

선택된 몇게의 모델을 비교하는 Curve Estimate와는 다르게 distribution estimate는 design space(parameterize 가능한 인공신경망 구조)에서 model을 선택합니다. 이를 통해서 error rate와 model complexity사이의 distribution을 확인할 수 있습니다.

이러한 방식은 model family의 character에 중접을 둡니다. 그렇기에 이러한 방식은 model 탐색을 위한 design space에 관한 연구를 가능하게 합니다.

# Related Work

### Reproducible research

최근 추세는 기계학습의 재현가능성이 높은 것을 권장합니다. 그래서 저자는 시각인식 분야의 model architecture를 평가하는 더 강력한 methodology를 소개하는 목표를 공유합니다.

### Empirical studies

Deep network에 관한 이론적인 이해의 부족으로 인해서, 개발을 보조하기 위한 deep network의 large-scale studies를 실시하는 것은 필수 불가결한것입니다. Empirical studies와 강력한 methodology는 더 강력한 인공신경망을 만들기 위한 진전에 아주 중요한 역할 을 합니다.

### Hyperparameter search

일반적은 Hyperparameter 탐색 기술은 힘든 model tuning 과정을 다룹니다. 하지만 이 논문에서 저자는 직접적으로 전체적인 model distribution을 다루기에 중요하지 않습니다.

### Neural Architecture search

NAS는 인공신경망 구조를 배우는 것에 효과적이라는 것이 증명되었습니다. NAS는 두개의 구성으로 되어있습니다. Network design space와 탐색 algorithm. 대부분의 NAS 관련 논문들은 search algorithm에 집중합니다. 하지만 이 논문에서는 model design space의 정의에 집중합니다.

### Complexity measures

이 논문에서 저자는 confounding factor를 조절하면서 network design space의 해석에 집중합니다. 저자는 일반적으로 사용되는 복잡도 계산 방법인 model parameter 수와 multiply-add 연산의 수를 활용합니다.

# Design Spaces

## Definitions

### Model Family

Model Family는 연관된 인공신경망 구조들을 묶은 무한에 가까운 거대한 집합입니다. 대체적으로 이들은 고차원적인 설계 구조나 설계 원칙(예를 들어 residual connection)을 공유합니다.

### Design Space

Model family에 Empirical Study를 한다는 것은 매우 어렵습니다. 왜냐하면 model family는 대략적으로 정의되있고 대체적으로 자세하게 설명되지 않습니다. 이러한 추상적인 model family들 사이를 구별하기 위해서 Design Space가 소개되었습니다. Design space는 model family로부터 예시화 가능한 구체적인 인공신경망의 집합입니다.

Degisn space는 2개의 구성으로 이루어져 있습니다.
1. Model faimly의 parameterization
2. 각각의 hyperparameter에 가능한 값의 집합

### Model Distribution

Design space는 기하급수적인 숫자의 인공신경망 model을 포함하고 있습니다. 이에 철저한 조사를 하는 것은 적절하지 못합니다. 그렇기에 design space에서 저자는 고정된 model의 집합을 만들고 평가합니다. 그 뒤로 통계학적인 해석을 사용합니다. 어떠한 standard distribution은 물론 NAS 처럼 learned distribution 또한 이 paradigm에 통합될 수 있습니다.

### Data Generation

Network Design Spaces를 해석하기 위해서 저자는 각각의 design space로부터 수많은 model을 sample 하고 평가합니다. 이를 통해서 저자는 empirical study에 사용할 훌녀된 모델의 dataset을 만듭니다.

## Instantiations

### Model Family

저자는 3개의 기본적인 model family에 대하여 연구합니다.
1. Vanilla model family(VGG에서 영감을 얻은 feedforward 인공신경망)
2. ResNet model family
3. ResNeXt model family

### Design space

![Design Space Parameterization](../DesignSpaceParameterization.png)

위의 표에서 보는 것처럼 저자는 stem을 포함하여 3개의 stage와 head를 가진 인공신경망을 사용했습니다.

* ResNet design space에서 하나의 block은 2개의 convolution과 하나의 residual connection으로 이루어 져 있습니다.
* Vanilla design space는 ResNet design space와 같지만 residual conneciton이 없습니다.
* ResNeXt design space는 bottleneck block과 group을 사용합니다.

![Design Spaces](./DesignSpace.png)

이 표는 각각의 모델에서 사용할 hyperparameter를 적은 것입니다. 여기서 ![a, b, n](https://latex.codecogs.com/svg.image?a,b,c) 표기법을 사용했습니다. n개의 hyperparameter를 a와 b 사이에서 log-scale uniform 하서 선택합니다. 3개의 독립적인 stage는 이러한 hyperparameter를 각자 block의 개수 ![d_i](https://latex.codecogs.com/svg.image?d_i)와 channel의 개수, ![w_i](https://latex.codecogs.com/svg.image?w_i)를 선택하게 됩니다. 

이를 통해서 전체 가능한 모델의 수는 group이 없을 경우 ![number of models without groups](https://latex.codecogs.com/svg.image?(dw)^3) 그룹이 있을 경우 ![number of models with groups](https://latex.codecogs.com/svg.image?(dwrg)^3) 개가 있습니다.

### Model Distribution

저자는 위에서 설정된 각각의 design space에서 hyperparameter를 uniform하게 선택하여 model distribution을 만들었습니다.

### Data generation

저자는 CIFAR-10 dataset을 이용해서 훈련을 하였습니다. 이러한 방식은 large-scale 해석을 가능하게 하고 가끔은 recognition network의 성능평가를 위해서 사용하기도 합니다. 위의 표에서 설정된 hyperparameter를 이용해 모든 design space 에서 25000개의 model을 선택해서 평가를 위해 사용되었습니다.

