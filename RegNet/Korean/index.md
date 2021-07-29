# [Designing Network Design Spaces](https://arxiv.org/pdf/2003.13678.pdf)

이 논문의 목표는 인공신경망 설계에 대한 이해를 도우는 것과, 범용적인 환경에서 사용가능한 설계 원칙을 찾는 것입니다. 인공신경망 하나를 설계하는 것에 집중하는 것이 아니라 저자는 인공신경망 집합을 paramterize하는 network design space를 설계합니다.

# Introduction

이 논문에서, 저자는 새로운 인공신경망 design paradigm을 소개합니다. 이 방식은 manual design의 장점과 인공신경망 탐색의 장점을 합쳤습니다. 인공신경망 하나를 설계하는 것에 집중하는 것이 아니라 저자는 인공신경망 집합을 paramterize하는 network design space를 설계합니다. Manual design 처럼, 저자는 해석가능성을 목표로 하고 있습니다. 이를 통해서 간단하며, 잘 장동하고, 모든 상황에서 사용가능 한 범용적인 design principle을 찾는 것을 목표로 합니다. 그리고 NAS 처럼 저자는 semi-automated procedure의 장점을 통해서 위의 목표를 이루려고 합니다.

![Design Space Design](../DesignSpaceDesign.png)

범용적인 전략은 성능을 유지시키거나 향상시키면서 초기의 design space에서 점진적으로 간단한 방식으로 설계를 하는 것입니다. 위의 이미지에서 보는 것처럼, 가장 초기의 design space A에서 두개의 개선방식을 적용시켜 design space B와 C를 얻었습니다. 이러한 경우, ![C B A](https://latex.codecogs.com/svg.image?C%5Csubseteq%20B%5Csubseteq%20A)이고, Error Dristribution이 A에서 B, B에서 C로 갈때 향상되는 것을 확인 할 수 있습니다. Model population에 적용된 design principle이 더 효과적이고 범용적이라는 희망을 가지고 있습니다. 전체적인 방식은 manual design과 비슷합니다. 다만 population level에 적용을 하고 network design space의 distribution estimate를 통해서 설명합니다.

이 논문의 주제는 인공신경망의 model family가 VGG, ResNet, ResNeXt를 포함하고 있다는 가정아래 network structure를 탐색하는 것입니다. 이러한 제약이 없는 design space를 AnyNet이라고 부르고, 이 AnyNet에 Human-in-the-loop 방식을 사용해서 low-dimensional design space에 도달한 network를 RegNet이라고 부릅니다. RegNet은 간단한 "regular" network로 이루어졌기에 그와 같은 이름이 붙었습니다.

