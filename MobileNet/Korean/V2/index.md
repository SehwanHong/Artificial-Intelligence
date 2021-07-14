# [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/pdf/1801.04381.pdf)

�� ������ ���ڴ� ���ο� mobile architecture, MobileNetV2�� �Ұ��մϴ�. MobileNetV2�� �� �𵨺��� �پ��� �׽�Ʈ ȯ��� ��ġ��ũ���� �� ���� ������ ���ϴ�.

# Introduction

�ΰ��Ű���� �پ��� �о��� ����н����� �������� �����׽��ϴ�. �̹��� �ν� ���������� �̹� �ΰ��� �νķ��� �پ�Ѿ����ϴ�. �̷��� ��Ȯ���� ������ �پ��� ����� ������� ����������ϴ�. ��÷�� �ΰ��Ű���� ���� ������ �Ҽ� �ִ� ��Ⱑ �ʿ��մϴ�. �̷��� �������Ѷ����� �޴�� ��⳪ �Ӻ���� ��⿡���� ����� �Ҽ��� �����ϴ�.

�� �������� ���ο� �ΰ��Ű�������� �Ұ��մϴ�. MobileNetV2��� �Ҹ���� �� �Ű���� �޴����� ���ҽ��� ���ѵ� ȯ�濡 �����ϵ��� ������� �����Դϴ�. ���귮�� �޸� ��뷮�� ���̸鼭, ����� ������ ��������, ���ڴ� inverted residual with linear bottleneck�̶�� ���ο� ���̾� ����� �Ұ��մϴ�

# Related Work

���� �ΰ��Ű���� ������ ���� �˰����� �ؼ��� Ž���� ���� ������ �־����ϴ�. �̷��� �������� hyperparameter ����ȭ, �پ��� ����� network pruning, �׸��� connectivity learning�� �ֽ��ϴ�. �̷��� �������� convolution block ���� connectivity ������ ��ȭ���׽��ϴ�.

����ȭ�� �ϴ� �ٸ� ������δ� ���� �˰����� ����ϴ� ��İ� ��ȭ�н��� ���� ���� Ž���� �ֽ��ϴ�. ������ �̷��� ����� ������ �ΰ��Ű���� ũ�Ⱑ Ŀ���ٴ� ���Դϴ�.

�� ������ ����� �ΰ��Ű���� �������� [MobileNetV1](../)�� ����մϴ�.

# Preliminaries, discussion and intuition

## Depthwise separable convolution

Depthwise Separable Convolution�� ȿ������ �ΰ��Ű���� ����� �� �ʿ��� �⺻ ����Դϴ�. �̴� full convolutional operator�� �����Ͽ� �ΰ��� ���̾�� ����ϴ�.

  1. Depthwise convolution
  2. pointwise convolution

�ڼ��� ������ [��ũ](../)�� Ȯ���� �ֽñ� �ٶ��ϴ�.

## Linear Bottlenecks

���� �ΰ��Ű���� N���� ���̾ �ִٰ� �����սô�. ���̾� ![L_i](https://latex.codecogs.com/svg.image?L_i)�� ���� ![tensor dimension](https://latex.codecogs.com/svg.image?h_i\times&space;w_i\times&space;d_i)�� activation tensor�� �����Ǿ����ϴ�. �̷��� activation layer���� ������ manifold of interest�� �����մϴ�. Manfold of interest�� low-dimensional subspace�� ����Ǿ� �ֽ��ϴ�. �ٸ� ���� �ϸ� deep convolutional layer�� d-channel pixel�鿡 ��ȣȭ�Ǿ� �ִ� ������ ���������� ��� mainfold �ȿ� ����Ǿ� �ֽ��ϴ�. �̴� low-dimensional subspace�� ���� �����մϴ�.

���������� layer transformation ![ReLU(Bx)](https://latex.codecogs.com/svg.image?ReLU(Bx))�� ����� non-zero volume ![S](https://latex.codecogs.com/svg.image?S)�� �����ٸ�, ![S](https://latex.codecogs.com/svg.image?S)������ ������ point�� linear transformation ![B](https://latex.codecogs.com/svg.image?B)�� �Է°��� ���ؼ� ���Ҽ� �ֽ��ϴ�. �̸� ���ؼ� ��ü ��°��� �����ϴ� �Է°��� �� �κ��� linear transformation���� ���ѵ˴ϴ�. �ٸ� ���� �ϸ�, ���� �ΰ��Ű���� ����� non-zero volume�� ���� linear claissifier�� ������ ǥ���˴ϴ�.

�ٸ� ��������, ReLU�� Channel�� �����ϰԵǸ�, �� Channel�� ������ �ʿ������� �ҽǵǰ� �˴ϴ�. ������ ä���� ���� ���ٸ�, activation manifold�� ������ �����ϰ� ���� ���ɼ��� �ֽ��ϴ�. �Ʒ��� �̹����� �̸� �����մϴ�.

![ReLU transformations of low-dimensional manifold embedded in higher-dimensional spaces](../../V2/ReLUtransformation.png)

�� ��������, ���� ù��° ������ n ���� ������ ����Ǿ� �ֽ��ϴ�. �̸� ������ ���� ��� ![T](https://latex.codecogs.com/svg.image?T)�� ���ϰ� ReLU�� ����� ���� ������� ![inverse of T](https://latex.codecogs.com/svg.image?T^{-1})�� ����� �ٽ� 2D�̹����� ��ȯ�� ���Դϴ�. ���⼭ ![n = 2,3](https://latex.codecogs.com/svg.image?n=2,3)�϶� ������ ������ ���� ���κп��� �ϳ��� ������ ��ȯ �� ���� Ȯ���Ҽ� �ֽ��ϴ�. ������ ![n = 15](https://latex.codecogs.com/svg.image?n=15)�� 30�϶����� ������ �ս��� ���� �Է°��� ����� �̹����� ���ɴϴ�.

������ڸ�, manifold of interest�� higher-dimenstional activation space �ȿ� �ִ� low-dimensional subspace�� �����ϱ� ���� �ΰ��� Ư���� :

1. Manfold of interest�� ReLU transformation �Ŀ� non-zero volumne�� �����ִٸ�, �̴� Linear transformation���� �� �� �ִ�.
2. ���� input manifold�� �Է� ���� ������ low-dimansional subspace�� �����Ѵٸ�, ReLU�� input manifold�� ���� ������ �սǾ��� �����Ҽ� �ֽ��ϴ�.

Manifold of interest�� low-dimension�̶��, linear bottleneck layer�� convolutional block�� �����Ͽ� manifold of interest�� �����Ҽ� �ִ�. ������ ���ؼ�, linear layer �� ����ϴ� ���� non-linearity�� ������ �ҽ� ��Ű�� ���� ���� �Ҽ� �ִ�.

## Inverted residuals

Bottleneck ������ ��� ������ �����Ѵٴ� ��Ƿκ��� �����Ͽ� skip connection�� bottleneck ���� ���̿� ��������ϴ�. ���⼭ expansion layer�� non-linear transpormation�� �����ϱ� ���� implementation deatil�Դϴ�.

Residual block | Inverted Residual Block
--------------|---------------
![Residual Block](../../V2/residualBlock.png) | ![Inverted Residual Block](../../V2/invertedResidualBlock.png)

Residual Block�� ���� ������ �̹����� ǥ���˴ϴ�. �̹������� ǥ���Ȱ� ó�� wide -> narrow -> wide �� ���·� bottleneck������ ��������ϴ�. ������ �� ������, ���ڴ� inverted residual�� �����մϴ�. �������� �̹���ó�� narrow -> wide -> narrow�� ������ ä���߽��ϴ�. �缱���� ǥ���� �κ��� non-linearlities�� ������� �ʽ��ϴ�. �̴� non-linearlity�� ����ؼ� ����� �����ս��� ���̱� �����Դϴ�.

Inverted residual block���� ����ϴ� skip connection�� [ResNet](../../../ResNet/Korean/)���� ��� �ϴ� �Ͱ� �����ϴ�. �̴� �������� ���̾ ����Կ��� gradient�� vanishing�ϴ� ���� �����ϱ� ���� �Դϴ�.

Inverted Residual block�� �޸� ��뷮�� ����, ���ɵ� �� �����ϴ�.

### Running time and parameter count for bottleneck convolution

![bottleneck residual block](../../V2/bottleneckResidualBlock.png)

���� ǥ�� inverse residual function�� ���� �⺻���� ������ ��Ÿ�� ���Դϴ�. ���� �Է°����� ![h w](https://latex.codecogs.com/svg.image?h\times&space;w)�� �̹����� ũ��, ![k](https://latex.codecogs.com/svg.image?k)�� Ŀ�� ������, ![t](https://latex.codecogs.com/svg.image?t)�� expansion factor, ![d'](https://latex.codecogs.com/svg.image?d')�� ![d''](https://latex.codecogs.com/svg.image?d'')�� ���� �Է�ä���� ���� ���ä���� �� �Դϴ�. �� ������ ����ؼ� multi-add�� ������ ���ϸ� �Ʒ��� �����ϴ�.

![complexity for bottlencck residual block](https://latex.codecogs.com/svg.image?h&space;\times&space;w&space;\times&space;t&space;\times&space;d'&space;\times&space;d'&space;&plus;&space;h&space;\times&space;w&space;\times&space;t&space;\times&space;d'&space;\times&space;k&space;\times&space;k&space;&plus;h&space;\times&space;w&space;\times&space;t&space;\times&space;d'&space;\times&space;d''&space;=&space;h&space;\times&space;w&space;\times&space;t&space;\times&space;d'&space;\times&space;(d'&space;&plus;&space;k^2&space;&plus;&space;d''))

�� ���ڴ� depthwise separable convolution(�� ��ũ���� �����)�� ���귮���� �����ϴ�. �̴� �߰������� �� ![1 by 1](https://latex.codecogs.com/svg.image?1\times1) convolution layer �����Դϴ�. ������, �Է°� ����� ������ depthwise convolution layer ���� �۱� ������, bottleneck residual block�� ��ü���� ���귮�� �۾����ϴ�.

![memory for mobilenet v1 and mobilenet v2](../../V2/memory.png)

���� ǥ���� Ȯ�� �Ҽ� �ֽ��ϴ�. ���⼭ ǥ���� ���ڵ���, channel�� ����/memory�� ���� �������ϴ�. 16bit float�� ����Ѵٰ� �������� ���� memory ��뷮�Դϴ�. ���⼭ MobileNetV2�� ���� ���� �Ÿ𸮸� ����մϴ�. ShuffleNet�� ũ��� 2x, g=3�� ����ߴµ�, �̴� MobileNetV1�� MobileNetV2���� ������ ���߱� �����Դϴ�.

# Model Architecture

*Building block�� ���� �ڼ��� ������ ���� ������ �ֽ��ϴ�*
*Detailed information about the building block is describe above.*

MobileNetV2�� ������ ���������� 32���� ���͸� ���� Fully convolution layer�� �����մϴ�. ���ķ� 19���� residual bottleneck layer�� ������ �ֽ��ϴ�. �Ʒ��� ǥ�� Ȯ���ϸ� MobileNetV2�� ������ Ȯ���� �� �ֽ��ϴ�.

![Structure of MobileNet Version 2](../../V2/mobileNetV2Structure.png)

�� ǥ���� c�� ��� ä���� ����, n�� building block�� �ݺ� Ƚ��, s �� ���� ù��° ���̾��� stride ũ��(�ٸ� ���̾��� stride�� 1�Դϴ�.) t�� expansion factor �Դϴ�.

������ non-Linearlity�� ReLU6�� ������ϴ�. ReLU6�� ���� 0���� ������ 0��, 0�� 6 ���� �� ���� �Է°� �״�θ� ����ϰ�, x�� 6���� Ŭ���� 6�� ����մϴ�.

# Implementation Notes

## Memory efficient inference

��κ��� machine learning platform���� �ΰ��Ű�� ������ directed acyclic compute hypergraph G�� ����� ���Դϴ�. Graph G���� edge�� operation�� �ǹ��ϰ� node�� �߰������� tensor�� �ǹ��մϴ�. �� �׷����� ���ؼ� memory ��뷮�� �Ʒ��� ���� ��� �� �� �ֽ��ϴ�.

![Computational cost of neural network graph](https://latex.codecogs.com/svg.image?M(G)&space;=&space;\min_{\pi&space;\in&space;\Sigma(G)}&space;\max_{i&space;\in&space;1&space;...&space;n}&space;\left&space;[&space;\sum_{A\in&space;R(i,\pi,&space;G)}|A|&space;\right&space;]&space;&plus;&space;size(\pi_i)&space;)

���⼭ ![intermediate tensors](https://latex.codecogs.com/svg.image?R(i,\pi,G)) ���� ���̿� �ִ� tensor�̰� �̴� ���� ![nodes](https://latex.codecogs.com/svg.image?\pi_i...\pi_n)�� node�� ������ �Ǿ� �ֽ��ϴ�. Tensor�� ũ��� ![size of tensor](https://latex.codecogs.com/svg.image?|A|)�̰� ������ ���� Kernel�� ũ��� ![size of storage](https://latex.codecogs.com/svg.image?size(\pi_i))�Դϴ�. 

MobileNetV2���� Residual Connection(identity Skip Connection)�� ������ �ٸ� ���� ������ ��������, �� �ΰ��Ű���� �����ϴµ� �ʿ��� memory�� ũ��� �Է°��� ũ��, ��°��� ũ��� ������ ���� kernel tensor�� ũ�⸦ ���� ���Դϴ�. �̴� �Ʒ��� ���Ŀ��� �� ǥ���Ǿ� �ֽ��ϴ�.

![memory usage for MobileNetV2](https://latex.codecogs.com/svg.image?M(G)=%5Cmax_%7Bop%5Cin%20G%7D%5Cleft%5B%20%5Csum_%7BA%5Cin%20op%7D%7CA%7C%20&plus;%20%5Csum_%7BB%5Cin%20op%7D%7CB%7C%20&plus;%20%7Cop%7C%20%5Cright%5D)

### Bottleneck Residual Block

![Inverted Residual Block](../../V2/invertedResidualBlock.png)

MobileNetV2�� ������ ���� �̹����͵� �����ϴ�. ���� ������ operation�� ������ ���� ������ ǥ���Ҽ� �ֽ��ϴ�. ![bottleneck operator](https://latex.codecogs.com/svg.image?F(x)=&space;\left&space;[&space;A&space;\circ&space;N&space;\circ&space;B&space;\right&space;]x)

���⼭ A�� B�� linear transformation�� N�� non-linear per-channel transformation�� �ǹ��մϴ�. ![inner tensor](https://latex.codecogs.com/svg.iamage?N=\mathrm{ReLU6}\circ\mathrm{dwise}\circ\mathrm{ReLU6}). �� ��Ȳ���� ![network](https://latex.codecogs.com/svg.iamage?F(x)) ������ �ϴ� �� �ʿ��� memory�� ���� �ּ� ![maxium memory](https://latex.codecogs.com/svg.image?|s^2k|&plus;|s'^2k'|&plus;O(\max(s^2,s'^2)))�Դϴ�. �� ���Ŀ��� s�� �Է� tensor�� �� ���� s'�� ��� tensor�� �Ѻ���. k�� �Է� channel�� ũ�⸦ k'�� ��� tensor�� ũ�⸦ �ǹ��մϴ�.

���� ������ ����, inner tensor ![I](https://latex.codecogs.com/svg.image?I)�� t���� n/tũ���� tensor���� ��ģ���� ǥ���˴ϴ�. �̴� �Ʒ��� ���� �������� ǥ���Ҽ� �ֽ��ϴ�.

![memory saving](https://latex.codecogs.com/svg.image?F(x)=\sum_{i=1}^{t}(A_i\circ&space;N\circ&space;B_i)(x))

�� ������ �̿��ϸ�, n=t �϶�, �ѹ��� �ϳ��� channel�� �����ϴ� ���� �ǹ��մϴ�. �̶� memory�� �ϳ��� channel�� �־ ����������, memory�� ���� �����Ҽ� �ֽ��ϴ�.

������ �� ����� ����ؼ� memory�� ������ �����ϰ� ���ִ� �ΰ��� ��������� �ֽ��ϴ�.

1. inner transformation(non-linearlity�� depthwise ������ ��� ������)�� per-channel�����Դϴ�.
2. ������ non-per-channel ��ȯ�� ��� channel�� ���� �Է� channel�� ������ �ξ� �۱� �����Դϴ�.

T�� ũ�⸦ ��ȭ��Ű�� �Ϳ� ��ü���� ���귮�� ��ȭ���� �ʽ��ϴ�. ������ ������ �ϴµ� �ɸ��� �ð��� t�� ũ�⿡ ���� ��ȭ�մϴ�. �� ������ t�� �ʹ� ������ cache miss�� �߻��Ͽ� ���� �ð��� �����ϱ� �����Դϴ�. �׷����� t�� 2�� 5 ������ ���� ����ϴ� ���� memory ��뷮�� ȿ������ ����ð��� ������� �˴ϴ�.
 
# Experiments

## ImageNet Classification

### Training setup

�� �ΰ��Ű�� model�� Tensorflow�� Ȱ���ؼ� �ƷõǾ����ϴ�. Optimizer�� RMSPropOptimizer�� ����߽��ϴ�. �̶� decay�� momentum�� 0.9�� ����߽��ϴ�. Batch normalization�� ��� layer�� �ڿ� ���Ǿ�����, weight decay�� 0.00004�� ���Ǿ����ϴ�. Learning rate�� ���ʿ� 0.045�� ���Ǿ���, �ϳ��� epoch �� �������� ���� 0.98�� �����־������ϴ�. �Ʒÿ��� 16���� GPU�� ����߰�, batch�� ũ��� 96�������ϴ�.

### Result

![Preformance Curve for full model](../../V2/performanceCurve.png)

���� �׷����� MobileNetV2, MobileNetV1, ShuffleNet, NasNet�� ������� �� ���� �� �ִ� �پ��� ����� ��Ÿ�� ���Դϴ�. �̶� resolution multiplier�� 0.35, 0.5, 0.75, 1�� ����� ���Դϴ�. MobileNetV2������ 1.4�� �߰������� ����ؼ� �� ���� ����� ������ϴ�.

![Performance Table for selected models](../../V2/performanceTable.png)

���� ǥ�� �׷������� ���õ� ���� ��Ÿ�� ���Դϴ�. ���⼭ �𵨿� ���� parameter�� ������ multi-add ���귮�� �˼� �ֽ��ϴ�. ������ ���ڴ� Google Pixel 1�̶�� ����Ʈ ������ Tensorflow Lite�� ������� ���� ���� �ð��� ǥ���� ���Դϴ�. �̶� ShuffleNet�� �ð��� ǥ���� ���� �ʾҴµ�, �������� shuffling�� group convolution �˰����� �������� �ʾұ� �����Դϴ�.


���� ǥ�� �ǹ��ϴ� �ٴ�, MobileNetV2�� MobileNetV1���� ��Ȯ���� ���� ����ð��� ���� ���� Ȯ���Ҽ� �ֽ��ϴ�. ���� NasNet-A�� MobileNetV2(1.4)�� ���ϸ�, MobileNetV2(1.4)�� �� ���� ��Ȯ���� ������ ������ ���� �뷫 30%���� ���� ���� Ȯ�� �� �� �ֽ��ϴ�.


## Object Detection

### SSD Lite

�� ������ mobile�� �� ����ȭ�� SSD�� ���ο� ������ �Ұ��մϴ�. SSD Lite��� �Ҹ���� �� ���� SSD�� ���� ���̾��� �Ϲ����� convolution ������ ��� separable convolution(depthwise �Ŀ� pointwise)�������� �ٲ� ���Դϴ�.


![SSD and SSDLite configuration](../../V2/SSD.png)

MobileNetV2�� ������� 80���� class�� �����ϴ� SSD�� SSDLite�� ũ��� ���귮�� ���غ���, SSDLite�� �뷫 7�� ���� ���� parameter ���� ������ �ְ� ���귮�� 4�� ���� ���� Ȯ�� �� �� �ֽ��ϴ�.


![result for object detection](../../V2/performanceObjectDetection.png)

MobileNetV2�� SSDLite�� ���ÿ� ��� �ϴ� ���� parameter�� multi-add�� ���� ���� ���̴� �Ϳ� ���� ���� percision�� �����ϴ�. MobileNetV1�� ���� ���, ����� ��Ȯ���� ������ ������, MobileNetV2�� ���ݴ� �����ϴ�. ����, YOLOv2�� ���� ���, MobileNetV2�� 20�� �� ȿ�����̰�, parameter �� 10�� �۽��ϴ�.

## Semantic Segmentation

DeepLabv3�� ����ϴ� MobileNetV1�� MobileNetV2�� mobile segmentic segmentation�� �۾����� ���غ��ô�. DeepLabv3�� atrous convolution�� ����մϴ�. Atrous convolution�� ���� feature map�� �ػ󵵸� �����ϴ� ������ �����Դϴ�. DeepLabv3�� 5���� ���� head�� ������ �ֽ��ϴ�. ���⿡�� a) atrous spatial pyramid pooling module(three ![3 by 3](https://latex.codecogs.com/svg.image?3\times3) convolution with different atrous rates)�� b) ![1 by 1](https://latex.codecogs.com/svg.image?1\times1) convolution head�� c) Image-level features�� �ֽ��ϴ�.

�� �������� 3���� �ٸ� �������� �����߽��ϴ�.

1. �پ��� feature extractor
2. ���� ������ ���� DeepLabv3 head�� ����ȭ
3. performance boosting�� ���� �پ��� inference ����

![semantic Segmentation result](../../V2/performanceSementicSegmentation.png)
**MNetV2\*** Second last feature map is used for DeepLabv3 head.
**OS**: output stride
**ASPP**: Atrous Spatial Pyramid Pooling
**MF**: Multi-scale and left-right flipped input

���� ǥ�� �м��� �� ���:
a. inference ������ ����� ��� multi-add computation�� ���� ���ϱ޼������� �����մϴ�. �̴� multi-scale input�� left-right flip���� �����ϴ� ���Դϴ�. ���귮�� ����� ���������� �̴� ���ȿ� �����ϱ⿡�� ���� ������ �ƴմϴ�.
b. ![output-stride = 16](https://latex.codecogs.com/svg.image?output\_stride=16)�� ����ϴ� ���� ![output stride = 8](https://latex.codecogs.com/svg.image?output\_stride=8)�� ����ϴ� �ͺ��� �� ȿ�����Դϴ�.
c. MobileNetV1�� ����ϴ� ���� ResNet-101�� ����ϴ� �Ͱ� ���� 5���� 6�� �� ȿ�����Դϴ�.
d. DeepLabv3�� MobileNetV2�� ���������� �ι�° feature map�� �����ϴ� ���� ������ featuremap �� �����ϴ� �ͺ��� �� ȿ�����Դϴ�.
e. DeepLabv3 heads�� ���꺹�⵵�� �����ϴ�. ASPP module�� �����ϴ� ���� ������ ����������� multi-add ���� ���� ���� ���� �� �ֽ��ϴ�.

# Reference

[Toward Data Science](https://towardsdatascience.com/mobilenetv2-inverted-residuals-and-linear-bottlenecks-8a4362f4ffd5)

[Hongl tistory 1](https://hongl.tistory.com/195)
[Hongl tistory 2](https://hongl.tistory.com/196)

## [Link to Neural Net](../../../)
## [Link to MobileNet](../)