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

The model is trained using Tensorflow. Optimizer is RMSPropOptimizer with decay and momentum set to 0.9. Batch normalization is used after every layer and standard weight decay is set to 0.00004. Initial learning rate is 0.045, and learning rate decay is rate of 0.98 per epoch. 16 GPU asynchronous workers and a batch size of 96.

### Result

![Preformance Curve for full model](../../V2/performanceCurve.png)

This table represents all possible result of the MobileNetV2, MobileNetV1, ShuffleNet, Nas Net. For these networks, multiplier of 0.35, 0.5, 0.75, and 1 is used for all resulutions, and additional 1.4 is used on MobileNetV2 for 224 to obtain better result.

![Performance Table for selected models](../../V2/performanceTable.png)

From some of the selected model, we get the number of parameters and the Multi-adds. Last coloumn is implimentation of network in Google Pixel 1 using Tensorflow Lite. The number for shuffleNet is not reported because shuffling and group convolution algorithm are not yet supported.

Above table explains, that MobileNet V2 have higher accuracy rate compared to mobileNet V1 and faster computation time. Also comparing NasNet-A and MobileNetV2(1.4), MobileNet have higher accuracy and is 30% faster than NasNet-A.

## Object Detection

### SSD Lite

In this papaer, we introduce a mobie friendly variant of regular SSD(single shot detector). SSD Lite replace all the regular convolutions with separable convolutions(depthwise followed by pointwise) in SSD prediction layers.

![SSD and SSDLite configuration](../../V2/SSD.png)

Comparison of the size and computational cost between SSD and SSDLite configured with MobileNetV2 and making predictions for 80 classes. SSDLite is approximately 7 times smaller in parameter size and 4 times smaller in computation.

![result for object detection](../../V2/performanceObjectDetection.png)

MobileNetV2 with SSDLite makes a decent predection using much less parameters and Multi-add computation. Compared to MobileNetV1, it have similar accuracy but MobileNetV2 computes littlebit faster than MobileNetV2. Also comparing with YOLOv2, MobileNetV2 is 20 times more efficient and 10 times more smaller while still outperforms YOLOv2.

## Semantic Segmentation

Compare MobileNetV1 and MobileNetV3 with DeepLabv3 for the task of mobile segmantic segmentation. DeepLabv3 use atrous convolution a powerful tool to explicitly control the resolution of computed feature maps and builds five parallel heads including (a) Atrous Spatial Pyramid Pooling module(ASPP) containing three ![3 by 3](https://latex.codecogs.com/svg.image?3\times3) convolution with different atrous rates, (b) ![1 by 1](https://latex.codecogs.com/svg.image?1\times1) convolution head, and (c) Image-level features.

Three design variation is tested in this paper.

1. Different feature extractor
2. Simplifying the DeepLapv3 heads for faster computation
3. Different inference strategies for boosting performance

![semantic Segmentation result](../../V2/performanceSementicSegmentation.png)
MNetV2\* Second last feature map is used for DeepLabv3 head.
OS: output stride
ASPP: Atrous Spatial Pyramid Pooling
MF: Multi-scale and left-right flipped input

Observation on the table
a. the inference strategies, including multi-scale inputs and adding left-right flipped images, significantly increases multi-add computation thus not suitable for on-device applications.
b. using ![output-stride = 16](https://latex.codecogs.com/svg.image?output\_stride=16) is more efficient than ![output stride = 8](https://latex.codecogs.com/svg.image?output\_stride=8)
c. MobileNetV1 is 5 to 6 times more efficient compared to ResNet-101
d. Building DeepLabv3 on top of the second last feature map of the MobileNetV2 is more efficient than on the original last-layer feature map.
e. DeepLabv3 heads are computationally expensive and removing the ASPP module significanlty reduces the Multi-add computation with only slight preformance degradation

# Reference

[Toward Data Science](https://towardsdatascience.com/mobilenetv2-inverted-residuals-and-linear-bottlenecks-8a4362f4ffd5)

[Hongl tistory 1](https://hongl.tistory.com/195)
[Hongl tistory 2](https://hongl.tistory.com/196)

## [Link to Neural Net](../../../)
## [Link to MobileNet](../)