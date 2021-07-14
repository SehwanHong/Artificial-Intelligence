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

*Detailed information about the building block is describe above.*

The architecture of MobileNetV2 contains the initiall fully convolutional layer with 32 filters, followed by 19 residual bottleneck layers descibed in the table below.

![Structure of MobileNet Version 2](../../V2/mobileNetV2Structure.png)

c is the number of output channel, n number of repeatition for building block, s stride of the first layer, otherwise stride is 1. t is the expanstion factor.

For non-linearlity, author chose ReLU 6, when x less then 0, returns 0, when x between 0 and 6, returns x, when x is greater than 6, returns 6.

# Implementation Notes

## Memory efficient inference

In most of the famous machine learning platform, network implementation builds a directed acyclic compute hypergraph G. In the graph G, the edge represents the operation and the node consists of tensor of intermediate computaiton. Though these graph, the memory usage can be calculated as following.

![Computational cost of neural network graph](https://latex.codecogs.com/svg.image?M(G)&space;=&space;\min_{\pi&space;\in&space;\Sigma(G)}&space;\max_{i&space;\in&space;1&space;...&space;n}&space;\left&space;[&space;\sum_{A\in&space;R(i,\pi,&space;G)}|A|&space;\right&space;]&space;&plus;&space;size(\pi_i)&space;)

Where ![intermediate tensors](https://latex.codecogs.com/svg.image?R(i,\pi,G)) is the intermediate tensors that are connected to any of ![nodes](https://latex.codecogs.com/svg.image?\pi_i...\pi_n). ![size of tensor](https://latex.codecogs.com/svg.image?|A|) is the size of tensor, and ![size of storage](https://latex.codecogs.com/svg.image?size(\pi_i)) is the total amound of memory in internal storage for operation.

Since there is no other structure rather than residual connection(identity shortcut), memory needed in this neural network is addition of input, output and the tensor size. Therefore, it could be presented as below.

![memory usage for MobileNetV2](https://latex.codecogs.com/svg.image?M(G)=%5Cmax_%7Bop%5Cin%20G%7D%5Cleft%5B%20%5Csum_%7BA%5Cin%20op%7D%7CA%7C%20&plus;%20%5Csum_%7BB%5Cin%20op%7D%7CB%7C%20&plus;%20%7Cop%7C%20%5Cright%5D)

### Bottleneck Residual Block

![Inverted Residual Block](../../V2/invertedResidualBlock.png)

In the MobileNetV2, the architecture is defined as the image above. The operation could be represented ad following equation, ![bottleneck operator](https://latex.codecogs.com/svg.image?F(x)=&space;\left&space;[&space;A&space;\circ&space;N&space;\circ&space;B&space;\right&space;]x)

A and B is linear transformation. N is a non linear per-channel transformation. ![inner tensor](https://latex.codecogs.com/svg.iamage?N=\mathrm{ReLU6}\circ\mathrm{dwise}\circ\mathrm{ReLU6}). In this situation, the memory required to compute ![network](https://latex.codecogs.com/svg.iamage?F(x)) can be as low as ![maxium memory](https://latex.codecogs.com/svg.image?|s^2k|&plus;|s'^2k'|&plus;O(\max(s^2,s'^2))), where s is one side of input tensor, s' is a side of output tensor, k is input channel size, k' is output channel size.

From this equation, the inner tensor ![I](https://latex.codecogs.com/svg.image?I) can be represented as concatenation of t tensors wite size of n/t. Following representation below.

![memory saving](https://latex.codecogs.com/svg.image?F(x)=\sum_{i=1}^{t}(A_i\circ&space;N\circ&space;B_i)(x))

From this equation, when n=t, calculating one channel at a time, we only need to keep one channel of the intermediate representation at all time, saving memory significantly.

However, there are two constaints when using this trick of reducing memory.

1. the inner transformation(which includes non-linearlity and depthwise) is per-channel
2. consecutive non-per-channel operators have significiant ratio of the input size to the output

Using differnt t does not effect the total calculation time, but have effect the runtime by increasing cache misses which cause significant increase in runtime. Using t between 2 to 5 is most helpful of reducing memory usage and utilzing efficient calculation.
 
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