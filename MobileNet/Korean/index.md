# [MobileNets:Efficient Convolutional Neural Networks for Mobile Vision Application](https://arxiv.org/pdf/1704.04861.pdf)

Depthwise separable convolutions�� ����� ����ȭ�� architecture�� ������� MobileNet�� ������ DNN�� ������� �մϴ�. �� ������ ���ڴ� �ΰ��� global hyperparameter�� �Ұ��մϴ�. �� �ΰ��� hyperparameter�� ȿ�������� ��Ȯ���� latency�� ��ȯ�մϴ�.


# Introduction

AlexNet�� ImageNet Challenge: ILSVRC 2012�� �̱�� �� ����, Convolutional Neural Network�� ��ǻ�� �������� ���������� ���Ǵ� ����� �Ǿ����ϴ�. �� ���ķ� �ΰ��Ű�� ������ ��ü���� �߼��� ��� ������ �Ű���� ���ؼ� ���� ��Ȯ���� ��� ���� ��ǥ�� �ϰ� �ֽ��ϴ�. ������ ��Ȯ���� ������Ű�� ������ ���� ��� ���⵵�� ���ؼ� �̷������ �ֽ��ϴ�. �ǻ�Ȱ �ӿ���, ���� ��� �κ�ƽ��, �������� �ڵ���, �׸��� �������ǿ���, ��ü �ν��� �����ð� ���� ���Ǿ�� �մϴ�.

�� ������ ���ڴ� ȿ������ �Ű�� ������ �ΰ��� hyperparameter�� �Ұ��մϴ�. �̸� ���ؼ� ���� �۰� ���� �ӵ��� ���� ���� ����ϴ�. �̴� ����� ����� �Ӻ���� ���� ���α׷��� ���� ��ǿ��� �´� �ΰ��Ű�� �Դϴ�.

# Prior Work

�۰� ȿ������ �ΰ��Ű���� ����� ����� ���������� �ֽ��ϴ�. �پ��� ��ĵ��� �ΰ����� Ŀ�ٶ� �з��� ������ �ֽ��ϴ�:

* �̸� �Ʒõ� �Ű���� �����ϴ� ���
* ���� �ΰ��Ű���� �Ʒ��ϴ� ���

�� ������ ���ڴ� ���ο� �η��� �ΰ��Ű�� ������ �����մϴ�. �� ������ ��� �Թ��ڷ� �Ͽ��� �ڽŵ��� ���ø����̼��� �ڿ� ���ѿ� �´� ���� �ΰ��Ű���� �� �� �ְ� ����� �ݴϴ�. MobileNet�� ���� �������� ������ �����Ͻø� ����ȭ �ϴ� ���� ���� ���� �Ű���� ����� ���Դϴ�.

## Training small networks

MobileNet�� depthwise separable convolution�� ����Ͽ� ����� �����ϴ�. Depthwise separable convolutiondms Inception �𵨿��� ù ����� ���̾�� ������ ���̱� ���ؼ� ���Ǿ����ϴ�. Flatten Network�� �ΰ��Ű���� fully factorized convolution�� ������� ���������, extremely factorized network�� ���� ���ɼ��� �����־����ϴ�. Factorized Networks�� small factorized convoutions�� topological conncetions�� ����� �Ұ��߽��ϴ�.  Xception Network�� depthwise separable filter�� ������Ŵ�� ���� Inception V3 ���� �ɰ��ϴ� ����� �����־����ϴ�. Squeezenet�� bottlenet ����� ����ؼ� ���� ���� �Ű���� ��������ϴ�. �ٸ� ������δ� structured transform networks�� deep fried convnet�� �ֽ��ϴ�.

���� �Ű���� �Ʒý�Ű�� �� �ٸ� ������δ� Ŀ�ٶ� �ΰ��Ű���� ���ؼ� ���� �Ű���� �Ʒý�Ű�� ��ĵ� �ֽ��ϴ�.

## Obtaining small networks by factorizing or compressing pretrained networks.

Quantization, hashing, pruing �׸��� vector quantization�� ������ �ڵ��� ����ϴ� ������ �پ��� ������ Ȯ�� �� �� �ֽ��ϴ�. �Դٰ�, �پ��� Factorization�� �̸� �Ʒõ� �Ű���� ������ ����� ������ ����� �ݴϴ�.

# MobileNet Architecture

## Depthwise separable Convolution

MobileNet ���� depthwise separable convolution�� ������� ����������ϴ�. Depthwise separable model�� factorized convolution�� �� ������� �⺻���� convolution�� depthwise convolution�� ![1 \times 1](https://latex.codecogs.com/svg.image?1\times1) convolution�� pointwise convolution���� ������ ���Դϴ�.

### Standard Convolution

�⺻���� convolution�� �Է°��� ���͸� �����԰� ���ÿ� ���ļ� �� �ϳ��� �ܰ�� ��°����� ��ȯ�մϴ�. �츮�� ���� �Է� feature map�� ![D_F \times D_F \times M](https://latex.codecogs.com/svg.image?D_F&space;\times&space;D_F&space;\times&space;M)���� ��� feature map�� ![D_F \times D_F \times N](https://latex.codecogs.com/svg.image?D_F&space;\times&space;D_F&space;\times&space;N)���� ǥ���Ѵٸ� �⺻ convolution layer�� kernel ũ�⸦ ������ ���� ǥ���� �� �ֽ��ϴ�.

![Standard Convolutional Filters](../standardConvFilter.png)

���� �̹������� ���ΰ� ó��, �⺻ convolutional layer�� ����ũ�Ⱑ ![D_K \times D_K \times M \times N](https://latex.codecogs.com/svg.image?D_K&space;\times&space;D_K&space;\times&space;M&space;\times&space;N)���� ǥ�� �� �� �ֽ��ϴ�. ���⼭ ![D_K](https://latex.codecogs.com/svg.image?D_K) ��  filter kernel�� ũ��,  ![M](https://latex.codecogs.com/svg.image?M)�� input Channel�� ũ��. ![N](https://latex.codecogs.com/svg.image?N)�� output Channel�� ũ���Դϴ�..

Kernel�� ũ��� �Է°��� ũ�⸦ �˰� �ִٸ�, �츮�� �⺻���� convolutional layer�� ���귮�� �뷫������ ��� �� �� �ִµ�, �� ������ �Ʒ��� �����ϴ�.

![D_K \times D_K \times M \times N \times D_F \times D_F](https://latex.codecogs.com/svg.image?D_K&space;\times&space;D_K&space;\times&space;M&space;\times&space;N&space;\times&space;D_F&space;\times&space;D_F)

### Depthwise Separable convolution

�⺻���� Convolution���� �ٸ��� depthwise separable convolution�� �ΰ��� ���̾�� �з��Ҽ� �ֽ��ϴ�.
* ���͸� ���� Depthwise convolution
* �ϳ��� ��ġ�� ���� Pointwise convolution

Depthwise convolution�� depthwise separable convolution���� �����ϱ� ���� �ܰ��Դϴ�. �� �ܰ迡�� �ϳ��� ���Ͱ� �ϳ��� �Է� Channel���� ����˴ϴ�. �Ʒ��� �̹����� ����, kernel ũ��� ![D_K \times D_K \times M \times 1](https://latex.codecogs.com/svg.image?D_K&space;\times&space;D_K&space;\times&space;M&space;\times&space;N)�� �˴ϴ�.

![Depthwise convolution filter](../depthwiseConvFilter.png)

Depthwise convolution ������ ���귮�� �⺻ convolution layer���� �������� �ſ� �۽��ϴ�. �ֳ��ϸ� ��� �Է°����� �����ϱ� ���� �߰����� parameter�� �ʿ� ���� �����Դϴ�. ���귮�� �������� ǥ���Ѵٸ� �Ʒ��� �����ϴ�.

![D_K \times D_K \times M \times D_F \times D_F](https://latex.codecogs.com/svg.image?D_K&space;\times&space;D_K&space;\times&space;M&space;\times&space;N&space;\times&space;D_F&space;\times&space;D_F)

Pointwise convolution�� depthwise separable convolution���� ���͵� ������ ��ġ�� �ܰ��Դϴ�. �� �ܰ迡�� ![1 \times 1](https://latex.codecogs.com/svg.image?1\times1) convolution�� ��� ���� ������ ��ġ�µ� ���˴ϴ�. �̶� kernel ũ��� ![1 \times 1 \times M \times N](https://latex.codecogs.com/svg.image?1\times1\times&space;M&space;\times&space;N)�Դϴ�

![Pointwise convolution filter](../pointwiseConvFilter.png)

Pointwise convolution ������ ���귮�� �Է°��� ��°��� ����մϴ�. ������ Kernel ũ��ʹ� �����մϴ�. �̶��� ���귮�� �Ʒ��� �������� ǥ���˴ϴ�.

![M \times N \times D_F \times D_F](https://latex.codecogs.com/svg.image?M&space;\times&space;N&space;\times&space;D_F&space;\times&space;D_F)

Depthwise convolution�� pointwise convolution�� ���� ���ϸ�, depthwise separable convolution�� �� ���귮�� ���� �� �ֽ��ϴ�. �̶��� ���� �Ʒ��� �������� ǥ���˴ϴ�.

![D_K \times D_K \times M \times D_F \times D_F + M \times N \times D_F \times D_F](https://latex.codecogs.com/svg.image?D_K&space;\times&space;D_K&space;\times&space;M&space;\times&space;D_F&space;\times&space;D_F+M&space;\times&space;N&space;\times&space;D_F&space;\times&space;D_F)

### Reduction in Computation.

���귮�� ����� �� ���� ������ ���� �츮�� depthwise separable convolution�� ����Ҷ��� standard convolution�� ����Ҷ��� ���귮�� ���� �� �ֽ��ϴ�. �̸� ���ؼ� depthwise separable convolution�� �󸶸�ŭ ������ ���� �� �ִ��� ��� �ϰ� �Ǹ�,

![computaion ratio](https://latex.codecogs.com/svg.image?\frac{D_K&space;\times&space;D_K&space;\times&space;M&space;\times&space;D_F&space;\times&space;D_F&space;&plus;&space;M&space;\times&space;N&space;\times&space;D_F&space;\times&space;D_F}{D_K&space;\times&space;D_K&space;\times&space;M&space;\times&space;N&space;\times&space;D_F&space;\times&space;D_F}&space;=&space;\frac{1}{N}&space;&plus;&space;\frac{1}{D_K^2})

MobileNet�� Kernel ũ�Ⱑ ![3\times3](https://latex.codecogs.com/svg.image?3\times3)�Դϴ�. �׷�����, MobileNet�� �뷫 8�迡�� 9�� ������ ������ ���� �մϴ�.

## Network Structure and Training

![mobilenet structure](../mobilenetStructure.png)

MobileNet�� �⺻�� ������ depthwise separable convolution�� ������� ����������ϴ�. �̶� ���� ù��° layer���� Full convolution�� ����մϴ�. �������� fully connected layer�� ������ ��� layer�� batch normalization�� ReLU non-lineality�� �ڵ����ϴ�. ������ ���̾��� Funnly Connected layer�� softmax layer�� ����մϴ�. Downsamping�� depthwise convolution�� stride�� �ٲٴ� ������ �մϴ�. ���� ù��° layer�� standard convolution layer������ downsampling�� stride�� �ٲٴ� ������ ����մϴ�. ������ average pooling�� ���� �ػ󵵸� 1�� �ٲپ� ������ layer�� fully convolutional layer�� �Է¹�Ŀ� ���߾� �ݴϴ�. depthwise convolution�� pointwise convolution�� �ٸ� layer��� ���������� MobileNet�� �� 28���� layer�� �̷�� ���ֽ��ϴ�.

Standard Convolutional layer | Depthwise Separable Convolutional Layer
-----------|-----------
![standard convolutional layer](../standardConvLayer.png) | ![depth wise separable convolutional layer](../depthwiseConvLayer.png)

���� �̹����� standard convolution layer�� depthwiese separable convolutional layer�� �������� �����ݴϴ�. ������ ǥ���Ѱ� ó�� standard convolution�� �Ѵܰ迡�� ��� ���� �����մϴ�. �ݸ�, depthwise seaprable convolution�� depthwise separable convolution�� ���ؼ� channel���� filter�ϰ� pointwise convolution�� ���ؼ� �Է°����� ���� ��°����� ��ȯ�մϴ�.

ȿ������ �Ű���� �ܼ��� Mult-Adds���������� ���ǵ��� �ʽ��ϴ�. ������ �󸶳� ȿ������ �������� ����Ǵ� ���� �� �߿��մϴ�. ���� �� sparse matrix������ dense matrix ���꺸�� �׻� ������ �ʽ��ϴ�.

## Width Multiplier: Thinner Models

MobileNet������ �̹� �۰� ������ �����ϴ�. ������ ���� ���, Ư���� ��쳪 ���ø����̼��� ��� �� ���� ���� ���ϰų� �� ���� ���� ���ϴ� ��찡 �����ϴ�. �̶� ���� �𵨰� ���귮�� ���� ���� ����� ���ؼ� ���ڴ� width multiplier�� �Ұ��մϴ�.

Width multiplier ![alpha](https://latex.codecogs.com/svg.image?\alpha)�� ������ ��ü������ �Ű���� ������ ��� ����� �ݴϴ�. �ϳ��� ���̾ �־�������, width multiplier�� ����ϸ�, �Է°��� ũ���� M�� ![alpha](https://latex.codecogs.com/svg.image?\alpha&space;M)�� ���ϰ� �ǰ�, ��°��� ����ϰ� ������ �˴ϴ�. �� ��� �� ���귮�� �Ʒ��� ���� ���ϰ� �˴ϴ�.

![Width Multiplier](https://latex.codecogs.com/svg.image?D_K&space;\times&space;D_K&space;\times&space;\alpha&space;M&space;\times&space;D_F&space;\times&space;D_F+\alpha&space;M&space;\times&space;\alpha&space;N&space;\times&space;D_F&space;\times&space;D_F)

���⼭ ![alpha](https://latex.codecogs.com/svg.image?\alpha)�� ���� 0�� 1 ������ ���� �˴ϴ�. ![alpha=1](https://latex.codecogs.com/svg.image?\alpha=1)�϶�, baseline MobileNet�̰�, ![alpha<1](https://latex.codecogs.com/svg.image?\alpha<1), reduced MobileNets�Դϴ�..

Width Multiplier����Ҷ�, ���귮�� width multiplier�� ������ ����ϰ� �پ��ϴ�. 

## Resolution Multiplier: Reduced Representation

Resolution Mutliplier ![rho](https://latex.codecogs.com/svg.image?\rho)�� �Է� �̹����� �߰� ǥ�����鿡�� ����Ǿ� ���귮�� �ٿ��ݴϴ�.

Width multiplier ![alpha](https://latex.codecogs.com/svg.image?\alpha)�� resolution Multiplier ![rho](https://latex.codecogs.com/svg.image?\rho)�� ���ÿ� �����Ų�ٸ�, depthwise separable convolutional layer�� ���귮�� �Ʒ��� ���� ǥ���Ҽ� �ֽ��ϴ�.

![Resolution Multiplier and Width Mutliplier](https://latex.codecogs.com/svg.image?D_K&space;\times&space;D_K&space;\times&space;\alpha&space;M&space;\times&space;\rho&space;D_F&space;\times&space;\rho&space;D_F+\alpha&space;M&space;\times&space;\alpha&space;N&space;\times&space;\rho&space;D_F&space;\times&space;\rho&space;D_F)

# Experiment

Depthwise separable convolution�� ȿ���� layer�� ������ ���̴� �� ��ſ� network�� width�� ���̴� ���� ȿ��

## Model Choices

Depthwise separable convolution�� ����� MobileNet�� Full convolution�� ����� MobileNet�� ��.

![depthwise separable vs full convolution mobile net](../compareConvolution.png)

���� ���̺��� Ȯ���� ���, depthwise seaprable convolution�� ����� ��� 9������ ������ �پ����� 1% ������ ��Ȯ���� �پ�����.

Width multiplier�� ����ϴ� ���� �𵨰� �� ���� ���� layer�� ����ϴ� ���� ���� ���Ѵ�. ���� MobileNet�� ����� ���ؼ�, 5���� separable filter�� �����ߴ�. �̶��� feature map ũ��� ![14 14 512](https://latex.codecogs.com/svg.image?14\times14\times512)�̴�.

![thin model vs shallow model](../compareThinShallow.png)

���� ǥ�� Ȯ���� ����, ���� �𵨰� ���� �� ��� ����� ���귮�� parameter ���� ������ �ִ�. ������ ���� ���� ���� �𵨿� ���ؼ� 3% �� ��Ȯ�ϴ�.

## Model shrinking hyperparameters

![mobile net width multiplier comparison](../mobilenetWidthMultiplier.png)

Width Multiplier ![alpha](https://latex.codecogs.com/svg.image?\alpha)�� �̿��� MobileNet�� ��Ҵ� ��Ȯ���� ���귮, ũ���� ��ȯ���� �̷�� ����. Width Mutliplier�� �پ�����, ��Ȯ���� �ε巴�� �پ��µ�, parameter�� ���� �ް��ϰ� �۾����� ![alpha](https://latex.codecogs.com/svg.image?\alpha=0.25)�϶� ��Ȯ���� �ް��ϰ� ��������. �� ���� parameter�� ���� �ʹ���� ��Ȯ�� ������� ã�Ⱑ ���������.

![mobile net resolution mutliplier comparison](../mobilenetResolutionMultiplier.png)

resolution multiplier ![rho](https://latex.codecogs.com/svg.image?\rho)�� �̿��� MobileNet�� ��Ҵ� ��Ȯ���� ���귮, ũ���� ��ȯ���� �̷�� ����. Resolution  Mutliplier�� �پ�����, ��Ȯ���� �ε巴�� �پ���.

![computation vs accuracy](../computationAccuracy.png)

���� �׷����� Ȯ���غ���, Computational complexity�� ���ؼ� ImageNet benchmark���� ����ϴ� ���� ���� �ִ�. ���⼭ x���� ���������� �����ϴ� ���� �˾ƾ��Ѵ�.

![parameter vs accuracy](../parameterAccuracy.png)

���� �׷����� parameter�� ������ ��Ȯ���� ���� ���̴�. ���⼭ parameter�� ���� ���� ���� ��Ȯ���� �ö󰡴� ���� ���� �ִ�. �׸��� parameter�� ���� resolution multiplier�ʹ� ���谡 ���� ���� Ȯ���� �� �ִ�.

![MobileNet vs popular models](../mobilenetPopularnet.png)

MobileNet�� �ٸ� ������ �ΰ��Ű���� ���غ��Ҵ�. MobileNet�� VGG16�� ����� ��Ȯ���� �����µ�, 32������ parameter ���� ������ �ְ� 27�� ���귮�� �۽��ϴ�. GoogleNet�� �񱳽�, MobileNet�� 1%���� ��Ȯ���� ������, 3�� ���� ������ ���� 1.5������ parameter ���� �����ϴ�.

![small mobile net vs popular models](../smallMobileNetPopularNet.png)

Width Multiplier�� 0.5�� ����ϰ� �ػ󵵸� ![160 160](https://latex.codecogs.com/svg.image?160\times160)���� ���� MobileNet�� Squeezenet�� Alexnet�� ���ؼ� Ȯ���ϰ� �����ϴ�. Squeezenet�� 22�� ���� ���귮���� �ұ��ϰ� 3%���� ���� ��Ȯ���� ������, AlexNet�� 45�� ���� parameter�� 9.4�� ���� ���귮���� �ұ��ϰ� 3%���� ��Ȯ���� �������ϴ�.


## [Link to Neural Net](../)
## [Link to Korean version](./Korean/)
## [Link to MobileNet V2](./V2/)