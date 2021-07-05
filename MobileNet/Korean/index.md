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

Although the MobileNet architecture is already small and low latency, many times a specific use case or application may require the model to be smaller and faster. To construct smaller and less computantionally expensive models, author introduce width multiplier.

The role of the width multiplier ![alpha](https://latex.codecogs.com/svg.image?\alpha) is to thin a network uniformly at each layer. For a given layer, with width multiplier ![alpha](https://latex.codecogs.com/svg.image?\alpha), the number of input channles M will become ![alpha](https://latex.codecogs.com/svg.image?\alpha&space;M), and visa-versa for the output channel. Thus tthe compuational cost will be following.

![Width Multiplier](https://latex.codecogs.com/svg.image?D_K&space;\times&space;D_K&space;\times&space;\alpha&space;M&space;\times&space;D_F&space;\times&space;D_F+\alpha&space;M&space;\times&space;\alpha&space;N&space;\times&space;D_F&space;\times&space;D_F)

Where ![alpha](https://latex.codecogs.com/svg.image?\alpha) is the value between 0 and 1. When ![alpha=1](https://latex.codecogs.com/svg.image?\alpha=1), it is baseline MobileNet, and when ![alpha<1](https://latex.codecogs.com/svg.image?\alpha<1), it is reduced MobileNets.

Using Width Multiplier, Computational Complexity is quadracially reduced by the factor of ![alpha^2](https://latex.codecogs.com/svg.image?\alpha^2)

## Resolution Multiplier: Reduced Representation

Resolution Mutliplier ![rho](https://latex.codecogs.com/svg.image?\rho) is applied to the input image and the internal representation of every layer to reduce the computation.

If we both apply width multiplier ![alpha](https://latex.codecogs.com/svg.image?\alpha) and resolution Multiplier ![rho](https://latex.codecogs.com/svg.image?\rho) then computational complexity for a single depthwise separable convolutional layer would be following.

![Resolution Multiplier and Width Mutliplier](https://latex.codecogs.com/svg.image?D_K&space;\times&space;D_K&space;\times&space;\alpha&space;M&space;\times&space;\rho&space;D_F&space;\times&space;\rho&space;D_F+\alpha&space;M&space;\times&space;\alpha&space;N&space;\times&space;\rho&space;D_F&space;\times&space;\rho&space;D_F)

# Experiment

Effect of depth wise convolution and the choice of shrinking by reducing the width of the network rather then the number of layers.

## Model Choices

First comparing the mobilenet with depthwise separable convolutions with a model built based on full convolutions. 

![depthwise separable vs full convolution mobile net](./compareConvolution.png)

Looking at the above image we could see that using depthwise separable convolution uses approximately nine times less computation but only reduces 1% accuracy.

Comparing thinner models with width multiplier to shallower models using less layers. To make MobileNet shalloweer, the 5 layers of separable filters with feature size ![14 14 512](https://latex.codecogs.com/svg.image?14\times14\times512) in Mobile net is removed.

![thin model vs shallow model](./compareThinShallow.png)

Looking at the table, both thinner and shallower model have similar computation numbers, such as Multi-adds and number of parameters. However, thinner model have 3% more accurate than using shallower model.

## Model shrinking hyperparameters

![mobile net width multiplier comparison](./mobilenetWidthMultiplier.png)

The accuracy, computation and size trade offs of shrinking the MobileNet architecure with the width multiplier ![alpha](https://latex.codecogs.com/svg.image?\alpha). Accuracy drops smoothly with lower width multiplier until the number of parameter is extremely small at ![alpha](https://latex.codecogs.com/svg.image?\alpha=0.25). At this point, number of parameter is extremely small thus have problem in finding correct classifier.

![mobile net resolution mutliplier comparison](./mobilenetResolutionMultiplier.png)

The accuracy, computation and size trade offs of shrinking the MobileNet architecture with the resolution multiplier ![rho](https://latex.codecogs.com/svg.image?\rho). Accuracy drop smoothly as the resolution decreases.

![computation vs accuracy](./computationAccuracy.png)

If we see the above graph, we compare computational complexity which is based on width multiplier and resolution multiplier and the accuracy. We could see the trend that larger computational number higher the accuracy for ImageNet benchmark. Must know that x axis are logarithmic.

![parameter vs accuracy](./parameterAccuracy.png)

The above graph compare the number of parameters and the accuracy. There is a trend that more number of parameters, better accuracy. Also notice that the number of parameters does not depend on the input resolutions.

![MobileNet vs popular models](./mobilenetPopularnet.png)

If we compare MobileNet to other popular neural networks, we would get above table. MobileNet have similar accuracy rate with VGG 16, but mobile net have approximately 32 times smaller in parameter size and 27 time smaller in computation. Also GoogleNet have approximately 3 times the computation and 1.5 times the parameters size than MobileNet but have lower accuracy rate.

![small mobile net vs popular models](./smallMobileNetPopularNet.png)

MobileNet using width multiplier of 0.5 and reduced resolution ![160 160](https://latex.codecogs.com/svg.image?160\times160) is better then both Squeezenet and AlexNet. Squeezenet have similar computation size but have 22 time more computation then MobileNet have 3% lower accuacy rate. Also AlexNet is 45 time more parameter and 9.4 time more computation have 3% less computation.

## Fine grained Recongition

![Stanford dog data](./StanfordDog.png)

Training fine grained recognition on the stanford Dogs dataset. MobileNet can almost achieve the state-of-art result using 9 times less computatoins and 7 times parameter size.

## [Link to Neural Net](../)
## [Link to Korean version](./Korean/)
## [Link to MobileNet V2](./V2/)