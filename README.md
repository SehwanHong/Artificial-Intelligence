# ResNet
Understanding of ResNet and different generations of residual networks

The Code is Consist of Two variation. resnet.py is ResNetVersion 1 and resnetPrenorm.py is commonly known ans ResNet Version 2.

For more detail about how these network works, please Read following websites:

* [ResNet](https://sehwanhong.github.io/ResNet/)
* [ResNet 한국어](https://sehwanhong.github.io/ResNet/Korean/)
* [ResNet Version 2](https://sehwanhong.github.io/ResNet/V2/)
* [ResNet Version 2 한국어](https://sehwanhong.github.io/ResNet/Korean/V2/)


# How to run these python Files.

```Cmd Line
python resnet.py [N] [D]
```
```Cmd Line
python resnetPrenorm.py [N] [D]
```

N is Number of Residual Block. If not entered assume it is 3.
D is Directory for Tensorboard logs. If not entered create log file in ./logs/
But if D is entered log file is created in ./logs/D/
