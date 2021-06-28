from tensorflow.keras.layers import Conv2D, Dense, ReLU, GlobalAveragePooling2D, BatchNormalization, Add, Flatten, Input, Softmax, ZeroPadding2D, AveragePooling2D
from tensorflow.keras import Model
from tensorflow.keras.initializers import VarianceScaling, HeNormal
from tensorflow.keras.regularizers import L2
import tensorflow as tf
import numpy as np

#Residual Block

weight_regularizer = L2(0.0001)
He_init = HeNormal(seed=1234)

def ResidualBlock(inputs, out_channel, stride=1, use_bias=True):
    #Convolution 1x1
    Conv_1 = Conv2D(out_channel, 1, 1, use_bias=use_bias, kernel_initializer=He_init, kernel_regularizer = weight_regularizer)
    BN_1 = BatchNormalization(momentum=0.9, epsilon=1e-5)(Conv_1)
    ReLu_1 = ReLU()(BN_1)
    
    #Convolution 3x3, when chaning dimention use stride 2
    Conv_2 = Conv2D(out_channel, 3, stride=stride, use_bias=use_bias, padding='same', kernel_initializer=He_init, kernel_regularizer=weight_regularizer)
    BN_2 = BatchNormalization(momentum=0.9, epsilon=1e-5)(Conv_2)
    ReLu_2= ReLU()(BN_2)
    
    Conv_3 = Conv2D(out_channel*4, 1, 1, use_bias=use_bias, kernel_initializer=He_init, kernel_regularizer=weight_regularizer)
    BN_3 = BatchNormalization(momentum=0.9, epsilon=1e-5)(Conv_3)
    
    if stride != 1:
        inputs = inputs[:,::2,::2,:]
        inputs = tf.pad(inputs, [(0,0), (0,0), (0,0), (0, out_channlel // 2)])
    Shortcut = Add()([BN_3, inputs])
    ReLu_3 = ReLU()(Shortcut)
    return ReLu_3

def ResNet(shape=(32,32,3), n = 3, use_bias=True):
    inputs = Input(shape=shape)
    ch = 64
    
    Conv_1 = Conv2D(ch, 3, 1, use_bias=use_bias, padding='same', kernel_initializer = He_init, kernel_regularizer = weight_regularizer)
    BN_1 = BatchNormalization(momentum=0.9, epsilon=1e-5)(Conv_1)
    ReLU_1 = ReLU()(BN_1)
    
    ResNet_1 = ReLU_1
    for i in range(n):
        ResNet_1 = ResidualBlock(ResNet_1, ch, use_bias=use_bias)


    ResNet_2 = ResidualBlock(ResNet_1, ch*2, 2, use_bias=use_bias)
    for i in range(1,n):
        ResNet_2 = ResidualBlock(ResNet_2, ch*2, use_bias=use_bias)


    ResNet_3 = ResidualBlock(ResNet_2, ch*4, 2, use_bias=use_bias)
    for i in range(1,n):
        ResNet_3 = ResidualBlock(ResNet_3, ch*4, use_bias=use_bias)
    
    GAP = GlobalAveragePooling2D()(ResNet_3)
    flatten = Flatten()(GAP)
    dense = Dense(10, activation='softmax', kernel_initializer=He_init, kernel_regularizer=weight_regularizer)(flatten)
    
    return Model(inputs=inputs, outputs=dense, name="ResNet{}".format(9*n+2))
