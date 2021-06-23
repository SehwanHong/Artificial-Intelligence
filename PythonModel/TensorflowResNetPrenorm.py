from tensorflow.keras.layers import Conv2D, Dense, ReLU, GlobalAveragePooling2D, BatchNormalization, Add, Flatten, Input, Softmax, ZeroPadding2D, AveragePooling2D
from tensorflow.keras import Model
from tensorflow.keras.initializers import VarianceScaling, HeNormal
from tensorflow.keras.regularizers import L2
import tensorflow as tf
import numpy as np


from tensorflow.keras.layers.experimental.preprocessing import RandomCrop, RandomFlip

weight_init = VarianceScaling()
weight_regularizer = L2(0.0001)

He_init = HeNormal(seed=123)

def ResBlock(inputs, channel, use_bias=True):
    BN_1 = BatchNormalization(momentum=0.9, epsilon=1e-5, center=True, scale=True)(inputs)
    ReLU_1 = ReLU()(BN_1)
    
    Conv_1 = Conv2D(channel, 3, 1, use_bias=use_bias, padding='same', kernel_initializer=He_init, kernel_regularizer=weight_regularizer)(ReLU_1)
    
    BN_2 = BatchNormalization(momentum=0.9, epsilon=1e-5, center=True, scale=True)(Conv_1)
    ReLU_2 = ReLU()(BN_2)
    
    Conv_2 = Conv2D(channel, 3, 1, use_bias=use_bias, padding='same', kernel_initializer=He_init, kernel_regularizer=weight_regularizer)(ReLU_2)
    
    Add_1 = Add()([Conv_2, inputs])
    
    return Add_1

def ResBlockDown(inputs, channel, use_bias=True):
    BN_1 = BatchNormalization(momentum=0.9, epsilon=1e-5, center=True, scale=True)(inputs)
    ReLU_1 = ReLU()(BN_1)
    
    Conv_1 = Conv2D(channel, 3, 2, use_bias=use_bias, padding='same', kernel_initializer=He_init, kernel_regularizer=weight_regularizer)(ReLU_1)
    #inputs = Conv2D(channel, 1, 2, use_bias=use_bias, padding='same', kernel_initializer=He_init, kernel_regularizer=weight_regularizer)(inputs)
    
    inputs = inputs[:,::2,::2,:]
    inputs = tf.pad(inputs, [[0,0],[0,0],[0,0],[0,channel//2]], "CONSTANT", name = "zeroPadding")
    
    BN_2 = BatchNormalization(momentum=0.9, epsilon=1e-5, center=True, scale=True)(Conv_1)
    ReLU_2 = ReLU()(BN_2)
    
    Conv_2 = Conv2D(channel, 3, 1, use_bias=use_bias, padding='same', kernel_initializer=He_init, kernel_regularizer=weight_regularizer)(ReLU_2)

    Add_1 = Add()([Conv_2, inputs])
    
    return Add_1

def ResNet(shape=(32,32,3), n=3, use_bias=True):
    inputs = Input(shape=shape)
    #zeroPadding = ZeroPadding2D((4,4))(inputs)
    #randomcrop = RandomCrop(32,32, seed=12341)(zeroPadding)
    #randomflip = RandomFlip('horizontal')(randomcrop)
    ch = 16
    
    Conv_1 = Conv2D(ch, 3, 1, use_bias=use_bias, padding='same', kernel_initializer=He_init, kernel_regularizer=weight_regularizer)(inputs)#(randomflip)
    
    ResNet_1 = Conv_1
    
    for i in range(n):
        ResNet_1 = ResBlock(ResNet_1, ch)
    
    ResNet_2 = ResBlockDown(ResNet_1, ch*2)
    
    for i in range(1,n):
        ResNet_2 = ResBlock(ResNet_2, ch*2)
    
    ResNet_3 = ResBlockDown(ResNet_2, ch*4)
    
    for i in range(1,n):
        ResNet_3 = ResBlock(ResNet_3, ch*4)
    
    BN = BatchNormalization(momentum=0.9, epsilon=1e-5, center=True, scale=True)(ResNet_3)
    relu = ReLU()(BN)
    
    GAP = GlobalAveragePooling2D()(relu)
    flatten = Flatten()(GAP)
    dense = Dense(10)(flatten)
    dense = Softmax()(dense)
    
    return Model(inputs=inputs, outputs=dense, name="ResNet_prenorm{}".format(6*n+2))