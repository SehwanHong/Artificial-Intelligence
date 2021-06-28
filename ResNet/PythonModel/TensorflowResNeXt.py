from tensorflow.keras.layers import Conv2D, Dense, ReLU, GlobalAveragePooling2D, BatchNormalization, Add, Flatten, Input, Softmax, ZeroPadding2D, AveragePooling2D
from tensorflow.keras import Model
from tensorflow.keras.initializers import VarianceScaling, HeNormal
from tensorflow.keras.regularizers import L2
import tensorflow as tf
import numpy as np

weight_init = VarianceScaling()
weight_regularizer = L2(0.0001)

He_init = HeNormal(seed=123)

def ResNextBlock(inputs, ch, d, cardinality, stride=1, use_bias=True):
    bottleneck_width = d * cardinality
    out_channel = ch * 4
   
    Conv_1 = Conv2D(bottleneck_width, 1, 1, use_bias=use_bias, kernel_initializer=He_init, kernel_regularizer=weight_regularizer)(inputs)
    BN_1 = BatchNormalization(momentum=0.9, epsilon=1e-5)(Conv_1)
    ReLu_1 = ReLU()(BN_1)
   
    Conv_2 = Conv2D(bottleneck_width, 3, strides=stride, use_bias=use_bias, padding='same', groups=cardinality, kernel_initializer=He_init, kernel_regularizer=weight_regularizer)(ReLu_1)
    BN_2 = BatchNormalization(momentum=0.9, epsilon=1e-5)(Conv_2)
    ReLu_2 = ReLU()(BN_2)
   
    Conv_3 = Conv2D(out_channel, 1, 1, use_bias=use_bias, kernel_initializer=He_init, kernel_regularizer=weight_regularizer)(ReLu_2)
    BN_3 = BatchNormalization(momentum=0.9, epsilon=1e-5)(Conv_3)

    
    if stride != 1:
        inputs = inputs[:,::2,::2,:]
#         inputs = tf.pad(inputs, [[0,0], [0,0], [0,0], [0, bottleneck_width]], "CONSTANT", name="zeroPadding")
       
    if BN_3.shape[3] != inputs.shape[3]:
        inputs = tf.pad(inputs, [[0,0], [0,0], [0,0], [0,BN_3.shape[3]-inputs.shape[3]]], "CONSTANT", name="zeroPadding")
       
    Shortcut = Add()([BN_3, inputs])
    ReLu_3 = ReLU()(Shortcut)
    return ReLu_3

def ResNeXt(shape=(32,32,3), n = 3, d=64, cardinality=1, use_bias=True):
    inputs = Input(shape=shape)
    ch = 64
   
    Conv_1 = Conv2D(ch, 3, 1, use_bias=use_bias, padding='same', kernel_initializer=He_init, kernel_regularizer=weight_regularizer)(inputs)
    BN_1 = BatchNormalization(momentum=0.9, epsilon=1e-5)(Conv_1)
    ReLU_1 = ReLU()(BN_1)
   
    ResNext_1 = ReLU_1
    for i in range(n):
        ResNext_1 = ResNextBlock(ResNext_1, ch, d, cardinality, use_bias=use_bias)
       
    ResNext_2 = ResNextBlock(ResNext_1, ch*2, d*2, cardinality, 2, use_bias=use_bias)
    for i in range(1,n):
        ResNext_2 = ResNextBlock(ResNext_2, ch*2, d*2, cardinality, use_bias=use_bias)
       
    ResNext_3 = ResNextBlock(ResNext_2, ch*4, d*4, cardinality, 2, use_bias=use_bias)
    for i in range(1,n):
        ResNext_3 = ResNextBlock(ResNext_3, ch*4, d*4, cardinality, use_bias=use_bias)
   
    GAP = GlobalAveragePooling2D()(ResNext_3)
    flatten = Flatten()(GAP)
    dense = Dense(10, activation='softmax', kernel_initializer=He_init, kernel_regularizer=weight_regularizer)(flatten)
   
    return Model(inputs=inputs, outputs=dense, name="ResNext{}({}x{}d)".format(9*n+2, cardinality, d))
