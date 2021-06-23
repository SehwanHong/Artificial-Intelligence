import tensorflow as tf
from keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, Dense, ReLU, GlobalAveragePooling2D, BatchNormalization, Add, Flatten, Input, Softmax, ZeroPadding2D, AveragePooling2D
from tensorflow.keras import Model
from tensorflow.keras.initializers import HeNormal, VarianceScaling
from tensorflow.keras.regularizers import L2
import numpy as np
import sys

# Data normalizer
def normalize(train_data, test_data):
    train_data = train_data.astype('float32') / 255
    test_data = test_data.astype('float32') / 255
    return train_data, test_data

# Data Loader
def load_cifar10():
    (train_data, train_target), (test_data, test_target) = cifar10.load_data()
    
    train_data, test_data = normalize(train_data, test_data)
    
    train_data_mean = np.mean(train_data, axis=0)
    train_data -= train_data_mean
    test_data -= train_data_mean
    
    train_target = to_categorical(train_target, len(np.unique(train_target)))
    test_target = to_categorical(test_target, len(np.unique(test_target)))
    
    seed = 123
    np.random.seed(seed)
    np.random.shuffle(train_data)
    np.random.seed(seed)
    np.random.shuffle(train_target)
    
    return train_data, train_target, test_data, test_target

# Learning Rate Scheduler
def lr_schedule(epoch):
    if epoch > 90:
        lr /= 10
    if epoch > 130:
        lr /= 10
    return lr

# Test Error metrics
def TestError(y_true, y_pred):
    accuracy = tf.keras.metrics.categorical_accuracy(y_true, y_pred)
    return (1 - accuracy) * 100


# CIFAR-10 Data Loading
train_data, train_target, test_data, test_target = load_cifar10()

# Residual Block 구성

weight_regularizer = L2(0.0001)
He_init = HeNormal(seed=1234)

def ResidualBlock(inputs, out_channel, stride=1, use_bias=True):
    Conv_1 = Conv2D(out_channel, 1, 1, use_bias=use_bias, kernel_initializer=He_init, kernel_regularizer = weight_regularizer)
    BN_1 = BatchNormalization(momentum=0.9, epsilon=1e-5)(Conv_1)
    ReLu_1 = ReLU()(BN_1)
    
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

def ResNet(shape=train_data.shape[1:], n = 3, use_bias=True):
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
