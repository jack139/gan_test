#! -*- coding: utf-8 -*-

import os
from distutils.util import strtobool

# 判断是tf.keras还是纯keras的标记
is_tf_keras = strtobool(os.environ.get('TF_KERAS', '0'))

if is_tf_keras:
    from tensorflow.keras.layers import *
    from tensorflow.keras.models import Model
    from tensorflow.keras import backend as K
else:
    from keras.layers import *
    from keras.models import Model
    from keras import backend as K

import numpy as np
from utils import SpectralNormalization


class ScaleShift(Layer):
    """平移缩放
    """
    def __init__(self, **kwargs):
        super(ScaleShift, self).__init__(**kwargs)
    def call(self, inputs):
        z, beta, gamma = inputs
        for i in range(K.ndim(z) - 2):
            beta = K.expand_dims(beta, 1)
            gamma = K.expand_dims(gamma, 1)
        return z * (gamma + 1) + beta

# SELF-MODE
def SelfModulatedBatchNormalization(h, c, z_dim):
    num_hidden = z_dim
    dim = K.int_shape(h)[-1]
    h = BatchNormalization(center=False, scale=False)(h)
    beta = Dense(num_hidden, activation='relu')(c)
    beta = Dense(dim)(beta)
    gamma = Dense(num_hidden, activation='relu')(c)
    gamma = Dense(dim)(gamma)
    return ScaleShift()([h, beta, gamma])


# 生成器和判别器 模型
def load_model(img_dim, z_dim, activation=None, sn=False, use_bias=True, self_mode=False):
    num_layers = int(np.log2(img_dim)) - 3
    if img_dim > 256:
        max_num_channels = img_dim * 4
    else:    
        max_num_channels = img_dim * 8
    f_size = img_dim // 2**(num_layers + 1)

    # 谱归一化
    if sn:
        SN = SpectralNormalization
    else:
        SN = lambda xx: xx

    # 判别器
    x_in = Input(shape=(img_dim, img_dim, 3))
    x = x_in

    for i in range(num_layers + 1):
        num_channels = max_num_channels // 2**(num_layers - i)
        x = SN(Conv2D(num_channels,
                   (4, 4),
                   strides=(2, 2),
                   use_bias=use_bias,
                   padding='same'))(x)
        if i > 0:
            x = SN(BatchNormalization())(x)
        x = LeakyReLU(0.2)(x)

    x = Flatten()(x)
    x = SN(Dense(1, use_bias=use_bias, activation=activation))(x)

    d_model = Model(x_in, x)


    # 生成器
    z_in = Input(shape=(z_dim, ))
    z = z_in

    z = Dense(f_size**2 * max_num_channels)(z)
    z = Reshape((f_size, f_size, max_num_channels))(z)
    if self_mode: # SELF-MODE
        z = SelfModulatedBatchNormalization(z, z_in, z_dim)
    else:
        z = BatchNormalization()(z)
    z = Activation('relu')(z)

    for i in range(num_layers):
        num_channels = max_num_channels // 2**(i + 1)
        z = Conv2DTranspose(num_channels,
                            (4, 4),
                            strides=(2, 2),
                            padding='same')(z)
        if self_mode:  # SELF-MODE
            z = SelfModulatedBatchNormalization(z, z_in, z_dim)
        else:
            z = BatchNormalization()(z)
        z = Activation('relu')(z)

    z = Conv2DTranspose(3,
                        (4, 4),
                        strides=(2, 2),
                        padding='same')(z)
    z = Activation('tanh')(z)

    g_model = Model(z_in, z)

    return d_model, g_model



# 生成器和判别器 模型
def load_model_2(img_dim, z_dim, activation=None, sn=False, use_bias=True):
    num_layers = int(np.log2(img_dim)) - 3
    if img_dim > 256:
        max_num_channels = img_dim * 4
    else:    
        max_num_channels = img_dim * 8
    f_size = img_dim // 2**(num_layers + 1)

    # 谱归一化
    if sn:
        SN = SpectralNormalization
    else:
        SN = lambda xx: xx

    # 判别器
    x_in = Input(shape=(img_dim, img_dim, 3))
    x = x_in

    x = SN(Conv2D(128, 3))(x)
    x = LeakyReLU()(x)
    x = SN(Conv2D(128, 4, strides=2))(x)
    x = LeakyReLU()(x)
    x = SN(Conv2D(128, 4, strides=2))(x)
    x = LeakyReLU()(x)
    x = SN(Conv2D(128, 4, strides=2))(x)
    x = LeakyReLU()(x)
    x = Flatten()(x)
    x = Dropout(0.4)(x)
    x = SN(Dense(1, use_bias=use_bias, activation=activation))(x)

    d_model = Model(x_in, x)

    # 生成器
    z_in = Input(shape=(z_dim, ))
    z = z_in

    z = Dense(128 * (img_dim//2) * (img_dim//2))(z_in)
    z = LeakyReLU()(z)
    z = Reshape((img_dim//2, img_dim//2, 128))(z)
    z = Conv2D(128, 4, padding='same')(z)
    z = LeakyReLU()(z)
    z = Conv2DTranspose(128, 4, strides=2, padding='same')(z)
    z = LeakyReLU()(z)
    z = Conv2D(128, 4, padding='same')(z)
    z = LeakyReLU()(z)
    z = Conv2D(128, 4, padding='same')(z)
    z = LeakyReLU()(z)
    z = Conv2D(3, 7, activation='tanh', padding='same')(z)

    g_model = Model(z_in, z)


    return d_model, g_model


# 生成器和判别器 模型
def load_model_3(img_dim, z_dim, activation=None, sn=False, use_bias=True):
    num_layers = int(np.log2(img_dim)) - 3
    if img_dim > 256:
        max_num_channels = img_dim * 4
    else:    
        max_num_channels = img_dim * 8
    f_size = img_dim // 2**(num_layers + 1)

    # 谱归一化
    if sn:
        SN = SpectralNormalization
    else:
        SN = lambda xx: xx

    # 判别器
    x_in = Input(shape=(img_dim, img_dim, 3))
    x = x_in

    x = SN(Conv2D(64, 3, strides=2, padding='same'))(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.25)(x)
    x = SN(Conv2D(128, 3, strides=2, padding='same'))(x)
    x = ZeroPadding2D(padding=((0,1),(0,1)))(x)
    x = SN(BatchNormalization(momentum=0.8))(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.25)(x)
    x = SN(Conv2D(256, 3, strides=2, padding='same'))(x)
    x = SN(BatchNormalization(momentum=0.8))(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.25)(x)
    x = SN(Conv2D(512, 3, strides=2, padding='same'))(x)
    x = SN(BatchNormalization(momentum=0.8))(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = SN(Dense(1, use_bias=use_bias, activation=activation))(x)

    d_model = Model(x_in, x)

    # 生成器
    z_in = Input(shape=(z_dim, ))
    z = z_in

    z = Dense(128 * (img_dim//4) * (img_dim//4))(z_in)
    z = LeakyReLU(alpha=0.2)(z)
    z = Reshape((img_dim//4, img_dim//4, 128))(z)
    z = UpSampling2D()(z)
    z = Conv2D(128, 4, padding='same')(z)
    z = BatchNormalization(momentum=0.8)(z)
    z = LeakyReLU(alpha=0.2)(z)
    z = UpSampling2D()(z)
    z = Conv2D(64, 4, padding='same')(z)
    z = BatchNormalization(momentum=0.8)(z)
    z = LeakyReLU(alpha=0.2)(z)
    z = Conv2D(3, 7, activation='tanh', padding='same')(z)

    g_model = Model(z_in, z)


    return d_model, g_model

