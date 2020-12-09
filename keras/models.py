#! -*- coding: utf-8 -*-

import numpy as np
from keras.layers import *
from keras.models import Model
from utils import SpectralNormalization

# 生成器和判别器 模型
def load_model(img_dim, z_dim, activation=None, sn=False, use_bias=True):
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
    z = BatchNormalization()(z)
    z = Activation('relu')(z)
    z = Reshape((f_size, f_size, max_num_channels))(z)

    for i in range(num_layers):
        num_channels = max_num_channels // 2**(i + 1)
        z = Conv2DTranspose(num_channels,
                            (4, 4),
                            strides=(2, 2),
                            padding='same')(z)
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

