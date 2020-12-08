#! -*- coding: utf-8 -*-

import numpy as np
from keras.layers import *
from keras.models import Model

# 生成器和判别器 模型
def load_model(img_dim, z_dim, activation=None):
    num_layers = int(np.log2(img_dim)) - 3
    if img_dim > 256:
        max_num_channels = img_dim * 4
    else:    
        max_num_channels = img_dim * 8
    f_size = img_dim // 2**(num_layers + 1)

    # 判别器
    x_in = Input(shape=(img_dim, img_dim, 3))
    x = x_in

    for i in range(num_layers + 1):
        num_channels = max_num_channels // 2**(num_layers - i)
        x = Conv2D(num_channels,
                   (4, 4),
                   strides=(2, 2),
                   padding='same')(x)
        if i > 0:
            x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)

    x = Flatten()(x)
    x = Dense(1, activation=activation)(x)

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

