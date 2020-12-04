# coding=utf-8

import keras
from keras import layers
from keras.preprocessing import image
from keras import backend as K
import numpy as np


# 返回获取特征卷积模型
def embeddings_model(width, height, channels, latent_dim, train=True):

    #  图片输入
    image_input = keras.Input(shape=(width, height, channels,))

    # 卷积 --- 与 生成器 顺序相反
    x = layers.Conv2D(channels, 7, activation='tanh', padding='same')(image_input)
    x = layers.LeakyReLU()(x)

    # Few more conv layers
    x = layers.Conv2D(256, 5, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(256, 5, padding='same')(x)
    x = layers.LeakyReLU()(x)

    # 缩小 一半
    x = layers.Conv2D(256, 4, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)

    # Then, add a convolution layer
    x = layers.Conv2D(128, 5, padding='same')(x)
    x = layers.LeakyReLU()(x)

    # 转换成向量
    x = layers.Flatten()(x)
    x = layers.Dense(latent_dim)(x)

    embeddings = keras.models.Model(image_input, x)
    embeddings.summary()

    return embeddings


# 返回gan模型 - 参考python书
def gan_model(width, height, channels, latent_dim, train=True):

    # 生成器
    generator_input = keras.Input(shape=(latent_dim,))

    # First, transform the input into a 16x16 128-channels feature map
    x = layers.Dense(128 * (width//2) * (height//2))(generator_input)
    x = layers.LeakyReLU()(x)
    x = layers.Reshape((width//2, height//2, 128))(x)

    # Then, add a convolution layer
    x = layers.Conv2D(256, 5, padding='same')(x)
    x = layers.LeakyReLU()(x)

    # Upsample to 32x32
    x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)

    # Few more conv layers
    x = layers.Conv2D(256, 5, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(256, 5, padding='same')(x)
    x = layers.LeakyReLU()(x)

    # Produce a 32x32 1-channel feature map
    x = layers.Conv2D(channels, 7, activation='tanh', padding='same')(x)
    generator = keras.models.Model(generator_input, x)
    generator.summary()


    # 判别器
    discriminator_input = layers.Input(shape=(height, width, channels))
    x = layers.Conv2D(128, 3)(discriminator_input)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(128, 4, strides=2)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(128, 4, strides=2)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(128, 4, strides=2)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Flatten()(x)

    # One dropout layer - important trick!
    x = layers.Dropout(0.4)(x)

    # Classification layer
    x = layers.Dense(1, activation='sigmoid')(x)

    discriminator = keras.models.Model(discriminator_input, x)
    discriminator.summary()

    if train:
        # To stabilize training, we use learning rate decay
        # and gradient clipping (by value) in the optimizer.
        discriminator_optimizer = keras.optimizers.RMSprop(lr=0.0008, clipvalue=1.0, decay=1e-8)
        discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')


    # 对抗网络

    # Set discriminator weights to non-trainable
    # (will only apply to the `gan` model)
    discriminator.trainable = False

    gan_input = keras.Input(shape=(latent_dim,))
    gan_output = discriminator(generator(gan_input))
    gan = keras.models.Model(gan_input, gan_output)

    if train:
        gan_optimizer = keras.optimizers.RMSprop(lr=0.0004, clipvalue=1.0, decay=1e-8)
        gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')

    return gan, generator, discriminator


# 返回gan模型, 参考 dcgan
def gan_model2(width, height, channels, latent_dim, train=True):

    gen_kernel_size = 3

    # 生成器
    generator_input = keras.Input(shape=(latent_dim,))

    # First, transform the input into a 16x16 128-channels feature map
    x = layers.Dense(128 * (width//4) * (height//4))(generator_input)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Reshape((width//4, height//4, 128))(x)

    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(128, gen_kernel_size, padding='same')(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(64, gen_kernel_size, padding='same')(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    # Produce a 32x32 1-channel feature map
    x = layers.Conv2D(channels, 7, activation='tanh', padding='same')(x)
    generator = keras.models.Model(generator_input, x)
    generator.summary()


    # 判别器
    discriminator_input = layers.Input(shape=(height, width, channels))
    x = layers.Conv2D(32, 3, strides=2, padding='same')(discriminator_input)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(64, 3, strides=2, padding='same')(x)
    x = layers.ZeroPadding2D(padding=((0,1),(0,1)))(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(128, 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(256, 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1, activation='sigmoid')(x)

    discriminator = keras.models.Model(discriminator_input, x)
    discriminator.summary()

    if train:
        discriminator.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(0.0002, 0.5))


    # 对抗网络

    # Set discriminator weights to non-trainable
    # (will only apply to the `gan` model)
    discriminator.trainable = False

    gan_input = keras.Input(shape=(latent_dim,))
    gan_output = discriminator(generator(gan_input))
    gan = keras.models.Model(gan_input, gan_output)

    if train:
        gan.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(0.0002, 0.5))

    return gan, generator, discriminator


# 返回gan模型, 参考 https://zhuanlan.zhihu.com/p/76357579
def gan_model3(width, height, channels, latent_dim, train=True):

    gen_kernel_size = 5

    # 生成器
    generator_input = keras.Input(shape=(latent_dim,))

    hid = layers.Dense(128 * (width//2) * (height//2))(generator_input)    
    hid = layers.LeakyReLU(alpha=0.1)(hid)
    hid = layers.BatchNormalization(momentum=0.9)(hid)
    hid = layers.LeakyReLU(alpha=0.1)(hid)
    hid = layers.Reshape(((width//2), (height//2), 128))(hid)

    hid = layers.Conv2D(128, kernel_size=gen_kernel_size, strides=1,padding='same')(hid)
    hid = layers.BatchNormalization(momentum=0.9)(hid)    
    #hid = layers.Dropout(0.5)(hid)
    hid = layers.LeakyReLU(alpha=0.1)(hid)

    hid = layers.Conv2DTranspose(128, 4, strides=2, padding='same')(hid)
    hid = layers.BatchNormalization(momentum=0.9)(hid)
    hid = layers.LeakyReLU(alpha=0.1)(hid)

    hid = layers.Conv2D(128, kernel_size=gen_kernel_size, strides=1, padding='same')(hid)
    hid = layers.BatchNormalization(momentum=0.9)(hid)
    #hid = layers.Dropout(0.5)(hid)
    hid = layers.LeakyReLU(alpha=0.1)(hid)

    hid = layers.Conv2D(128, kernel_size=gen_kernel_size, strides=1, padding='same')(hid)
    hid = layers.BatchNormalization(momentum=0.9)(hid)
    hid = layers.LeakyReLU(alpha=0.1)(hid)
                      
    hid = layers.Conv2D(channels, kernel_size=7, strides=1, padding="same")(hid)
    out = layers.Activation("tanh")(hid)

    generator = keras.models.Model(generator_input, out)
    generator.summary()

    # 判别器
    discriminator_input = layers.Input(shape=(height, width, channels))
    hid = layers.Conv2D(128, kernel_size=3, strides=1, padding='same')(discriminator_input)
    hid = layers.BatchNormalization(momentum=0.9)(hid)
    hid = layers.LeakyReLU(alpha=0.1)(hid)

    hid = layers.Conv2D(128, kernel_size=4, strides=2, padding='same')(hid)
    hid = layers.BatchNormalization(momentum=0.9)(hid)
    hid = layers.LeakyReLU(alpha=0.1)(hid)

    hid = layers.Conv2D(128, kernel_size=4, strides=2, padding='same')(hid)
    hid = layers.BatchNormalization(momentum=0.9)(hid)
    hid = layers.LeakyReLU(alpha=0.1)(hid)

    hid = layers.Conv2D(128, kernel_size=4, strides=2, padding='same')(hid)
    hid = layers.BatchNormalization(momentum=0.9)(hid)
    hid = layers.LeakyReLU(alpha=0.1)(hid)

    hid = layers.Flatten()(hid)
    hid = layers.Dropout(0.4)(hid)
    out = layers.Dense(1, activation='sigmoid')(hid)

    discriminator = keras.models.Model(discriminator_input, out)
    discriminator.summary()

    if train:
        discriminator.compile(optimizer=keras.optimizers.Adam(0.0002, 0.5), loss='binary_crossentropy')

    # 对抗网络

    # Set discriminator weights to non-trainable
    # (will only apply to the `gan` model)
    discriminator.trainable = False

    gan_input = keras.Input(shape=(latent_dim,))
    gan_output = discriminator(generator(gan_input))
    gan = keras.models.Model(gan_input, gan_output)

    if train:
        gan.compile(optimizer=keras.optimizers.Adam(0.0002, 0.5), loss='binary_crossentropy')

    return gan, generator, discriminator



# used for WGAN
def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


# 返回gan模型, 在model3基础 修改为 wgan
def gan_model4(width, height, channels, latent_dim, train=True):


    gen_kernel_size = 5

    # 生成器
    generator_input = keras.Input(shape=(latent_dim,))

    hid = layers.Dense(128 * (width//2) * (height//2))(generator_input)    
    hid = layers.LeakyReLU(alpha=0.1)(hid)
    hid = layers.BatchNormalization(momentum=0.9)(hid)
    hid = layers.LeakyReLU(alpha=0.1)(hid)
    hid = layers.Reshape(((width//2), (height//2), 128))(hid)

    hid = layers.Conv2D(128, kernel_size=gen_kernel_size, strides=1,padding='same')(hid)
    hid = layers.BatchNormalization(momentum=0.9)(hid)    
    #hid = layers.Dropout(0.5)(hid)
    hid = layers.LeakyReLU(alpha=0.1)(hid)

    hid = layers.Conv2DTranspose(128, 4, strides=2, padding='same')(hid)
    hid = layers.BatchNormalization(momentum=0.9)(hid)
    hid = layers.LeakyReLU(alpha=0.1)(hid)

    hid = layers.Conv2D(128, kernel_size=gen_kernel_size, strides=1, padding='same')(hid)
    hid = layers.BatchNormalization(momentum=0.9)(hid)
    #hid = layers.Dropout(0.5)(hid)
    hid = layers.LeakyReLU(alpha=0.1)(hid)

    hid = layers.Conv2D(128, kernel_size=gen_kernel_size, strides=1, padding='same')(hid)
    hid = layers.BatchNormalization(momentum=0.9)(hid)
    hid = layers.LeakyReLU(alpha=0.1)(hid)
                      
    hid = layers.Conv2D(channels, kernel_size=7, strides=1, padding="same")(hid)
    out = layers.Activation("tanh")(hid)

    generator = keras.models.Model(generator_input, out)
    generator.summary()

    # 判别器
    discriminator_input = layers.Input(shape=(height, width, channels))
    hid = layers.Conv2D(128, kernel_size=3, strides=1, padding='same')(discriminator_input)
    hid = layers.BatchNormalization(momentum=0.9)(hid)
    hid = layers.LeakyReLU(alpha=0.1)(hid)

    hid = layers.Conv2D(128, kernel_size=4, strides=2, padding='same')(hid)
    hid = layers.BatchNormalization(momentum=0.9)(hid)
    hid = layers.LeakyReLU(alpha=0.1)(hid)

    hid = layers.Conv2D(128, kernel_size=4, strides=2, padding='same')(hid)
    hid = layers.BatchNormalization(momentum=0.9)(hid)
    hid = layers.LeakyReLU(alpha=0.1)(hid)

    hid = layers.Conv2D(128, kernel_size=4, strides=2, padding='same')(hid)
    hid = layers.BatchNormalization(momentum=0.9)(hid)
    hid = layers.LeakyReLU(alpha=0.1)(hid)

    hid = layers.Flatten()(hid)
    #hid = layers.Dropout(0.4)(hid)
    out = layers.Dense(1)(hid)

    discriminator = keras.models.Model(discriminator_input, out)
    discriminator.summary()

    if train:
        discriminator.compile(optimizer=keras.optimizers.RMSprop(lr=0.00005), loss=wasserstein_loss)

    # 对抗网络

    # Set discriminator weights to non-trainable
    # (will only apply to the `gan` model)
    discriminator.trainable = False

    gan_input = keras.Input(shape=(latent_dim,))
    gan_output = discriminator(generator(gan_input))
    gan = keras.models.Model(gan_input, gan_output)

    if train:
        gan.compile(optimizer=keras.optimizers.RMSprop(lr=0.00005), loss=wasserstein_loss)

    return gan, generator, discriminator



# 返回gan模型, 参考 dcgan, 改为 wgan
def gan_model5(width, height, channels, latent_dim, train=True):

    gen_kernel_size = 4

    # 生成器
    generator_input = keras.Input(shape=(latent_dim,))

    # First, transform the input into a 16x16 128-channels feature map
    x = layers.Dense(128 * (width//4) * (height//4))(generator_input)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Reshape((width//4, height//4, 128))(x)

    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(128, gen_kernel_size, padding='same')(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(64, gen_kernel_size, padding='same')(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    # Produce a 32x32 1-channel feature map
    x = layers.Conv2D(channels, 7, activation='tanh', padding='same')(x)
    generator = keras.models.Model(generator_input, x)
    generator.summary()


    # 判别器
    discriminator_input = layers.Input(shape=(height, width, channels))
    x = layers.Conv2D(32, 3, strides=2, padding='same')(discriminator_input)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(64, 3, strides=2, padding='same')(x)
    x = layers.ZeroPadding2D(padding=((0,1),(0,1)))(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(128, 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(256, 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1)(x)

    discriminator = keras.models.Model(discriminator_input, x)
    discriminator.summary()

    if train:
        discriminator.compile(loss=wasserstein_loss, optimizer=keras.optimizers.RMSprop(lr=0.00005))


    # 对抗网络

    # Set discriminator weights to non-trainable
    # (will only apply to the `gan` model)
    discriminator.trainable = False

    gan_input = keras.Input(shape=(latent_dim,))
    gan_output = discriminator(generator(gan_input))
    gan = keras.models.Model(gan_input, gan_output)

    if train:
        gan.compile(loss=wasserstein_loss, optimizer=keras.optimizers.RMSprop(lr=0.00005))

    return gan, generator, discriminator
