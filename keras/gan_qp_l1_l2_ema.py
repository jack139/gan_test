#! -*- coding: utf-8 -*-
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import numpy as np
from keras.models import Model
from keras.layers import Input
from keras import backend as K
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from utils import sample, ExponentialMovingAverage
from models import load_model



if not os.path.exists('samples'):
    os.mkdir('samples')


img_dim = 64
z_dim = 100
EMA = False # whether use EMA
L1_or_L2 = 'L1' #  L1 或 L2
total_iter = 1000000
batch_size = 64
iters_per_sample = 100 # 采样频率


img_dir = '/media/gt/_dde_data/Datasets/CASIA-maxpy-clean'
#img_dir = '../../datasets/CASIA-maxpy-clean'

# 数据生成器
img_datagen = ImageDataGenerator(
    preprocessing_function=lambda x: x.astype(np.float32) / 255 * 2 - 1,
    zoom_range=0.0 # 缩放， 0.5 放大
)
img_generator = img_datagen.flow_from_directory(
    img_dir,
    target_size=(img_dim, img_dim),
    batch_size=batch_size,
    class_mode=None # 只生成图片，不生成标签
)


# 载入基本模型： 判别器，生成器
d_model, g_model = load_model(img_dim, z_dim, use_bias=False)
d_model.summary()
g_model.summary()


# 整合模型（训练判别器）
x_in = Input(shape=(img_dim, img_dim, 3))
z_in = Input(shape=(z_dim, ))
g_model.trainable = False

x_real = x_in
x_fake = g_model(z_in)

x_real_score = d_model(x_real)
x_fake_score = d_model(x_fake)

d_train_model = Model([x_in, z_in],
                      [x_real_score, x_fake_score])

d_loss = x_real_score - x_fake_score
d_loss = d_loss[:, 0]
if L1_or_L2=='L1':
    d_norm = 10 * K.mean(K.abs(x_real - x_fake), axis=[1, 2, 3])
else:
    d_norm = 10 * K.sqrt(K.mean(K.square(x_real - x_fake), axis=[1, 2, 3]))
d_loss = K.mean(- d_loss + 0.5 * d_loss**2 / d_norm)

d_train_model.add_loss(d_loss)
d_train_model.compile(optimizer=Adam(2e-4, 0.5))


# 整合模型（训练生成器）
g_model.trainable = True
d_model.trainable = False

x_real = x_in
x_fake = g_model(z_in)

x_real_score = d_model(x_real)
x_fake_score = d_model(x_fake)

g_train_model = Model([x_in, z_in],
                      [x_real_score, x_fake_score])

g_loss = K.mean(x_real_score - x_fake_score)

g_train_model.add_loss(g_loss)
g_train_model.compile(optimizer=Adam(2e-4, 0.5))


# 检查模型结构
d_train_model.summary()
g_train_model.summary()


# EMA
if EMA:
    EMAer_g_train = ExponentialMovingAverage(g_train_model, 0.999) # 在模型compile之后执行
    EMAer_g_train.inject() # 在模型compile之后执行


if __name__ == '__main__':

    import json

    n_size = 9
    Z = np.random.randn(n_size**2, z_dim)

    for i in range(total_iter):
        for j in range(2):
            x_sample = next(img_generator)
            if x_sample.shape[0]<batch_size: # 数据量有可能不能与batch_size对齐
                x_sample = next(img_generator)
            z_sample = np.random.randn(len(x_sample), z_dim)
            d_loss = d_train_model.train_on_batch(
                [x_sample, z_sample], None)
        for j in range(1):
            x_sample = next(img_generator)
            if x_sample.shape[0]<batch_size: # 数据量有可能不能与batch_size对齐
                x_sample = next(img_generator)
            z_sample = np.random.randn(len(x_sample), z_dim)
            g_loss = g_train_model.train_on_batch(
                [x_sample, z_sample], None)
            if EMA:
                EMAer_g_train.ema_on_batch()
        if i % 10 == 0:
            print('iter: %s, d_loss: %s, g_loss: %s' % (i, d_loss, g_loss))
        if i % iters_per_sample == 0:
            sample('samples/test_%s.png' % i, g_model, img_dim, z_dim, n=n_size, z_samples=Z)
            g_train_model.save_weights('./g_train_model.weights')
            if EMA:
                EMAer_g_train.apply_ema_weights() # 将EMA的权重应用到模型中
                sample('samples/test_ema_%s.png' % i, g_model, img_dim, z_dim, n=n_size, z_samples=Z)
                g_train_model.save_weights('./g_train_ema_model.weights')
                EMAer_g_train.reset_old_weights() # 继续训练之前，要恢复模型旧权重
