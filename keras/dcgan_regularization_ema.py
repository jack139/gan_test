#! -*- coding: utf-8 -*-
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import numpy as np
#from scipy import misc
import glob
import imageio
from keras.models import Model
from keras.layers import *
from keras import backend as K
from keras.optimizers import Adam
from keras import losses
from PIL import Image
from utils import ExponentialMovingAverage


if not os.path.exists('samples'):
    os.mkdir('samples')

imgs = glob.glob('/media/gt/_dde_data/Datasets/CASIA-maxpy-clean/*/*.jpg')
#imgs = glob.glob('../../datasets/CASIA-maxpy-clean/*/*.jpg')
np.random.shuffle(imgs)


#height, width = misc.imread(imgs[0]).shape[:2]
height, width = imageio.imread(imgs[0]).shape[:2]
center_height = int((height - width) / 2)
img_dim = 64
z_dim = 100
EMA = False # whether use EMA

def imread(f):
    #x = misc.imread(f)
    x = imageio.imread(f)
    x = x[center_height:center_height + width, :]
    #x = misc.imresize(x, (img_dim, img_dim))
    im = Image.fromarray(x)
    x = np.array(im.resize((img_dim, img_dim), Image.BICUBIC))
    return x.astype(np.float32) / 255 * 2 - 1

def data_generator(batch_size=32):
    X = []
    while True:
        np.random.shuffle(imgs)
        for f in imgs:
            X.append(imread(f))
            if len(X) == batch_size:
                X = np.array(X)
                yield X
                X = []


# 判别器
x_in = Input(shape=(img_dim, img_dim, 3))
x = x_in

x = Conv2D(img_dim,
           (5, 5),
           strides=(2, 2),
           padding='same')(x)
x = LeakyReLU()(x)

for i in range(3):
    x = Conv2D(img_dim * 2**(i + 1),
               (5, 5),
               strides=(2, 2),
               padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

x = Flatten()(x)
x = Dense(1, activation='sigmoid')(x)

d_model = Model(x_in, x)
d_model.summary()


# 生成器
z_in = Input(shape=(z_dim, ))
z = z_in

z = Dense(4 * 4 * img_dim * 8)(z)
z = BatchNormalization()(z)
z = Activation('relu')(z)
z = Reshape((4, 4, img_dim * 8))(z)

for i in range(3):
    z = Conv2DTranspose(img_dim * 4 // 2**i,
                        (5, 5),
                        strides=(2, 2),
                        padding='same')(z)
    z = BatchNormalization()(z)
    z = Activation('relu')(z)

z = Conv2DTranspose(3,
                    (5, 5),
                    strides=(2, 2),
                    padding='same')(z)
z = Activation('tanh')(z)

g_model = Model(z_in, z)
g_model.summary()


# 整合模型（训练判别器）
x_in = Input(shape=(img_dim, img_dim, 3))
z_in = Input(shape=(z_dim, ))
g_model.trainable = False

x_fake = g_model(z_in)
x_real_score = d_model(x_in)
x_fake_score = d_model(x_fake)

d_train_model = Model([x_in, z_in],
                      [x_real_score, x_fake_score])

d_r_loss = K.binary_crossentropy(K.ones_like(x_real_score), x_real_score)
d_f_loss = K.binary_crossentropy(K.zeros_like(x_fake_score), x_fake_score)
d_loss = (d_r_loss + d_f_loss) / 2.0
d_train_model.add_loss(d_loss)
d_train_model.compile(optimizer=Adam(2e-4, 0.5))


# 整合模型（训练生成器）
g_model.trainable = True
d_model.trainable = False
x_fake = g_model(z_in)
x_fake_score = d_model(x_fake)

g_train_model = Model([x_in, z_in], x_fake_score)

# 正则项参考 https://kexue.fm/archives/5716
g_loss = K.binary_crossentropy(K.ones_like(x_fake_score), x_fake_score)
g_loss_plus = losses.mean_squared_error(x_in, x_fake) * 5 # new regularization
g_loss += g_loss_plus
g_train_model.add_loss(g_loss)
g_train_model.compile(optimizer=Adam(2e-4, 0.5))


# 检查模型结构
d_train_model.summary()
g_train_model.summary()

# EMA
if EMA:
    EMAer_g_train = ExponentialMovingAverage(g_train_model, 0.999) # 在模型compile之后执行
    EMAer_g_train.inject() # 在模型compile之后执行

# 采样函数
def sample(path):
    n = 9
    figure = np.zeros((img_dim * n, img_dim * n, 3))
    for i in range(n):
        for j in range(n):
            z_sample = np.random.randn(1, z_dim)
            x_sample = g_model.predict(z_sample)
            digit = x_sample[0]
            figure[i * img_dim:(i + 1) * img_dim,
                   j * img_dim:(j + 1) * img_dim] = digit
    figure = (figure + 1) / 2 * 255
    figure = np.round(figure, 0).astype(np.uint8)
    imageio.imwrite(path, figure)


iters_per_sample = 100
total_iter = 1000000
batch_size = 64
img_generator = data_generator(batch_size)

for i in range(total_iter):
    for j in range(1):
        z_sample = np.random.randn(batch_size, z_dim)
        d_loss = d_train_model.train_on_batch(
            [next(img_generator), z_sample], None)
    z_sample = np.random.randn(batch_size, z_dim)
    z_fake = g_model.predict(z_sample)
    for j in range(2):
        g_loss = g_train_model.train_on_batch(
            [z_fake, z_sample], None)
        if EMA:
            EMAer_g_train.ema_on_batch()
    if i % 10 == 0:
        print('iter: %s, d_loss: %s, g_loss: %s' % (i, d_loss, g_loss))
    if i % iters_per_sample == 0:
        if EMA:
            EMAer_g_train.apply_ema_weights() # 将EMA的权重应用到模型中
        sample('samples/test_%s.png' % i)
        g_train_model.save_weights('./g_train_model.weights')
        if EMA:
            EMAer_g_train.reset_old_weights() # 继续训练之前，要恢复模型旧权重
