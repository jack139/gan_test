# coding=utf-8
# 实现基于“谱归一化”的Keras代码，实现方式是添加kernel_constraint
# 注意使用代码前还要修改Keras源码，修改
# keras/engine/base_layer.py的Layer对象的add_weight方法
# 修改方法见 https://kexue.fm/archives/6051#Keras%E5%AE%9E%E7%8E%B0
#
# !!!!!!! 已不需要修改源码 !!!!!!!!!! 见 https://kexue.fm/archives/6311
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
from PIL import Image
from utils import SpectralNormalization


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

x = SpectralNormalization(Conv2D(img_dim,
           (5, 5),
           strides=(2, 2),
           padding='same'))(x)
x = LeakyReLU()(x)

for i in range(3):
    x = SpectralNormalization(Conv2D(img_dim * 2**(i + 1),
               (5, 5),
               strides=(2, 2),
               padding='same'))(x)
    x = SpectralNormalization(BatchNormalization())(x)
    x = LeakyReLU()(x)

x = Flatten()(x)
x = SpectralNormalization(Dense(1, use_bias=False))(x)

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

d_loss = K.mean(x_fake_score - x_real_score)
d_train_model.add_loss(d_loss)
d_train_model.compile(optimizer=Adam(2e-4, 0.5))


# 整合模型（训练生成器）
g_model.trainable = True
d_model.trainable = False
x_fake_score = d_model(g_model(z_in))

g_train_model = Model(z_in, x_fake_score)
g_train_model.add_loss(K.mean(- x_fake_score))
g_train_model.compile(optimizer=Adam(2e-4, 0.5))


# 检查模型结构
d_train_model.summary()
g_train_model.summary()


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
    for j in range(5):
        z_sample = np.random.randn(batch_size, z_dim)
        d_loss = d_train_model.train_on_batch(
            [next(img_generator), z_sample], None)
    for j in range(1):
        z_sample = np.random.randn(batch_size, z_dim)
        g_loss = g_train_model.train_on_batch(z_sample, None)
    if i % 10 == 0:
        print('iter: %s, d_loss: %s, g_loss: %s' % (i, d_loss, g_loss))
    if i % iters_per_sample == 0:
        sample('samples/test_%s.png' % i)
        g_train_model.save_weights('./g_train_model.weights')
