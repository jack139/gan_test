# coding=utf-8

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import keras
from keras import layers
from keras.preprocessing import image
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

from models import gan_model3, gan_model2, gan_model

latent_dim = 100
height = 64
width = 64
channels = 3

def generate_noise(n_samples, noise_dim):
  X = np.random.normal(0, 1, size=(n_samples, noise_dim))
  return X

if __name__ == '__main__':

    # gan 包括 其他两个模型
    # gan.layers[1] == generator
    # gan.layers[2] == discriminator
    gan, generator, discriminator = gan_model3(width, height, channels, latent_dim, train=False)

    # 载入权重
    gan.load_weights('gan.h5')

    generate_num = 10
    save_dir = 'images'

    random_latent_vectors = generate_noise(generate_num, latent_dim)

    generated_images = generator.predict(random_latent_vectors)

    for i in range(generate_num):
        # Save one generated image
        img = image.array_to_img(generated_images[i] * 127.5 + 127.5, scale=False)
        img.save(os.path.join(save_dir,  'gan_' + str(i) + '.png'))



