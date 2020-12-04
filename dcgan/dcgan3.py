# coding=utf-8

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

from keras.preprocessing import image
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

from models import gan_model, gan_model2

latent_dim = 32
height = 64
width = 64
channels = 3


def load_face(filename, required_size=(300, 300)):
    img = image.load_img(filename, target_size=required_size)
    x = image.img_to_array(img)
    #x = np.expand_dims(x, axis=0)
    return x

# 装入数据
def load_data():
    """Loads dataset.

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train)`.
    """
    required_size=(width, height)
    #dirname = '../1mage-data/1'
    dirname = '../1mage-data/3148203'
    file_list = os.listdir(dirname)

    num_train_samples = len(file_list)

    x_train = np.empty((num_train_samples, width, height, 3), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')

    for i, v in enumerate(file_list):
        fpath = os.path.join(dirname, v)
        x_train[i, :, :, :] = load_face(fpath, required_size=required_size)
        y_train[i] = 1

    y_train = np.reshape(y_train, (len(y_train), 1))

    if K.image_data_format() != 'channels_last':
        x_train = x_train.transpose(0, 3, 1, 2)

    return (x_train, y_train)



if __name__ == '__main__':

    # gan 包括 其他两个模型
    # gan.layers[1] == generator
    # gan.layers[2] == discriminator
    gan, generator, discriminator = gan_model(width, height, channels, latent_dim)


    # 训练

    # Load data
    (x_train, y_train) = load_data()

    # Normalize data
    x_train = x_train.reshape((x_train.shape[0],) + (height, width, channels)).astype('float32') / 255.
    print(x_train.shape)

    iterations = 10000
    batch_size = 6
    save_dir = 'images'

    # Start training loop
    start = 0
    for step in range(iterations):
        # Sample random points in the latent space
        random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))

        # Decode them to fake images
        generated_images = generator.predict(random_latent_vectors)

        # Combine them with real images
        stop = start + batch_size
        real_images = x_train[start: stop]
        combined_images = np.concatenate([generated_images, real_images])

        # Assemble labels discriminating real from fake images
        labels = np.concatenate([np.ones((batch_size, 1)),
                                 np.zeros((batch_size, 1))])
        # Add random noise to the labels - important trick!
        labels += 0.05 * np.random.random(labels.shape)

        # Train the discriminator
        d_loss = discriminator.train_on_batch(combined_images, labels)

        # sample random points in the latent space
        random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))

        # Assemble labels that say "all real images"
        misleading_targets = np.zeros((batch_size, 1))

        # Train the generator (via the gan model,
        # where the discriminator weights are frozen)
        a_loss = gan.train_on_batch(random_latent_vectors, misleading_targets)
        
        start += batch_size
        if start > len(x_train) - batch_size:
          start = 0


        # Occasionally save / plot
        if step % 50 == 0:
            # Save model weights
            gan.save_weights('gan.h5')

            # Print metrics
            print ("%d [D loss: %f] [A loss: %f]" % (step, d_loss, a_loss))

            # Save one generated image
            img = image.array_to_img(generated_images[0] * 255., scale=False)
            img.save(os.path.join(save_dir,  str(step) + '_generated' + '.png'))

            # Save one real image, for comparison
            #img = image.array_to_img(real_images[0] * 255., scale=False)
            #img.save(os.path.join(save_dir, str(step) + '_real' + '.png'))

    '''
    # 显示结果
    # Sample random points in the latent space
    random_latent_vectors = np.random.normal(size=(10, latent_dim))

    # Decode them to fake images
    generated_images = generator.predict(random_latent_vectors)

    for i in range(generated_images.shape[0]):
        img = image.array_to_img(generated_images[i] * 255., scale=False)
        plt.figure()
        plt.imshow(img)
        
    plt.show()
    '''