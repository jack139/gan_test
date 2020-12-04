# coding=utf-8

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

from keras.preprocessing import image
from keras import backend as K
import numpy as np

from models import gan_model3, gan_model2, gan_model

latent_dim = 100
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


def generate_noise(n_samples, noise_dim):
  X = np.random.normal(0, 1, size=(n_samples, noise_dim))
  return X


if __name__ == '__main__':

    # gan 包括 其他两个模型
    # gan.layers[1] == generator
    # gan.layers[2] == discriminator
    gan, generator, discriminator = gan_model3(width, height, channels, latent_dim)


    # 训练

    # Load data
    (x_train, y_train) = load_data()

    # Normalize data
    x_train = (x_train.reshape((x_train.shape[0],) + (height, width, channels)).astype('float32') - 127.5) / 127.5
    print(x_train.shape)

    iterations = 10000
    batch_size = 6
    save_dir = 'images'

    # Start training loop
    start = 0
    for step in range(iterations):
        # Sample random points in the latent space
        random_latent_vectors = generate_noise(batch_size, latent_dim)

        # Decode them to fake images
        generated_images = generator.predict(random_latent_vectors)

        # Combine them with real images
        stop = start + batch_size
        real_images = x_train[start: stop]
        #combined_images = np.concatenate([generated_images, real_images])

        # Train on soft labels (add noise to labels as well)
        noise_prop = 0.05 # Randomly flip 5% of labels
        
        # Prepare labels for real data
        true_labels = np.zeros((batch_size, 1)) + np.random.uniform(low=0.0, high=0.1, size=(batch_size, 1))
        flipped_idx = np.random.choice(np.arange(len(true_labels)), size=int(noise_prop*len(true_labels)))
        true_labels[flipped_idx] = 1 - true_labels[flipped_idx]

        # Train discriminator on real data
        d_loss_true = discriminator.train_on_batch(real_images, true_labels)

        # Prepare labels for generated data
        gene_labels = np.ones((batch_size, 1)) - np.random.uniform(low=0.0, high=0.1, size=(batch_size, 1))
        flipped_idx = np.random.choice(np.arange(len(gene_labels)), size=int(noise_prop*len(gene_labels)))
        gene_labels[flipped_idx] = 1 - gene_labels[flipped_idx]
        
        # Train discriminator on generated data
        d_loss_gene = discriminator.train_on_batch(generated_images, gene_labels)

        d_loss = 0.5 * np.add(d_loss_true, d_loss_gene)

        # Assemble labels that say "all real images"
        misleading_targets = np.zeros((batch_size, 1))

        # Train generator
        random_latent_vectors = generate_noise(batch_size, latent_dim)
        a_loss = gan.train_on_batch(random_latent_vectors, misleading_targets)
        
        start += batch_size
        if start > len(x_train) - batch_size:
          start = 0

        # Occasionally save / plot
        if step % 50 == 0:
            # Save model weights
            gan.save_weights('gan.h5')

            # Print metrics
            print ("{} [D loss: {}] [A loss: {}]".format(step, d_loss, a_loss))

            # Save one generated image
            random_latent_vectors = generate_noise(batch_size, latent_dim)
            generated_images = generator.predict(random_latent_vectors)
            img = image.array_to_img(generated_images[0] * 127.5 + 127.5, scale=False)
            img.save(os.path.join(save_dir,  str(step) + '_generated' + '.png'))

            # Save one real image, for comparison
            #img = image.array_to_img(real_images[0] * 255., scale=False)
            #img.save(os.path.join(save_dir, str(step) + '_real' + '.png'))
