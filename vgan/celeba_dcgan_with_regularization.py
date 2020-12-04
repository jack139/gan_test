# coding=utf-8

import glob
import utils
import traceback
import numpy as np
import tensorflow as tf
import models_64x64 as models


""" param """
epoch = 50
batch_size = 64
lr = 0.0002
z_dim = 100
with_bn = True # whether us BN layer


gpu_id = 0

""" data """
# you should prepare your own data in ./data/img_align_celeba
# celeba original size is [218, 178, 3]
origin_height = 250 #  CASIA-maxpy-clean's size is  250*250
origin_with = 250


def preprocess_fn(img):
    crop_size = 200 # use the center part
    re_size = 64
    img = tf.image.crop_to_bounding_box(img, (origin_height - crop_size) // 2, (origin_with - crop_size) // 2, crop_size, crop_size)
    img = tf.to_float(tf.image.resize_images(img, [re_size, re_size], method=tf.image.ResizeMethod.BICUBIC)) / 127.5 - 1
    return img


img_paths = glob.glob('/media/gt/_dde_data/Datasets/CASIA-maxpy-clean/*/*.jpg')
#img_paths = glob.glob('../../datasets/CASIA-maxpy-clean/*/*.jpg')
data_pool = utils.DiskImageData(img_paths, batch_size, shape=[origin_height, origin_with, 3], preprocess_fn=preprocess_fn)


""" graphs """
with tf.device('/gpu:%d' % gpu_id):
    """ models """
    generator = models.generator
    discriminator = models.discriminator

    """ graph """
    # inputs
    real = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
    z = tf.placeholder(tf.float32, shape=[None, z_dim])

    # generate
    fake = generator(z, reuse=False, with_bn=with_bn)

    # dicriminate
    r_logit = discriminator(real, reuse=False, with_bn=with_bn)
    f_logit = discriminator(fake, with_bn=with_bn)

    # losses
    d_r_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(r_logit), r_logit)
    d_f_loss = tf.losses.sigmoid_cross_entropy(tf.zeros_like(f_logit), f_logit)
    d_loss = (d_r_loss + d_f_loss) / 2.0
    g_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(f_logit), f_logit)
    g_loss_plus = tf.losses.mean_squared_error(real, fake) * 5 # new regularization
    g_loss += g_loss_plus

    # otpims
    d_var = utils.trainable_variables('discriminator')
    g_var = utils.trainable_variables('generator')
    d_step = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5).minimize(d_loss, var_list=d_var)
    g_step = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5).minimize(g_loss, var_list=g_var)

    # summaries
    d_summary = utils.summary({d_loss: 'd_loss'})
    g_summary = utils.summary({g_loss: 'g_loss'})

    # sample
    f_sample = generator(z, training=False, with_bn=with_bn)


""" train """
""" init """
# session
sess = utils.session()
# iteration counter
it_cnt, update_cnt = utils.counter()
# saver
saver = tf.train.Saver(max_to_keep=5)
# summary writer
summary_writer = tf.summary.FileWriter('./summaries/celeba_dcgan_regular', sess.graph)

""" initialization """
ckpt_dir = './checkpoints/celeba_dcgan_regular'
utils.mkdir(ckpt_dir + '/')
if not utils.load_checkpoint(ckpt_dir, sess):
    sess.run(tf.global_variables_initializer())

""" train """
try:
    z_ipt_sample = np.random.normal(size=[100, z_dim])

    batch_epoch = len(data_pool) // (batch_size)
    max_it = epoch * batch_epoch
    for it in range(sess.run(it_cnt), max_it):
        sess.run(update_cnt)

        # which epoch
        epoch = it // batch_epoch
        it_epoch = it % batch_epoch + 1

        # batch data
        real_ipt = data_pool.batch()
        z_ipt = np.random.normal(size=[batch_size, z_dim])

        # train D
        d_summary_opt, _ = sess.run([d_summary, d_step], feed_dict={real: real_ipt, z: z_ipt})
        summary_writer.add_summary(d_summary_opt, it)

        # train G
        z_ipt = np.random.normal(size=[batch_size, z_dim])
        old_fake = sess.run(fake, feed_dict={z: z_ipt})
        sess.run([g_step], feed_dict={z: z_ipt, real: old_fake})
        g_summary_opt, _ = sess.run([g_summary, g_step], feed_dict={z: z_ipt, real: old_fake})
        summary_writer.add_summary(g_summary_opt, it)

        # display
        if it % 1 == 0:
            print("Epoch: (%3d) (%5d/%5d)" % (epoch, it_epoch, batch_epoch))

        # save
        if (it + 1) % 1000 == 0:
            save_path = saver.save(sess, '%s/Epoch_(%d)_(%dof%d).ckpt' % (ckpt_dir, epoch, it_epoch, batch_epoch))
            print('Model saved in file: % s' % save_path)

        # sample
        if (it + 1) % 100 == 0:
            f_sample_opt = sess.run(f_sample, feed_dict={z: z_ipt_sample})

            save_dir = './sample_images_while_training/celeba_dcgan_regular'
            utils.mkdir(save_dir + '/')
            utils.imwrite(utils.immerge(f_sample_opt, 10, 10), '%s/Epoch_(%d)_(%dof%d).jpg' % (save_dir, epoch, it_epoch, batch_epoch))

except Exception as e:
    traceback.print_exc()
finally:
    print(" [*] Close main session!")
    sess.close()