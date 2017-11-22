# -*- coding: utf-8 -*-

"""
@author:sunwill

A implemention of conditional genertive adversarial network using tensorflow

reference paper:
arXiv:https://arxiv.org/abs/1411.1784

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.contrib.layers as tcl

save_dir = './logs/'

learning_rate = 0.0002
batch_size = 64
# X_dim = 784
image_size = 28
y_dim = 10
z_dim = 100

mnist = input_data.read_data_sets('./datasets/mnist/', one_hot=True)
images = mnist.train.images
labels = mnist.train.labels

print(images.shape)
print(labels.shape)


def get_shape(tensor):  # static shape
    return tensor.get_shape().as_list()


def lkrelu(x, slope=0.01):
    return tf.maximum(slope * x, x)


## building the generator
def generator(zs, ys, reuse=False, is_training=True):
    with tf.variable_scope('Generator', initializer=tf.truncated_normal_initializer(stddev=0.02), reuse=reuse):
        # TensorFlow Layers automatically create variables and calculate their
        # shape, based on the input.
        x = tf.concat([zs, ys], axis=1)

        g = tcl.fully_connected(x, 7 * 7 * 512, activation_fn=lkrelu, normalizer_fn=tcl.batch_norm)
        g = tf.reshape(g, (-1, 7, 7, 512))

        print(get_shape(g))
        g = tcl.conv2d(g, 128, 3, stride=1,  # (batch,7, 7 ,128)
                       activation_fn=lkrelu, normalizer_fn=tcl.batch_norm, padding='SAME',
                       weights_initializer=tf.random_normal_initializer(0, 0.02))
        print(get_shape(g))
        g = tcl.conv2d_transpose(g, 64, 4, stride=2,  # (batch,14, 14 ,64)
                                 activation_fn=lkrelu, normalizer_fn=tcl.batch_norm, padding='SAME',
                                 weights_initializer=tf.random_normal_initializer(0, 0.02))
        print(get_shape(g))
        g = tcl.conv2d_transpose(g, 1, 4, stride=2,  # (batch,28, 28 ,1)
                                 activation_fn=tf.nn.tanh, padding='SAME',
                                 weights_initializer=tf.random_normal_initializer(0, 0.02))
        print(get_shape(g))
    return g


## building the discriminatior
def discriminator(xs, ys, reuse=False, is_training=True):
    with tf.variable_scope('Discriminator', initializer=tf.truncated_normal_initializer(stddev=0.02), reuse=reuse):
        # Typical convolutional neural network to classify images.
        x = tf.layers.conv2d(xs, 32, kernel_size=[5, 5], padding='SAME')
        x = tf.concat([x, tf.tile(tf.reshape(ys, [-1, 1, 1, get_shape(ys)[-1]]),
                                  [1, tf.shape(x)[1], tf.shape(x)[2], 1])], axis=3)
        x = lkrelu(x)
        x = tf.layers.conv2d(x, 16, kernel_size=[5, 5], padding='SAME')
        x = tf.layers.batch_normalization(x, axis=3, training=is_training)
        x = lkrelu(x)
        x = tf.layers.conv2d(x, 8, kernel_size=[5, 5])
        x = tf.layers.batch_normalization(x, axis=3, training=is_training)
        x = lkrelu(x)
        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, 1024)
        x = tf.layers.dropout(x, rate=0.5)
        x = tf.layers.dense(x, 1)
    return tf.nn.sigmoid(x), x


def sample_data(z_dim, batch_size):
    return np.random.uniform(-1, 1, size=(batch_size, z_dim))


def combine_images(images):
    images = np.reshape(images, (-1, 28, 28))
    num = images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num) / width))
    shape = images.shape[1:]
    image = np.zeros((height * shape[0], width * shape[1]),
                     dtype=images.dtype)
    for index, img in enumerate(images):
        i = int(index / width)
        j = index % width
        image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1]] = img
    return image


noise = tf.placeholder(tf.float32, shape=(None, z_dim))
X = tf.placeholder(tf.float32, shape=(None, image_size, image_size, 1))
y = tf.placeholder(tf.float32, shape=(None, y_dim))

generated_images = generator(noise, y)  # pixes values between -1 and 1

d_real, d_real_logit = discriminator(X, y)
d_fake, d_fake_logit = discriminator(generated_images, y, reuse=True)

d_out = tf.concat(values=[d_real, d_fake], axis=0)
d_true = tf.concat(values=[tf.ones_like(d_real, dtype=tf.int64), tf.zeros_like(d_fake, dtype=tf.int64)], axis=0)
corrected = tf.equal(tf.cast(tf.greater(d_out, 0.5), tf.int64), d_true)
d_accuracy = tf.reduce_mean(tf.cast(corrected, tf.float32))

d_loss_real = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_real_logit), logits=d_real_logit))
d_loss_fake = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_fake_logit), logits=d_fake_logit))
d_loss = d_loss_real + d_loss_fake

gen_loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_fake_logit), logits=d_fake_logit))

gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')

gen_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5).minimize(gen_loss, var_list=gen_vars)
disc_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5).minimize(d_loss, var_list=disc_vars)

init = tf.global_variables_initializer()
sess = tf.Session()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(100000):
        if epoch % 1000 == 0:
            n_sample = 100
            noise_sample = sample_data(z_dim, n_sample)
            y_sample = np.zeros(shape=(n_sample, y_dim))
            for c in range(10):
                y_sample[c*10:(c+1)*10, c] = 1
            gen_images = sess.run(generated_images, feed_dict={noise: noise_sample, y: y_sample})
            images = combine_images(gen_images)
            print(' ---> ', images.max())
            images = images * 127.5 + 127.5
            print('----> ', images.max())
            Image.fromarray(images.astype(np.uint8)).save(save_dir + 'generated_{}.png'.format(epoch/1000))

        x_batch, y_batch = mnist.train.next_batch(batch_size)
        x_batch = np.reshape(x_batch, [-1, image_size, image_size, 1])
        ## if not did the normalization,the network would work,otherwise it did't work
        # x_batch = (x_batch.astype(np.uint8) - 127.5) / 127.5
        noise_batch = sample_data(z_dim, batch_size)
        d_loss_, _, acc = sess.run([d_loss, disc_train_op, d_accuracy],
                                   feed_dict={X: x_batch, y: y_batch, noise: noise_batch})
        g_loss_, _ = sess.run([gen_loss, gen_train_op], feed_dict={noise: noise_batch, y: y_batch})

        if epoch % 100 == 0:
            print('epoch {}, d_loss={}, acc = {}, g_loss={}'.format(epoch, d_loss_, acc, g_loss_))
            saver.save(sess, './model/cgan_tf.ckpt')
