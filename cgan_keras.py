# -*- coding: utf-8 -*-

"""
@author:sunwill

A implemention of conditional genertive adversarial network using keras

reference paper:
arXiv:https://arxiv.org/abs/1411.1784

"""
import math
import numpy as np
from PIL import Image
from keras.datasets.mnist import load_data
from keras.optimizers import Adam
from keras.layers import concatenate
from keras.models import Sequential,Model
from keras.layers import Dense, Reshape, Conv2D, UpSampling2D, Input, Flatten, LeakyReLU, Dropout
from keras.losses import binary_crossentropy
from keras.utils import plot_model, to_categorical

image_size = 28
image_channel = 1

latent_size = 100
y_dim = 10
batch_size = 64
epochs = 30
learning_rate = 2e-4


def generator():
    cnn = Sequential()

    cnn.add(Dense(1024, input_dim=latent_size+y_dim, activation='tanh'))
    cnn.add(Dense(128 * 7 * 7, activation='tanh'))
    cnn.add(Reshape((7, 7, 128)))

    # upsample to (14, 14, ...)
    cnn.add(UpSampling2D(size=(2, 2)))
    cnn.add(Conv2D(256, 5, padding='same',
                   activation='tanh',
                   kernel_initializer='glorot_normal'))

    # upsample to (28, 28, ...)
    cnn.add(UpSampling2D(size=(2, 2)))
    cnn.add(Conv2D(128, 5, padding='same',
                   activation='tanh',
                   kernel_initializer='glorot_normal'))

    # take a channel axis reduction
    cnn.add(Conv2D(1, 2, padding='same',
                   activation='tanh',
                   kernel_initializer='glorot_normal'))

    input1 = Input(shape=(latent_size, ))
    input2 = Input(shape=(y_dim,))
    inputs = concatenate([input1, input2], axis=1)
    outs = cnn(inputs)

    return Model(inputs=[input1, input2], outputs=outs)


def discriminator():

    cnn = Sequential()

    cnn.add(Conv2D(32, 3, padding='same', strides=2,
                   input_shape=(28, 28, image_channel+y_dim)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Conv2D(64, 3, padding='same', strides=1))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Conv2D(128, 3, padding='same', strides=2))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Conv2D(256, 3, padding='same', strides=1))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Flatten())
    cnn.add(Dense(1, activation='sigmoid'))

    inputs = Input(shape=(image_size, image_size, image_channel+y_dim))
    outs = cnn(inputs)

    return Model(inputs=inputs, outputs=outs)


def disc_on_gen(g, d):

    input1 = Input(shape=(image_size, image_size, y_dim))
    input2 = Input(shape=(latent_size,))
    input3 = Input(shape=(y_dim,))

    g_out = g([input2, input3])
    d_input = concatenate([g_out, input1], axis=3)
    d.trainable = False
    outs = d(d_input)
    model = Model(inputs=[input1, input2, input3], outputs=outs)
    return model


def combine_images(images):
    num = images.shape[0]
    images = np.reshape(images, (-1, 28, 28))
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=images.dtype)
    for index, img in enumerate(images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = img[:, :]
    return image


def random_sample(x, y, batch_size):
    x_bs = []
    y_bs = []
    i = 0
    while (i < batch_size):
        rand = np.random.randint(0, x.shape[0])
        x_bs.append(x[rand])
        y_bs.append(y[rand])
        i += 1
    return np.array(x_bs), np.array(y_bs)


def train():
    (X_train, y_train), (X_test, y_test) = load_data()
    y_train = to_categorical(y_train)
    print X_train.shape  ## (60000,28,28)
    print y_train.shape  ## (60000,10)

    num_samples = X_train.shape[0]

    X_train = np.expand_dims(X_train, axis=3)
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5

    g = generator()
    d = discriminator()
    d_on_g = disc_on_gen(g, d)

    g_optimiper = Adam(lr=learning_rate)
    d_optimizer = Adam(lr=learning_rate)

    g.compile(loss=binary_crossentropy, optimizer=g_optimiper)
    d_on_g.compile(loss='binary_crossentropy', optimizer=g_optimiper)

    d.trainable = True
    d.compile(loss='binary_crossentropy', optimizer=d_optimizer, metrics=['accuracy'])

    plot_model(g, to_file='./model/cgan_generator.png', show_shapes=True)
    plot_model(d, to_file='./model/cgan_discriminator.png', show_shapes=True)
    plot_model(d_on_g, to_file='./model/cgan.png', show_shapes=True)
    p = 0
    for epoch in range(epochs):  ## 多轮训练
        print 'epoch {}/{}'.format(epoch + 1, epochs)

        for i in range(num_samples / batch_size):  ## 在每一轮迭代里面训练
            ## 随机生成高斯噪声
            noise = np.random.uniform(-1, 1, size=(batch_size, latent_size))
            ## 随机采样真实图片
            x_bs, y_bs = random_sample(X_train, y_train, batch_size)

            generate_images = g.predict([noise, y_bs], verbose=0)
            # print generate_images.shape
            ## 每经过500次训练输出生成图像
            if i % 500 == 0:
                images = combine_images(generate_images)
                images = images * 127.5 + 127.5
                Image.fromarray(images.astype(np.uint8)).save('./images/generated_{}_{}.png'.format(str(epoch + 1), i))

            ## 训练判别器
            xs = np.concatenate([generate_images, x_bs])
            ys = np.concatenate([y_bs, y_bs])
            ys = np.reshape(ys, newshape=[-1, 1, 1, y_dim])
            ys = np.tile(ys, [1, 28, 28, 1])
            X = np.concatenate([xs, ys], axis=3)

            y = [0] * batch_size + [1] * batch_size

            d_loss, acc = d.train_on_batch(X, y)
            if i % 100 == 0:
                print 'epoch {}, iter {},d_loss = {}, acc = {}'.format(epoch + 1, i, d_loss, acc)
            ## 训练生成器，此时需要固定判别器

            d.trainable = False

            g_loss = d_on_g.train_on_batch([ys[:batch_size], noise, y_bs], [1] * batch_size)
            if i % 100 == 0:
                print 'epoch {}, iter {},g_loss = {} '.format(epoch + 1, i, g_loss)

            d.trainable = True
            if i%500 == 0:
                noise = np.random.uniform(-1, 1, size=(100, latent_size))
                ys = np.zeros(shape=(100, y_dim))
                for c in range(10):
                    ys[c * 10:(c + 1) * 10, c] = 1
                generate_images = g.predict([noise, ys])
                images = combine_images(generate_images)
                images = images * 127.5 + 127.5
                Image.fromarray(images.astype(np.uint8)).save('./logs/cgan_{}.png'.format(p))
                p += 1

        g.save_weights('./images/cgan_generator.h5'.format(epoch))
        d.save_weights('./images/cgan_discriminator.h5'.format(epoch))


def generate(batch_size, flag=True):
    g = generator(latent_size)
    g.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
    g.load_weights('./logs/generator.h5')
    if flag:  ##生成多张图片，选出最好的几张图片
        d = discriminator(image_size, image_channel)
        d.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
        d.load_weights('./logs/discriminator.h5')
        noise = np.random.uniform(-1, 1, size=(batch_size * 10, latent_size))
        generate_images = g.predict(noise)
        d_pred = d.predict(generate_images)
        index = np.reshape(np.arange(0, batch_size * 10), (-1, 1))
        index_with_prob = list(np.append(index, d_pred, axis=1))
        index_with_prob.sort(key=lambda x: x[0], reverse=True)
        nices = np.zeros(shape=((batch_size,) + generate_images.shape[1:]))
        for i in range(batch_size):
            idx = int(index_with_prob[i][0])
            nices[i] = generate_images[idx]
        images = combine_images(nices)
    else:
        noise = np.random.uniform(-1, 1, size=(batch_size, latent_size))
        generate_images = g.predict(noise)
        images = combine_images(generate_images)

    Image.fromarray(images).save('./generated_images.png')


train()

# generate(64)

