# -*- coding: utf-8 -*-

"""
@author:sunwill

An implemention of generative adversarial network using keras

reference paper:
arXiv:https://arxiv.org/abs/1406.2661
"""
import math
import numpy as np
from PIL import Image
from keras.datasets.mnist import load_data
from keras.optimizers import SGD
from keras.losses import binary_crossentropy
from keras.utils import plot_model
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Reshape, \
    UpSampling2D


image_size = 28
image_channel = 1

latent_size = 100
batch_size = 32
epochs = 30
learning_rate = 1e-3


## 定义生成器g
def generator(latent_size):
    input = Input(shape=(latent_size,))

    x = Dense(128, activation='tanh')(input)
    x = Dense(128 * 7 * 7, activation='tanh')(x)
    x = BatchNormalization(axis=1)(x)
    x = Reshape((7, 7, 128))(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(64, kernel_size=(3, 3), activation='tanh', padding='same')(x)
    x = UpSampling2D(size=(2, 2))(x)
    output = Conv2D(1, kernel_size=(5, 5), activation='tanh', padding='same')(x)
    model = Model(inputs=input, outputs=output)

    return model


## 定义判别器d
def discriminator(image_size, image_channel):
    input = Input(shape=(image_size, image_size, image_channel))

    x = Conv2D(64, kernel_size=(5, 5), activation='tanh', padding='same')(input)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(128, kernel_size=(5, 5), activation='tanh', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)

    x = Dense(1024, activation='tanh')(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input, outputs=output)
    return model


## 将生成器和判别器拼接成一个模型，用于训练生成器
def generator_on_disciminator(g, d):
    ## 将前面定义的生成器架构和判别器架构组拼接成一个大的神经网络，用于判别生成的图片
    model = Sequential()
    ## 先添加生成器架构，再令d不可训练，即固定d
    ## 因此在给定d的情况下训练生成器，即通过将生成的结果投入到判别器进行辨别而优化生成器
    model.add(g)
    d.trainable = False
    model.add(d)
    return model


def random_sample(data, batch_size):
    ret_data = []
    i = 0
    while (i < batch_size):
        rand = np.random.randint(0, data.shape[0])
        ret_data.append(data[rand])
        i += 1
    return np.array(ret_data)


def combine_images(images):
    #生成图片拼接
    num = images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=images.dtype)
    for index, img in enumerate(images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image

(X_train, y_train), (X_test, y_test) = load_data()
print X_train.shape ## (60000,28,28)
print X_test.shape ## (10000,28,28)

num_samples = X_train.shape[0]

X_train = np.expand_dims(X_train, axis=3)
X_train = (X_train.astype(np.float32)-127.5)/127.5

g = generator(latent_size)
d = discriminator(image_size, image_channel)
g_on_d = generator_on_disciminator(g, d)

g_optimiper = SGD(lr=0.001, momentum=0.9, nesterov=True)
d_optimizer = SGD(lr=0.001, momentum=0.9, nesterov=True)

g.compile(loss=binary_crossentropy, optimizer='SGD')
g_on_d.compile(loss='binary_crossentropy', optimizer=g_optimiper)

d.trainable = True
d.compile(loss='binary_crossentropy', optimizer=d_optimizer, metrics=['accuracy'])

plot_model(g, to_file='./model/gan_generator.png', show_shapes=True)
plot_model(d, to_file='./model/gan_discriminator.png', show_shapes=True)

for epoch in range(epochs):## 多轮训练
    print 'epoch {}/{}'.format(epoch+1, epochs)

    for i in range(num_samples/batch_size):## 在每一轮迭代里面训练
        ## 随机生成高斯噪声
        # noise = np.random.uniform(-1, 1, size=(batch_size, image_size, image_size, image_channel))
        noise = np.random.uniform(-1, 1, size=(batch_size, 100))
        ## 随机采样真实图片
        real_samples = random_sample(X_train, batch_size)

        generate_images = g.predict(noise, verbose=0)
        # print generate_images.shape
        ## 每经过500次训练输出生成图像
        if i%500 == 0:
            images = combine_images(generate_images)
            images = images*127.5+127.5
            Image.fromarray(images.astype(np.uint8)).save('./images/generated_{}_{}.png'.format(str(epoch+1), i))

        ## 训练判别器
        X = np.concatenate([generate_images, real_samples])
        y = [0]*batch_size+[1]*batch_size

        d_loss, acc = d.train_on_batch(X, y)
        if i%100 == 0:
            print 'epoch {}, iter {},d_loss = {}, acc = {}'.format(epoch+1, i, d_loss, acc)
        ## 训练生成器，此时需要固定判别器

        d.trainable = False

        g_loss = g_on_d.train_on_batch(noise, [1]*batch_size)
        if i % 100 == 0:
            print 'epoch {}, iter {},g_loss = {} '.format(epoch+1, i, g_loss)

        d.trainable = True

    g.save_weights('./logs/generator_epoch{}.h5'.format(epoch))
    d.save_weights('./logs/discriminator_eopch{}.h5'.format(epoch))


def generate(batch_size,flag=True):
    g = generator(latent_size)
    g.compile(optimizer='SGD', loss='binary_crossentropy')
    g.load_weights('./logs/generator.h5')
    if flag: ##生成多张图片，选出最好的几张图片
        d = discriminator(image_size, image_channel)
        d.compile(optimizer='SGD', loss='binary_crossentropy')
        d.load_weights('./logs/discriminator.h5')
        noise = np.random.uniform(-1, 1, size=(batch_size*10, latent_size))
        generate_images = g.predict(noise)
        d_pred = d.predict(generate_images)
        index = np.reshape(np.arange(0, batch_size*10), (-1, 1))
        index_with_prob = list(np.append(index, d_pred, axis=1))
        index_with_prob.sort(key=lambda x:x[0], reverse=True)
        nices = np.zeros(shape=((batch_size,)+generate_images.shape[1:]))
        for i in range(batch_size):
            idx = int(index_with_prob[i][0])
            nices[i] = generate_images[idx]
        images = combine_images(nices)
    else:
        noise = np.random.uniform(-1, 1, size=(batch_size, latent_size))
        generate_images = g.predict(noise)
        images = combine_images(generate_images)

    Image.fromarray(images).save('./generated_images.png')

generate(64)