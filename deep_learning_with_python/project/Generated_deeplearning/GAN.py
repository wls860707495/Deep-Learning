# GAN
# GAN生成器网络
# 训练循环的大致流程
# (1)从潜在空间中抽取随机的点（随机噪声）
# (2)利用这个随机噪声用generator生成图像
# (3)将生成图像与真实图像混合
# (4)使用这些混合后的图像以及相应的标签(真实图像为“真”，生成的图像为“假”)
# (5)在潜在空间中随机抽取新的点
# (6)使用这些随机向量以及全部是“真实图像”的标签来训练gan。这会更新生成器的权重（只更新生成器的权重，因为判别器在gan中被冻结），其更新方向是使得判别器能够将生成图像预测为“真实图像”。这个过程是训练生成器去欺骗判别器。

import keras
from keras import layers
import numpy as np
import os
from keras.preprocessing import image

latent_dim = 32
height = 32
width = 32
channels = 3

generator_input = keras.Input(shape=(latent_dim,))

# 将输入转换为大小为16x16的128个通道的特征图
x = layers.Dense(128 * 16 * 16)(generator_input)
x = layers.LeakyReLU()(x)
x = layers.Reshape((16, 16, 128))(x)

x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)

# 上采样为32x32
x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)

# 生成一个大小为32x32的单通道特征图
x = layers.Conv2D(channels, 7, activation='tanh', padding='same')(x)
generator = keras.models.Model(generator_input, x)
generator.summary()

# GAN判别器网络
discriminator_input = layers.Input(shape=(height, width, channels))
x = layers.Conv2D(128, 3)(discriminator_input)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Flatten()(x)

x = layers.Dropout(0.4)(x)

# 将判别器模型实例化，它将形状为(32,32,3)的输入转换为一个二进制分类决策（真 / 假）
x = layers.Dense(1, activation='sigmoid')(x)

discriminator = keras.models.Model(discriminator_input, x)
discriminator.summary()

discriminator_optimizer = keras.optimizers.RMSprop(lr=0.0008, clipvalue=1.0, decay=1e-8)

discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')

# 对抗网络
discriminator.trainable = False    # 将判别器权重设置为不可训练

gan_input = keras.Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = keras.models.Model(gan_input, gan_output)
gan_optimizer = keras.optimizers.RMSprop(lr=0.0004, clipvalue=1.0, decay=1e-8)
gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')

# 实现GAN的训练
(x_train, y_train), (_, _) = keras.datasets.cifar10.load_data()
x_train = x_train[y_train.flatten() == 6]
x_train = x_train.reshape((x_train.shape[0],) + (height, width, channels)).astype('float32')/255.
iterations = 10000
batch_size = 20
save_dir = 'D:\generate_picture'

start = 0
for step in range(iterations):
    # 在潜在空间中采样随机点
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))

    # 将这些点解码为虚假图像
    generated_images = generator.predict(random_latent_vectors)

    # 将这些虚假图像与真实图像合在一起
    stop = start + batch_size
    real_images = x_train[start: stop]
    combined_images = np.concatenate([generated_images, real_images])

    # 合并标签，区分真实和虚假的图像
    labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])
    # 向标签中添加随机噪声，这是一个很重要的技巧，防止GAN在寻找动态平衡时“卡住”
    labels += 0.05 * np.random.random(labels.shape)

    # 训练判别器
    d_loss = discriminator.train_on_batch(combined_images, labels)

    # 在潜在空间中 采样随机点
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))

    # 合并标签，全部是 “真实图像”（这是在撒谎）
    misleading_targets = np.zeros((batch_size, 1))

    # 通过gan模型来训练生成器（此时冻结判别器权重）
    a_loss = gan.train_on_batch(random_latent_vectors, misleading_targets)

    start += batch_size
    if start > len(x_train) - batch_size:
        start = 0

    if step % 100 == 0:   # 每100步保存并绘图
        gan.save_weights('gan.h5')

        print('discriminator loss:', d_loss)
        print('adversarial loss:', a_loss)

        # 保存一张生成图像
        img = image.array_to_img(generated_images[0] * 255., scale=False)
        img.save(os.path.join(save_dir, 'generated_frog' + str(step) + '.png'))

        # 保存一张真实图像，用于对比
        img = image.array_to_img(real_images[0] * 255., scale=False)
        img.save(os.path.join(save_dir, 'real_frog' + str(step) + '.png'))