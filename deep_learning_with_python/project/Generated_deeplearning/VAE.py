# 变分自编码器
## VAE大致代码
# z_mean,z_log_variance = encoder(input_img)      # 将输入编码为平均值和方差两个参数
# z = z_mean + exp(z_log_variance) * epsilon        # 使用小随机数epsilon来抽取一个潜在点
# reconstructed_img = decoder(z)             # 将z解码为一场图像
# model = Model(input_img,reconstructed_img)        # 将自编码器模型实例化，它将一张输入图像映射为它的重构
import keras
from keras import layers
from keras import backend as K
from keras.models import Model
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
from scipy.stats import norm

img_shape = (28, 28, 1)
batch_size = 16
latent_dim = 2          #潜在空间的维度：一个二维平面

input_img = keras.Input(shape=img_shape)

x = layers.Conv2D(32, 3, padding='same', activation='relu')(input_img)
x = layers.Conv2D(64, 3, padding='same', activation='relu', strides=(2, 2))(x)
x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
shape_before_flattening = K.int_shape(x)

x = layers.Flatten()(x)
x = layers.Dense(32, activation='relu')(x)

# 输入图像最终被编码为这两个参数
z_mean = layers.Dense(latent_dim)(x)
z_log_var = layers.Dense(latent_dim)(x)

# 潜在空间采样的函数
def sampling(args):
    z_mean,z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0],latent_dim),mean=0.,stddev=1.)
    return z_mean + K.exp(z_log_var) * epsilon

z = layers.Lambda(sampling)([z_mean,z_log_var])

# VAE解码器网络，将潜在空间点映射为图像
decoder_input = layers.Input(K.int_shape(z)[1:])          # 需要将z输入到这里

# 对输入进行上采样
x = layers.Dense(np.prod(shape_before_flattening[1:]), activation='relu')(decoder_input)

# 将z转换为特征图，使其形状与编码器模型最后一个Flatten层之前的特征图的形状相同
x = layers.Reshape(shape_before_flattening[1:])(x)

# 使用一个Conv2DTranspose层和一个Conv2D层，将z解码为与原始输入图像具有相同尺寸的特征图
x = layers.Conv2DTranspose(32, 3, padding='same', activation='relu', strides=(2, 2))(x)
x = layers.Conv2D(1, 3, padding='same', activation='sigmoid')(x)

# 将解码器模型实例化，它将decoder_input转换为解码后的图像
decoder = Model(decoder_input, x)

# 将这个实例应用与z，以得到解码后的z
z_decoded = decoder(z)

# 用于计算VAE损失的自定义层
class CustomVariationalLayer(keras.layers.Layer):
    def vae_loss(self, x, z_decoded):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)
        xent_loss = keras.metrics.binary_crossentropy(x, z_decoded)
        kl_loss = -5e-4 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    # 编写一个call方法来实现自定义层
    def call(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        loss = self.vae_loss(x, z_decoded)
        self.add_loss(loss, inputs=inputs)
        return x        # 我们不使用这个输出，但层必须要有返回值

# 对输入和编码后的输出调用自定义层，以得到最终的模型输出
y = CustomVariationalLayer()([input_img, z_decoded])

# 训练VAE
vae = Model(input_img, y)
vae.compile(optimizer='rmsprop', loss=None)
vae.summary()

(x_train, _), (x_test, y_test) = mnist.load_data(path="D:\minist\mnist.npz")

x_train = x_train.astype('float32') / 255.
x_train = x_train.reshape(x_train.shape + (1,))
x_test = x_test.astype('float32') / 255.
x_test = x_test.reshape(x_test.shape + (1,))

vae.fit(x=x_train, y=None, shuffle=True, epochs=10, batch_size=batch_size, validation_data=(x_test, None))

# 从二维潜在空间中采样一组点的网格，并将其解码为图像
n = 15

#使用SciPy的ppf函数对线性分隔的坐标进行变换，以生成潜在变量z的值（因为潜在空间的先验分布是高斯分布）
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])

        # 将z多次重复，以构建一个完整的批量
        z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)
        x_decoded = decoder.predict(z_sample, batch_size=batch_size)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size, j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()