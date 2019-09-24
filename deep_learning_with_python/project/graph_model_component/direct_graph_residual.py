# 残差连接
## 一、特征图尺寸相同时
from keras import layers
x = '四维输入张量'
y = layers.Conv2D(128,3,activation='relu',padding='same')(x)
y = layers.Conv2D(128,3,activation='relu',padding='same')(y)
y = layers.Conv2D(128,3,activation='relu',padding='same')(y)

y = layers.add([y,x])    #将原始x与输出特征相加
## 二、特征图尺寸不同时
y = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
y = layers.Conv2D(128, 3, activation='relu', padding='same')(y)
y = layers.MaxPooling2D(2, strides=2)(y)

# 使用1x1卷积，将原始x张量先行下采样为与y具有相同的形状
residual = layers.Conv2D(128, 1, strides=2, padding='same')(x)
y = layers.add([y,residual])