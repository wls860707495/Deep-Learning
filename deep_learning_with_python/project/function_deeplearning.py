# 对数据进行多模态输入或者多任务输出，函数式API模型（不仅仅只是线性堆叠）
## 使用函数式API，你可以直接操作张量，也可以把层当作函数来使用，接收张量并返回张量
from  keras import  Input,layers
from  keras.models import Sequential,Model
#
# input_tensor = Input(shape=(32,))  # 一个张量
# dense = layers.Dense(32,activation='relu')    # 将一个层看作一个函数
# output_tensor = dense(input_tensor)     # 可以在一个张量上调用一个层， 它会返回一个张量

## 线性堆叠的模型
seq_model = Sequential()
seq_model.add(layers.Dense(32,activation='relu',input_shape=(64,)))
seq_model.add(layers.Dense(32,activation='relu'))
seq_model.add(layers.Dense(10,activation='softmax'))

## 对应的函数式API实现
input_tensor = Input(shape=(64,))
x = layers.Dense(32,activation='relu')(input_tensor)
x = layers.Dense(32,activation='relu')(x)
output_tensor = layers.Dense(10,activation='softmax')(x)

model = Model(input_tensor,output_tensor)
model.summary()