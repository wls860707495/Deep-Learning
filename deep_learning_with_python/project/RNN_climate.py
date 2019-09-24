import os
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

data_dir = 'D:\jena_climate'
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')

f = open(fname)
data = f.read()
f.close()

lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]

float_data = np.zeros((len(lines), len(header) - 1))

for i,line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i, :] = values

## 绘制的温度变化图
# temp = float_data[:, 1]  # 温度（单位：摄氏度）
# plt.plot(range(len(temp)), temp)
# plt.figure()
# plt.plot(range(1440), temp[:1440])
# plt.show()

mean = float_data[:200000].mean(axis=0)
float_data -= mean
std = float_data[:200000].std(axis=0)
float_data /= std

#生成时间序列样本及其目标的生成器
## data：浮点数数据组成的原始数组，在代码清单 6-32 中将其标准化。 
## lookback：输入数据应该包括过去多少个时间步。 
## delay：目标应该在未来多少个时间步之后。 
## min_index 和 max_index：data 数组中的索引，用于界定需要抽取哪些时间步。这有 助于保存一部分数据用于验证、另一部分用于测试。 
## shuffle：是打乱样本，还是按顺序抽取样本。 
## batch_size：每个批量的样本数。 
## step：数据采样的周期（单位：时间步）。我们将其设为 6，为的是每小时抽取一个数据点。
## 其中 samples 是输入数据的一个批量，targets 是对应的目标温度数组。
def generator(data,lookback,delay,min_index,max_index,shuffle=False,batch_size  = 128,step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:      #函数是否具有shuffle属性
            rows = np.random.randint(min_index + lookback,max_index,size= batch_size)   # 一部分用于验证，另一部分用于测试
        else:
            if i + batch_size >= max_index:
                i = min_index +lookback
            rows = np.arange(i,min(i+batch_size,max_index))
            i += len(rows)
        samples = np.zeros((len(rows),lookback//step,data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j,row in enumerate(rows):
            indices = range(rows[j] - lookback,rows[j],step)
            samples[j] = data[indices]
            targets[j] = data[rows[j]+delay][1]
        yield samples,targets              # yield作用将普通函数变为迭代器
        # yield samples[:, ::-1, :], targets     #逆序时间步长
# 做三个生成器，分别用于训练（200000）、验证（100000）、测试
lookback = 1440
step = 6
delay = 144
batch_size = 128

train_gen = generator(float_data,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=200000,
                      shuffle=True,
                      step=step,
                      batch_size=batch_size)
val_gen = generator(float_data,
                    lookback=lookback,
                    delay=delay,
                    min_index=200001,
                    max_index=300000,
                    step=step,
                    batch_size=batch_size)
test_gen = generator(float_data,
                     lookback=lookback,
                     delay=delay,
                     min_index=300001,
                     max_index=None,
                     step=step,
                     batch_size=batch_size)
val_steps = (300000 - 200001 - lookback) // batch_size   # 为了查看整个验证集，需要从val_gen中抽取多少次

test_steps = (len(float_data) - 300001 - lookback) // batch_size  # 为了查看整个测试集，需要从 test_gen 中抽取多少次

# # 一种基于常识的方法就是始终预测24小时后的温度等于现在的温度。我们使用平均绝对误差（MAE）指标来评估这种方法
# def evaluate_naive_method():
#     batch_maes = []
#     for step in range(val_steps):
#         samples, targets = next(val_gen)
#         preds = samples[:, -1, 1]
#         mae = np.mean(np.abs(preds - targets))
#         batch_maes.append(mae)
#     print(np.mean(batch_maes))
# evaluate_naive_method()
#
# # mae转换为摄氏温度差
# celsius_mae = 0.29 * std[1]
# print(celsius_mae)

## 训练评估一个密集连接模型
# model = Sequential()
# model.add(layers.Flatten(input_shape=(lookback // step, float_data.shape[-1])))
# model.add(layers.Dense(32, activation='relu'))
# model.add(layers.Dense(1))
#
# model.compile(optimizer=RMSprop(), loss='mae')
# history = model.fit_generator(train_gen, steps_per_epoch=500,
#                               epochs=20, validation_data=val_gen,
#                               validation_steps=val_steps)

## 训练评估一个基于GRU的模型
#。对每个时间步使用相同的 dropout 掩码，可以让网络 沿着时间正确地传播其学习误差，
# 而随时间随机变化的dropout 掩码则会破坏这个误差信号，并 且不利于学习过程。
model = Sequential()
model.add(layers.GRU(32,dropout=0.1,recurrent_dropout=0.5,return_sequences=True,input_shape=(None, float_data.shape[-1])))
model.add(layers.GRU(64, activation='relu',
                     dropout=0.1,
                     recurrent_dropout=0.5))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen, steps_per_epoch=500, epochs=40,
                              validation_data=val_gen,
                              validation_steps=val_steps)

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


