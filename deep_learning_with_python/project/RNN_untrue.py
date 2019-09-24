## RNN伪代码实现

# state_t = 0            #t时刻的状态
# for input_t in input_sequence:       #对序列元素进行遍历
#     output_t = activation(dot(W,input_t) + dot(U,state_t)+b)
#     state_t = output_t              #前一次的输出变成下一次迭代的状态

## 简单实现处理单个序列的循环层
# import numpy as np
# timesteps = 100              #输入序列的时间步长
# input_features = 32             #输入特征空间的维度
# output_features = 64            #输出特征空间的维度
#
# inputs = np.random.random((timesteps, input_features))
# state_t = np.zeros((output_features,))
#
# W = np.random.random((output_features, input_features))
# U = np.random.random((output_features, output_features))
# b = np.random.random((output_features,))
#
# successive_outputs = []
# for input_t in inputs:
#     output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)
#     successive_outputs.append(output_t)
#     state_t = output_t                         #更新网络的状态，用于下一个时间步长
#
# final_output_sequence = np.stack(successive_outputs, axis=0)       #最终输出是一个形状为 (timesteps, output_features) 的二维张量

from keras.layers import SimpleRNN
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN
model = Sequential()
model.add(Embedding(10000,32))
model.add(SimpleRNN(32,return_sequences=True))       #需要堆叠多个循环层时需要让所有的中间层都返回完整的序列
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32, return_sequences=True))
model.summary()

