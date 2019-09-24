from keras.datasets import imdb
import numpy as np
import matplotlib.pyplot as plt

from keras import models, regularizers
from keras import layers
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(path="D:\IMDBdata\imdb.npz", num_words=10000)


def vectorize_sequences(sequences, dimension=10000):    #将整数序列编码为二进制矩阵
    results = np.zeros((len(sequences), dimension))

    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

#将训练数据向量化
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

#将标签向量化
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

#模型层
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

#优化器及损失与指标
model.compile(optimizer='rmsprop', loss='binary_crossentropy',metrics=['accuracy'])

#留出验证集
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

#将所有的数据组成512个小批量，将模型训练20个伦茨
history = model.fit(partial_x_train,partial_y_train,epochs=20,batch_size=512,validation_data=(x_val, y_val))

#绘制训练损失和验证损失
history_dict = history.history

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')               #bo表示蓝色圆点
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')                #b表示蓝色实线
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

#清空图像并画出训练集和验证集的精确度
plt.clf()
acc = history_dict['acc']
val_acc = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

print(x_train[0])
print(y_train)
print( max([max(sequence) for sequence in train_data]) )
print(history_dict.keys())

# l2(0.001) 的意思是该层权重矩阵的每个系数都会使网络总损失增加 0.001 * weight_ coefficient_value
model = models.Sequential()
model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),activation='relu', input_shape=(10000,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)
print(results)

m = model.predict(x_test)
print(m)

