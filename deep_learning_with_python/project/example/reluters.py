import numpy as np
from keras.datasets import reuters
from keras import models
from keras import layers
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import copy

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(path="D:\\relutersData\\reuters.npz", num_words=10000)

#将整数序列编码为二进制矩阵
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))   #创建一个此形状的0矩阵
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)


def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results

#内置one_hot向量转变
one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)

#模型构建每层64个内置单元，防止参数丢失训练不到
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

#留出验证集
x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

#训练模型，且分为20个伦茨，每个轮次批量512个数据
history = model.fit(partial_x_train, partial_y_train,epochs=20,batch_size=512, validation_data=(x_val, y_val))

#画出损失曲线
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(loss)+1)


plt.plot(epochs,loss,'bo',label ='Training loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.title('Tranining and validation loss')
plt.xlabel('Epochs')
plt.legend

plt.show()

#准确度曲线
plt.clf()

acc = history.history['acc']
val_acc = history.history['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

#九轮后开始过拟合
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(partial_x_train, partial_y_train, epochs=9, batch_size=512, validation_data=(x_val, y_val))
results = model.evaluate(x_test, one_hot_test_labels)

predictions = model.predict(x_test)
print(results)
# print(len(train_data))
# print(len(test_data))