# from keras.datasets import imdb
# from keras.layers import preprocessing
# from keras.models import Sequential
# from keras.layers import Flatten, Dense, Embedding

# max_features = 10000
# maxlen = 20
# (x_train, y_train), (x_test, y_test) = imdb.load_data(path="D:\IMDBdata\imdb.npz", num_words=max_features)
#
# # 将整数列表转换成形状为(samples, maxlen) 的二维整数张量
# x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
# x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
#
# model = Sequential()
#
# #指定 Embedding 层的最大输入长度，以 便后面将嵌入输入展
# # 平。Embedding 层 激活的形状为 (samples, maxlen, 8)
# model.add(Embedding(10000, 8, input_length=maxlen))
#
# model.add(Flatten())
#
# model.add(Dense(1, activation='sigmoid'))
# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
# model.summary()
#
# history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

###################################################

import  os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
import numpy as np
import matplotlib.pyplot as plt

imdb_dir = 'D:\BaiduNetdiskDownload\\aclImdb\\aclImdb'
train_dir = os.path.join(imdb_dir,'train')

labels = []
texts = []

for label_type in ['neg','pos']:
    dir_name = os.path.join(train_dir,label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name,fname),encoding='utf-8')
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)

maxlen = 100
training_samples = 200
validation_samples = 10000
max_words = 10000

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=maxlen)   #data中包含所有文本序列

labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# 将数据划分为训练集验证集，但首先要打乱数据，因为一开始数据中的样本是排序好的
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]        #为每个data中的数据上分类标签

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]


#将Glove词嵌入
glove_dir = 'D:\data\glove.6B'

embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'),encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))
embedding_dim = 100

embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector         # 嵌入缩影中找不到的词，其嵌入向量全为0

print("**************")
# 构建模型层
model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

#从模型中加载GloVe嵌入
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False                       #冻结

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
history = model.fit(x_train,y_train,epochs=10,batch_size=32,validation_data=(x_val, y_val))
model.save_weights('pre_trained_glove_model.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(acc)+1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
