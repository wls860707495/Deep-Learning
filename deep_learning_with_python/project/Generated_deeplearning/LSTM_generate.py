# 使用LSTM来生成序列文本（字符级的神经语言模型）
import keras
import numpy as np
from keras import layers
import random
import sys

# original_distribution 是概率值组成的一维Numpy数组，这些概率值之
# 和必须等于1。temperature是一个因子，用于定量描述输出分布的熵
def reweight_distribution(original_distribution,temperature = 0.5):
    distribution = np.log(original_distribution)/temperature
    distribution = np.exp(distribution)
    return  distribution/np.sum(distribution) # --> 返回原始分布重新加权后的结果。distribution的求和可能不再等于1，因此需要将它除以求和，以得到新的分布

# 给定模型预测，采样下一个字符的函数
def sample(preds,temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)/temperature
    exp_preds = np.exp(preds)
    preds = exp_preds/np.sum(exp_preds)
    probas = np.random.multinomial(1,preds,1)
    return np.argmax(probas)

path = 'D:\LSTM_data\\nietzsche.txt'

text = open(path).read().lower()
print('Cirpus length',len(text))

# 提取长度为maxlen的序列并转换为one-hot向量
maxlen = 60
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])

print(sentences)
print(next_chars)
print('Number of sequences:', len(sentences))

chars = sorted(list(set(text)))            #语料中唯一字符组成的列表
print('Unique characters:', len(chars))
char_indices = dict((char, chars.index(char)) for char in chars)    # 一个字典，将唯一字符映射为它在列表chars中的索引

# 将one-hot编码转换为二进制数组
print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)   #其中包含对应的目标，即在每一个所提 取的序列之后出现的字符（已进行 one-hot 编码）。
for i,sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

#用于预测下一个字符的单层LSTM模型
model = keras.models.Sequential()
model.add(layers.LSTM(128,input_shape=(maxlen,len(chars))))
model.add(layers.Dense(len(chars),activation='softmax'))

optimizer = keras.optimizers.RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

for epoch in range(1, 60):                   # <-- 将模型训练60轮
    print('epoch', epoch)
    model.fit(x, y, batch_size=128, epochs=1)          # <-- 将模型在数据上拟合一次

    # 随机选择一个文本种子
    start_index = random.randint(0, len(text) - maxlen - 1)
    generated_text = text[start_index: start_index + maxlen]
    print('--- Generating with seed: "' + generated_text + '"')

    for temperature in [0.2, 0.5, 1.0, 1.2]:        # <-- 尝试一系列不同的采样温度
        print('------ temperature:', temperature)
        sys.stdout.write(generated_text)

        for i in range(400):                   # <-- 从种子文本开始，生成400个字符
            # 对目前的生成的字符进行one-hot编码
            sampled = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(generated_text):
                sampled[0, t, char_indices[char]] = 1.

            # 对下一个字符进行采样
            preds = model.predict(sampled, verbose=0)[0]
            next_index = sample(preds, temperature)
            next_char = chars[next_index]

            generated_text += next_char
            generated_text = generated_text[1:]

            sys.stdout.write(next_char)