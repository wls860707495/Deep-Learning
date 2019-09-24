# 两个例子，分别是词、字符转换为one-hot编码，最后为使用Keras内置函数实现单词级one-hot编码
## one
# import  numpy as np
#
# samples = ['The cat sat on the mat.','The dog ate my homework.']
#
# token_index = {}
# for sample in samples:
#     for word in sample.split():
#         if word not in token_index:
#             token_index[word] = len(token_index) + 1      ## 为每个单词建立一个唯一索引
#
# max_length = 10               ## 对样本进行分词，只考虑样本前max_length个单词
#
# results = np.zeros(shape =(len(samples),max_length,max(token_index.values()) + 1))  # 创建一个大小为0的张量
# print(results)
# # 向创建的张量中赋值1
# for i,sample in enumerate(samples):
#     for j,word in list(enumerate(sample.split()))[:max_length]:
#         index = token_index.get(word)
#         results[i,j,index] = 1


## two
# import string
# import numpy as np
# samples = ['The cat sat on the mat.','The dog ate my homework.']
# charactrers = string.printable
# token_index = dict(zip(range(1,len(charactrers)+1),charactrers))
#
# max_length = 50
# results = np.zeros((len(samples),max_length,max(token_index.keys()) + 1))
# for i,sample in enumerate(samples):
#     for j,charactrer in enumerate(sample):
#         index = token_index.get(charactrer)
#         results[i,j,index] = 1
#
# print(results)

## three
from keras.preprocessing.text import Tokenizer

samples = ['The cat sat on the mat.','The dog ate my homework.']

tokenizer = Tokenizer(num_words = 1000)      # 创建一个分词器只考虑前1000个单词
tokenizer.fit_on_texts(samples)     #构建单词索引

sequences = tokenizer.texts_to_sequences(samples)     #将字符串转换为整数索引组成的列表

# 也可以直接得到 one-hot 二进制表示。 这个分词器也支持除one-hot 编码外 的其他向量化模式
one_hot_results = tokenizer.texts_to_matrix(samples,mode='binary')

word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))