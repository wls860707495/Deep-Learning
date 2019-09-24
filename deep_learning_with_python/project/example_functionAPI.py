# 一、使用函数式API模型完成问答问题的训练 --> 两个输入：一个自然语言描述的问题和一个文本片段（比如新闻文章），后者提供用于回答问题的信息
import keras
from keras.models import Model
from keras import layers
from keras import Input
import numpy as np

text_vocabulary_size = 10000
question_vocabulary_size = 10000
answer_vocabulary_size = 500

text_input = Input(shape=(None,), dtype='int32', name='text')  # 文本输入是一个长度可变的整数序列。注意，你可以选择对输入进行命名
embedded_text = layers.Embedding(text_vocabulary_size, 64)(text_input)   #将输入嵌入长度为 64 的向量
encoded_text = layers.LSTM(32)(embedded_text)  #利用 LSTM 将向量编码为单个向量

question_input = Input(shape=(None,),
                       dtype='int32',
                       name='question')
embedded_question = layers.Embedding(question_vocabulary_size, 32)(question_input)
encoded_question = layers.LSTM(16)(embedded_question)

# 将编码后的问题和文本连接起来
concatenated = layers.concatenate([encoded_text, encoded_question], axis=-1)
answer = layers.Dense(answer_vocabulary_size, activation='softmax')(concatenated)

model = Model([text_input,question_input],answer)
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics = ['acc'])

num_samples = 1000
max_length = 100

text = np.random.randint(1, text_vocabulary_size, size=(num_samples, max_length))    #生成虚构的Numpy数据
question = np.random.randint(1, question_vocabulary_size, size=(num_samples, max_length))
answers = np.random.randint(answer_vocabulary_size, size=(num_samples))
answers = keras.utils.to_categorical(answers, answer_vocabulary_size)

## 第一种拟合方法
model.fit([text, question], answers, epochs=10, batch_size=128)    # 使用输入组成的 列表来拟合
## 第二种拟合方法
model.fit({'text': text, 'question': question}, answers, epochs=10, batch_size=128)  # 使用输入组成的字典来拟合 （只有对输入进行命名之后才 能用这种方法）


# 二、输入某个匿名人士的一系列社交媒体发帖，然后尝试预测那个人的属性，比如年龄、性别和收入水平
vocabulary_size = 50000
num_income_groups = 10

post_input = Input(shape=(None,),dtype='int32',name='posts')
embedded_posts = layers.Embedding(256,vocabulary_size)(post_input)
x = layers.Conv1D(128,5,activation='relu')(embedded_posts)
x = layers.MaxPool1D(5)(x)
x = layers.Conv1D(256,5,activation='relu')(x)
x = layers.Conv1D(256,5,activation='relu')(x)
x = layers.MaxPool1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dense(128, activation='relu')(x)

age_prediction = layers.Dense(1,name='age')(x)   # --> 输出层都具有名称

income_prediction = layers.Dense(num_income_groups,activation='softmax',name='income')(x)

gender_prediction = layers.Dense(1, activation='sigmoid', name='gender')(x)

model = Model(post_input,[age_prediction,income_prediction,gender_prediction])

# 可以在编译时使用损失组成的列表或 字典来为不同输出指定不同损失，然后
# 将得到的损失值相加得到一个全局损失，并在训练过程 中将这个损失最小化。
# ## 多重损失
# ## 第一种赋予损失方法
# model.compile(optimizer='rmsprop',loss = ['mse','categorical_crossentropy','binary_crossentropy'])
# ## 第二种赋予损失方法，需要各个输出层有名字，与上述等效
# model.compile(optimizer='rmsprop', loss={'age': 'mse','income': 'categorical_crossentropy','gender': 'binary_crossentropy'})

## 多输出模型的损失加权
## 第一种损失加权
model.compile(optimizer='rmsprop',loss=['mse','categorical_crossentropy','binary_crossentropy'],loss_weights=[0.25, 1., 10.])
## 第二种损失加权
model.compile(optimizer='rmsprop',loss={'age': 'mse','income': 'categorical_crossentropy','gender': 'binary_crossentropy'},loss_weights={'age': 0.25,'income': 1.,'gender': 10.})

## 将数据输入到多输出模型中
# model.fit(post_input, [age_targets, income_targets, gender_targets],epochs=10, batch_size=64)
# model.fit(posts, {'age': age_targets,'income': income_targets,'gender': gender_targets},epochs=10, batch_size=64)