# 共享权重的LSTM层，例子为预测两个句子的相似度以及相似摄像头图片融合
from keras import layers, applications
from keras import  Input
from keras.models import Model
left_data = []
right_data = []
targets = []
lstm = layers.LSTM(32)

# 构建模型的左分支，输入是长度128向量组成的变长序列
left_input = Input(shape=(None,128))
left_output = lstm(left_input)

#构建模型的右分支，如果调用已有的层实例，那么就会重复使用它的权重
right_input = Input(shape=(None, 128))
right_output = lstm(right_input)

# 连接两个模型并构建一个分类器
merged = layers.concatenate([left_output,right_output],axis=-1)
predictions = layers.Dense(1,activation='sigmod')(merged)

model = Model([left_input,right_input],predictions)
model.fit([left_data,right_data],targets)

##将层作为模型，共享层双摄像头（距离很近）共享
#图像处理基础模型是Xception 网络（只包 括卷积基)
xception_base = applications.Xception(weights = None,include_top = False)

left_input = Input(shape=(250, 250, 3))
right_input = Input(shape=(250, 250, 3))

left_features = xception_base(left_input)
right_input = xception_base(right_input)

#合并后的特征包含来自左右 两个视觉输入中的信息
merged_features = layers.concatenate([left_features, right_input], axis=-1)
