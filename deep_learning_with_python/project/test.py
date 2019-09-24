from keras import  layers
from keras import  models

# 创建卷积模型
model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))  ## 此处(3,3)为kenel_size
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# 在模型中添加分类器
model.add(layers.Flatten())      # 压平三维向量变为一维
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))