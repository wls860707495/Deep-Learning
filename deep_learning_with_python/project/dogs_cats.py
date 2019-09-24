from keras import layers
from keras import models
from keras import optimizers
from keras_preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
from keras.preprocessing import image

# 对猫狗的数据集进行增强(示例) ---> 起始
original_dataset_dir = 'D:\dags-and-cats\kaggle_original_data'
base_dir = 'D:\dags-and-cats\cats_and_dogs_small'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')

validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

test_cats_dir = os.path.join(test_dir, 'cats')
test_dogs_dir = os.path.join(test_dir, 'dogs')

datagen = ImageDataGenerator(
    rotation_range=40,            # 角度值
    width_shift_range=0.2,             # 水平方向平移
    height_shift_range=0.2,              # 垂直方向平移
    shear_range=0.2,             # 随机错切变换的角度
    zoom_range=0.2,              # 图像随机缩放的范围
    horizontal_flip=True,              #随机将一半的图像水平翻转
    fill_mode='nearest')               # 用于填充新创建的像素

fnames = [os.path.join(train_cats_dir, fname) for
          fname in os.listdir(train_cats_dir)]
img_path = fnames[3]         # --> 选择一张图像进行增强
img = image.load_img(img_path, target_size=(150, 150))        # --> 读取图像并调整大小
x = image.img_to_array(img)            # --> 将图片形状转变为Numpy数组
x = x.reshape((1,) + x.shape)              # --> 将其形状进行改变
i = 0

#生成随机变换后的图像批量。 循环是无限的，因此你需要 在某个时刻终止循环
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break
plt.show()

# ---> 结束

model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4),metrics=['acc'])


# 此处为使用迭代器从目录中读取图像 --> 文件目录下的数目即为类的数目
# train_datagen = ImageDataGenerator(rescale=1./255)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True, )

test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    'D:\dags-and-cats\cats_and_dogs_small\\train',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    'D:\dags-and-cats\cats_and_dogs_small\\validation',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')

for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break

# 利用批量生成器拟合模型
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=50)

# 保存模型
model.save('cats_and_dogs_small_2.h5')

# 绘制损失与精度曲线
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)
## 精确度曲线
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

## 损失曲线
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
