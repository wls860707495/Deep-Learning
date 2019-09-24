# 对猫狗的数据集进行增强（示例）
import os
import matplotlib.pyplot as plt
from keras_preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

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
