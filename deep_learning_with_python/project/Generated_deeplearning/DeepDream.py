# 使用keras实现DeepDream Inception.V3

from keras.applications import inception_v3
from keras import backend as K
import numpy as np
import scipy
from keras.preprocessing import image

# 禁用所有有关与训练的
from tensorflow.contrib.gan.python.eval import preprocess_image

K.set_learning_phase(0)

# 构建不包括全连接层的Inception.V3网络。使用预训练的ImageNet权重来加载模型
model = inception_v3.InceptionV3(weights = 'imagenet',include_top = False)
model.summary()

# 这个字典将层的名称映射为一个系数，这个系数定量 表示该层激活对你要最大化的损失的贡献大小。注
# 意，层的名称硬编码在内置的Inception V3应用中。可以 使用 model.summary() 列出所有层的名称
layer_contributions = {
    'mixed8':0.2,
    'mixed6':3.,
    'mixed7':2.,
    'mixed9':1.5,
}

# 定义需要最大化的损失
layer_dict = dict([(layer.name,layer) for layer in model.layers])  #创建一个字典，将层的名称映射为层的实例

loss = K.variable(0.)    # 在定义损失时将层的贡献添加到这个标量变量中
for layer_name in layer_contributions:
    coeff = layer_contributions[layer_name]
    activation = layer_dict[layer_name].output    # 获取层的输出

    scaling = K.prod(K.cast(K.shape(activation), 'float32'))
    loss += coeff * K.sum(K.square(activation[:, 2: -2, 2: -2, :])) / scaling     # 将该层特征的L2范数添加到loss中。为了避免出现边界伪影，损失中仅包含非边界的像素

# 梯度上升过程
dream = model.input      # 这个张量用于保存生成的图像，即梦境图像
grads = K.gradients(loss,dream)[0]        # 计算损失相对于梦境图像的梯度
grads /= K.maximum(K.mean(K.abs(grads)), 1e-7)       #将梯度标准化

# 给定一张输出图像，设置一个Keras函数来获取损失值和梯度值
outputs = [loss, grads]
fetch_loss_and_grads = K.function([dream], outputs)

def eval_loss_and_grads(x):
    outs = fetch_loss_and_grads([x])
    loss_value = outs[0]
    grad_values = outs[1]
    return loss_value, grad_values

# 这个函数运行iteration次梯度上升
def gradient_ascent(x, iterations, step, max_loss=None):
    for i in range(iterations):
        loss_value, grad_values = eval_loss_and_grads(x)
        if max_loss is not None and loss_value > max_loss:
            break
        print('...Loss value at', i, ':', loss_value)
        x += step * grad_values
        return x


def resize_img(img, size):
    img = np.copy(img)
    factors = (1, float(size[0]) / img.shape[1], float(size[1]) / img.shape[2], 1)
    return scipy.ndimage.zoom(img, factors, order=1)


def save_img(img, fname):
    pil_img = deprocess_image(np.copy(img))
    scipy.misc.imsave(fname, pil_img)

# 通用函数，，用于打开图像、改变图像大小以及将图像格式转换为 Inception V3 模型能够处理的张量
def preprocess_image(image_path):
    img = image.load_img(image_path)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = inception_v3.preprocess_input(img)
    return img

# 通用函数，将一个张量转换为有效图像
def deprocess_image(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, x.shape[2], x.shape[3]))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((x.shape[1], x.shape[2], 3))         # 对 inception_v3.preprocess_input 所做的预处理进行反向操作
    x /= 2.
    x += 0.5
    x *= 255.
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# 在多个连续尺度上运行梯度上升
step = 0.01             # 梯度上升的步长
num_octave = 3              # 运行梯度上升的尺度个数
octave_scale = 1.4              # 两个尺度之间的大小比例
iterations = 20             # 在每个尺度上运行梯度上升的步数

max_loss = 10             # 如果损失增大到大于10，我们要中断梯度上升过程，以避免得到丑陋的伪影
base_image_path = 'D:\cat_deepdream.jpg'
img = preprocess_image(base_image_path)

original_shape = img.shape[1:3]
successive_shapes = [original_shape]

# 准备一个由形状元组组成的列表，它定义了运行梯度上升的不同尺度
for i in range(1,num_octave):
    shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
    successive_shapes.append(shape)
    successive_shapes = successive_shapes[::-1]       # 将形状列表反转，变为升序


    # 将图像Numpy数组的大小缩放到最小尺寸
    original_img = np.copy(img)
    shrunk_original_img = resize_img(img, successive_shapes[0])


    for shape in successive_shapes:
        print('Processing image shape', shape)
        img = resize_img(img, shape)  # 将梦境图像放大

        # 运行梯度上升，改变梦境图像
        img = gradient_ascent(img, iterations=iterations, step=step, max_loss=max_loss)

        # 将原始图像的较小版本放大，它会变得像素化
        upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)

        # 在这个尺寸上计算原始图像的高质量版本
        same_size_original = resize_img(original_img, shape)

        #二者的差别就是在放大过程中丢失的细节
        lost_detail = same_size_original - upscaled_shrunk_original_img

        #将丢失的细节重新注入到梦境图片中
        img += lost_detail
        shrunk_original_img = resize_img(original_img, shape)
        save_img(img, fname='dream_at_scale_' + str(shape) + '.png')

    save_img(img, fname='final_dream2.png')

