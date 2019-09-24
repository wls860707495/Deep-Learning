# import numpy as np
# import pandas as pd
#
# data = pd.read_csv(r"D:\Bostendata\housing.csv")
# np.savez(r"D:\Bostendata\housing", data)

from keras.datasets import boston_housing
from keras import  models
from keras import  layers
import numpy as np
import matplotlib.pyplot as plt

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data(path="D:\Bostendata\housing.npz")

#对数据进行标准化，减去平均值之后除以标准差
mean = train_data.mean(axis=0)  #axis表示沿标签列进行索引方法
train_data -= mean

std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])     #mse均值平方损失方法，mae（平均绝对误差）回归的评价指标
    return model

# 于数据点很少，验证集会非常小（比如大约 100 个样本）。
# 因此，验证分数可能会有很大波动，这取决于你所选择的验证集和训练集。
# 也就 是说，验证集的划分方式可能会造成验证分数上有很大的方差，这样就
# 无法对模型进行可靠的 评估。在这种情况下，最佳做法是使用 K 折交叉验证

###  K折验证
#这种方法将可用数据划分为 K 个分区（K 通常取 4 或 5），实例化K 个相同的模型，将每个 模型在
# K-1 个分区上训练，并在剩 下的一个分区上进行评估。模型的验证分数等于K 个验证分数的平均值。
k = 4
num_val_samples = len(train_data) // k
num_epochs = 500
all_mae_histories = []
for i in range(k):
    print('processing fold #', i)

#准备验证数据：第k个分区的数据
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

###准备训练数据：其他所有分区的数据
    partial_train_data = np.concatenate(
         [train_data[:i * num_val_samples],
          train_data[(i + 1) * num_val_samples:]],
         axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
        train_targets[(i + 1) * num_val_samples:]],
         axis=0)

    model = build_model()           #构建Keras模型
    # history = model.fit(partial_train_data, partial_train_targets,    #批次为100时的训练代码
    #          epochs=num_epochs, batch_size=1, verbose=0)
    # val_mse,  val_mae = model.evaluate(val_data, val_targets, verbose=0)   # -->评估验证集上的模型
    history = model.fit(partial_train_data, partial_train_targets,     #批次为500时的训练代码
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=1, verbose=0)
    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)
# print(all_scores)
# print(np.mean(all_scores))

print(all_mae_histories)

#计算每个轮次中的K折验证分数平均值
average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

#删除前十个取值范围不同的点
def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

smooth_mae_history = smooth_curve(average_mae_history[10:])

plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
# plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()
