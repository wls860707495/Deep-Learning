#### 留出验证
# 留出一定比例的数据作为测试集。在剩余的数据上训练模型，然后在测试集上评估模型。
# 如前所述，为了防止信息泄露，你不能基于测试集来调节模型，所以还应该保留一个验证集。
# 缺点：不适用于数据集数量少的情况

import numpy as np
# num_validation_samples = 10000
#
# np.random.shuffle(data)
#
# validation_data = data[:num_validation_samples]
# data = data[num_validation_samples:]
#
# training_data = data[:]
#
# # 在训练集上训练模型，并在验证集上评估模型
# model = get_model()
# model.train(training_data)
# validation_score = model.evaluate(validation_data)
#
# # 现在你可以调节模型、重新训练、评估，然后再次调节……
# model = get_model()
# model.train(np.concatenate([training_data, validation_data]))
# test_score = model.evaluate(test_data)


#### K折交叉验证
# 将数据划分为大小相同的K 个分区。对于每个分区 i，在剩 余的K-1 个分区上训练模型，
# 然后在分区 i 上评估模型。最终分数等于K 个分数的平均值。对 于不同的训练集- 测
# 试集划分，如果模型性能的变化很大，那么这种方法很有用

# k = 4
# num_validation_samples = len(data) // k
#
# np.random.shuffle(data)
#
# validation_scores = []
# for fold in range(k):
#     #选择验证集数据分区
#     validation_data = data[ num_validation_samples * fold:num_validation_samples * (fold + 1)]
#     training_data = data[:num_validation_samples * fold] + \
#                     data[num_validation_samples * (fold + 1):]   #使用剩余数据作为训练数据，其中“+”为列表合并，并非求和
#
#     #创建一个全新的模型
#     model = get_model()
#     model.train(training_data)
#     validation_score = model.evaluate(validation_data)
#     validation_scores.append(validation_score)
#
# #最终验证分数：K 折验证 分数的平均值
# validation_score = np.average(validation_scores)
#
# #在所有非测试数据 上训练最终模型
# model = get_model()
# model.train(data)
# test_score = model.evaluate(test_data)


#### 带有打乱数据的重复K折验证
# 多次使用 K 折验证，在每次将数据划分为 K 个分区之前都先将数据打乱。 最终分数是每次K 折验
# 证分数的平均值。注意，这种方法一共要训练和评估P×K 个模型（P 是重复次数），计算代价很大。


# 防止神经网络过拟合的常用方法包括： 
#   1、获取更多的训练数据 
#   2、减小网络容量 
#   3、添加权重正则化 
#   4、添加 dropout