# Mask Scoring R-CNN
## 核心要点
实际增加了对Mask 分支的打分机制，将Mask IoU进行回归，同时使用上对于分类分支已经分好的类进行损失。？MaskIOU与Mask R-CNN中的损失函数有何不同？？
之前的Mask R-CNN中mask分支使用的是逐像素的二分类损失，在Mask Scoring R-CNN中使用的是真实Mask与预测Mask之间的IoU，对此进行回归损失。
