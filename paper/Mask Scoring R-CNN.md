# Mask Scoring R-CNN
## 核心要点
实际增加了对Mask 分支的打分机制，将Mask IoU进行回归，同时使用上对于分类分支已经分好的类进行损失。？MaskIOU与Mask R-CNN中的损失函数有何不同？？
之前的Mask R-CNN中mask分支使用的是逐像素的二分类损失，在Mask Scoring R-CNN中使用的是真实Mask与预测Mask之间的IoU，对此进行回归损失。  
## Mask IoU
Mask IoU实质上是将RoI Align得到的特征图与原Mask Head中得到的Mask输出进行了“+”操作。在论文中详细比较了各种连接操作，其中只有“+”操作具有最好的效果。在Mask IoU的输出当中，我们的Smask = Siou * Scls，其中Scls在RCNN Head中已经得到，而Siou是我们需要进行计算的。在论文中使用了4个卷积层（所有的核大小以及滤波器个数分别设置为3和256）以及3个全连接层（遵循RCNN Head的设置），前两个FC为1024，最后一个FC输出个数为类别的数量。
## 损失函数
Mask Head 和RCNN Head 的损失与Mask RCNN中的这两部分的损失相同，而在Mask IoU Head中，其损失为对Scoring进行l2回归损失。
