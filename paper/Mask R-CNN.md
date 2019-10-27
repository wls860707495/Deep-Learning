# Mask R-CNN
## 关与mask分支中的二进制掩码
Mask R-CNN 通过向 Faster R-CNN 添加一个分支来进行像素级分割，该分支输出一个二进制掩码，该掩码表示给定像素是否为目标对象的一部分：该分支是基于卷积神经网络特征映射的全卷积网络。将给定的卷积神经网络特征映射作为输入，输出为一个矩阵，其中像素属于该对象的所有位置用 1 表示，其他位置则用 0 表示，这就是二进制掩码。  
## 关于两类分类损失
 ![rongqi](https://github.com/wls860707495/Deep-Learning/blob/master/img/two_class_loss.png)
## 架构
主干网络使用的是FPN与ResNet融合的网络，在主干网络提取出特征图之后，RPN网络也要进行相应的修改，此时RPN网络接收到的是5个特征图，在这些特征图中我们根据特征图的大小在原图上采取不同规模的anchor box（比如，5x5大小的特征图我们在原图上使用规模为128x128的anchor box，12x12大小的特征图我们在原图上使用256x256大小的特征图），之后的目标分类与框回归和ppt上讲述相同。
## 关于Training
我们对这个网络进行训练的时候，我们首先训练的RPN的网络，第二次训练的是Faster R-CNN的回归分类网络（其输入是我们经过RPN得到的RoI），在这之后我们使用经过训练过后的Faster R-CNN的参数来调节RPN网络（因为两者共享相同的卷积层），将网络进行第二轮的训练。当所有轮次的训练结束后，我们这两部分将共用一个卷积层。这个过程与Faster R-CNN中的过程类似。
## 注
## PPT中有核心的讲述，这里只记录一些需要注意的点
双线性插值：https://blog.csdn.net/qq_37577735/article/details/80041586   
源代码详解：https://www.cnblogs.com/YouXiangLiThon/p/9178861.html  
论文详解：https://www.cnblogs.com/kk17/p/9991446.html  
