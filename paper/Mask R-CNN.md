# Mask R-CNN


## 关与mask分支中的二进制掩码
Mask R-CNN 通过向 Faster R-CNN 添加一个分支来进行像素级分割，该分支输出一个二进制掩码，该掩码表示给定像素是否为目标对象的一部分：该分支是基于卷积神经网络特征映射的全卷积网络。将给定的卷积神经网络特征映射作为输入，输出为一个矩阵，其中像素属于该对象的所有位置用 1 表示，其他位置则用 0 表示，这就是二进制掩码。
## 关于两类分类损失
 ![rongqi](https://github.com/wls860707495/Deep-Learning/blob/master/img/two_class_loss.png)
## 注
双线性插值：https://blog.csdn.net/qq_37577735/article/details/80041586   
源代码详解：https://www.cnblogs.com/YouXiangLiThon/p/9178861.html  
论文详解：https://www.cnblogs.com/kk17/p/9991446.html  
