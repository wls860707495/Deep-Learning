# 注
零基础入门深度学习：https://www.zybuluo.com/hanbingtao/note/433855  
RCNN:
```
Rich feature hierarchies for accurate object detection and semantic segmentation Tech report(v5)   
author:Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik   
```
SPP-Net:--->解决RCNN中，每个备选区域都进行特征提取，花费大量时间、空间且多有重叠部分。SPP-Net则是一个网络共享整张图片的。是一个过度方法。
```
Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition
author:Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
```
其训练依旧分为多个阶段，步骤繁琐：微调网络+训练SVM+训练回归器  
Fast RCNN:--->解决SPP-Net缺点，开始使用softmax及ROI Pooling
```
Fast R-CNN
author： Ross Girshick， Microsoft Research
```
Region proposals是通过SS提取的，在一定程度上限制了速度
Faster R-CNN--->使用RPN来提取候选区域
```
Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks
Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun 
```
