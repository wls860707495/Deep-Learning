# 注
零基础入门深度学习：https://www.zybuluo.com/hanbingtao/note/433855  
## 两阶段方法
RCNN:
```
Rich feature hierarchies for accurate object detection and semantic segmentation Tech report(v5)   
author:Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik   
```
----------------------------------------------------------------------------------------------------  
SPP-Net:--->解决RCNN中，每个备选区域都进行特征提取，花费大量时间、空间且多有重叠部分。SPP-Net则是一个网络共享整张图片的。是一个过度方法。
```
Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition
author:Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
```
其训练依旧分为多个阶段，步骤繁琐：微调网络+训练SVM+训练回归器  

----------------------------------------------------------------------------------------------------  
Fast RCNN:--->解决SPP-Net缺点，开始使用softmax及ROI Pooling
```
Fast R-CNN
author： Ross Girshick， Microsoft Research
```
Region proposals是通过SS提取的，在一定程度上限制了速度

-----------------------------------------------------------------------------------------------------  
Faster R-CNN--->使用RPN来提取候选区域，相比fast-rcnn检测快很多
```
Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks
Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun 
```
RPN会直接通过conv产生的特征产生区域候选，不需要额外算法产生区域候选。在RPN之后会像Fast RCNN一样使用ROI Pooling和后续的分类器和回归器。  
RPN训练用来提取proposal，并利用训练RPN的CNN来训练分类器即参数共享。  

----------------------------------------------------------------------------------------------------
FPN：--->解决对于卷积层比较浅时，特征提取很少的问题。实质是一个学习特征的框架。        
```
Feature Pyramid Networks for Object Detection
author: Tsung-Yi Lin, Piotr Dollar, Ross Girshick,  Kaiming He, Bharath Hariharan, Serge Belongie  
```
先对深层的特征进行提取，再反过来根据已知的浅层特征指导深层特征进行融合以便于提高语义信息。  

-----------------------------------------------------------------------------------------------------
R-FCN:--->图片的平移不变性与物体检测之间的平平移变换性之间的矛盾。  
```
Object Detection via Region-based Fully Convolutional Networks
author： Jifeng Dai, Yi Li, Kaiming He, Jian Sun
```
对ROI的位置判别更为敏感。  
## 单阶段方法










