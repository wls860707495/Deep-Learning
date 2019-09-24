# Inception.V3
from keras import layers

branch_a = layers.Conv2D(128,1,activation='relu',strides=2)(x)
branch_b = layers.Conv2D(128,1,activation='relu')(x)
branch_b = layers.Conv2D(128,3,activation='relu',strides=2)(branch_b)

branch_c = layers.AveragePooling2D(3,strides=2)(x)
branch_c = layers.Conv2D(128,3,activation='relu')(branch_c)

branch_d = layers.Conv2D(128,1,activation='relu')(x)
branch_d = layers.Conv2D(128,3,activation='relu')(branch_d)
branch_d = layers.Conv2D(128,3,activation='relu',strides=2)(branch_d)

output = layers.concatenate([branch_a,branch_b,branch_c,branch_d],axis=-1)