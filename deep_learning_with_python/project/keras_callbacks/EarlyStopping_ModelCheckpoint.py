import keras

x =[]
y = []
x_val = []
y_val = []

callbacks_list = [      # 通过fit的callback参数将回调函数传入模型中，这个参数 接收一个回调函数的列表。你可以传入任意个数的回调函数
                  keras.callbacks.EarlyStopping(
                                    monitor='acc',       # 监控模型的验证精度
                                    patience=1,          # 如果精度在多于一轮的时间（即两轮）内不再改善，中断训练
                                                ),
                  keras.callbacks.ModelCheckpoint(               # 在每轮过后保存当前权重
                                    filepath='my_model.h5',      # 目标模型文件的保存路径
                                    monitor='val_loss',          # 这两个参数的含义是，如果val_loss没有改善，那么不需要覆盖模型文件。这就可以始终保存在训练过程中见到的最佳模型
                                    save_best_only=True, )]

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])                # 你监控精度，所以它应该是模型指标的一部分

# 注意，由于回调函数要监控验证损失和验证精度，所以在调用fit时需要传入validation_data（验证数据）
model.fit(x, y,
          epochs=10,
          batch_size=32,
          callbacks=callbacks_list,
          validation_data=(x_val, y_val))