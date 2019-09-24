# 如果验证损失不再改善，你可以使用这个回调函数来降低学习率。在训练过程中如果出现
# 了损失平台（loss plateau），那么增大或减小学习率都是跳出局部最小值的有效策略
# import keras
#
# callbacks_list = [
#                     keras.callbacks.ReduceLROnPlateau(
#                         monitor='val_loss'
#                         factor = 0.1,        # 触发时将学习率除以 10
#                         patience = 10,)]     # 如果验证损失在10轮内都没有改善，那么就触发这个回调函数
#
# model.fit(x, y, epochs=10, batch_size=32, callbacks=callbacks_list, validation_data=(x_val, y_val))
