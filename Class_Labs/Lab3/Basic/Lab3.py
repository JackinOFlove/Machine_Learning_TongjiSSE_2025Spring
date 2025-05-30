import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf
from keras.models import Sequential  # 采用贯序模型
from keras.layers import Input, Dense, Dropout, Activation, Conv2D, Flatten, BatchNormalization, MaxPooling2D
from keras.models import Model
# from keras.optimizer_v2.gradient_descent import SGD
from keras.optimizers import SGD
from keras.datasets import mnist
# from keras.utils.np_utils import to_categorical
from keras.utils import to_categorical
from keras.callbacks import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt

# 固定随机数种子，提高结果可重复性（在CPU上测试有效）
tf.random.set_seed(233)
np.random.seed(233)

'''第一步：选择模型'''
model = Sequential()  # 采用贯序模型

'''第二步：构建网络层'''
# 在此处构建你的网络
#####################################################################################
# 输入层
model.add(Input(shape=(28, 28, 1)))

# 简化的卷积网络结构
# 第一个卷积层 - 32个3x3卷积核
model.add(Conv2D(32, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))  # 14x14

# 第二个卷积层 - 64个3x3卷积核
model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))  # 7x7
model.add(Dropout(0.25))

# Flatten层 - 将卷积特征展平
model.add(Flatten())

# 一个简单的全连接层 - 128个神经元
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# 输出层 - 10个神经元，对应10个数字类别
model.add(Dense(10))
# softmax激活函数进行多分类
model.add(Activation('softmax'))
#####################################################################################

'''第三步：网络优化/编译/模型输出'''
# 在此处调整优化器
# learning_rate：大于0的浮点数，学习率
# momentum：大于0的浮点数，动量参数
# decay：大于0的浮点数，每次更新后的学习率衰减值
# nesterov：布尔值，确定是否使用Nesterov动量

# 使用Adam优化器，通常比SGD收敛更快，对初始学习率不太敏感
from keras.optimizers import Adam
adam = Adam(learning_rate=0.001)  # 使用适中的学习率

# 在此处调整损失函数，并编译网络
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])  # 使用交叉熵作为loss函数

# 在此处输出网络的架构。此处参数可以不用调整。
# model表示自定义的模型名 to_file表示存储的文件名 show_shapes是否显示形状  rankdir表示方向T(top)B(Bottow)
from keras.utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False, rankdir='TB')

'''第四步：训练'''

# 数据集获取 mnist 数据集的介绍可以参考 https://blog.csdn.net/simple_the_best/article/details/75267863
(X_train, y_train), (X_test, y_test) = mnist.load_data()  # 使用Keras自带的mnist工具读取数据（第一次运行需要联网）

# 数据处理与归一化
# 注意：X_train和X_test可以直接输入卷积层，但需要先Flatten才能输入全连接层
X_train = X_train.reshape((60000, 28, 28, 1)).astype('float') / 255
X_test = X_test.reshape((10000, 28, 28, 1)).astype('float') / 255

# 生成OneHot向量
Y_train = to_categorical(y_train)
Y_test = to_categorical(y_test)

# 在此处调整训练细节
'''
   .fit的一些参数
   batch_size：对总的样本数进行分组，每组包含的样本数量
   epochs ：训练次数
   shuffle：是否把数据随机打乱之后再进行训练
   validation_split：拿出百分之多少用来做交叉验证
   verbose：屏显模式 0：不输出  1：输出进度  2：输出每次的训练结果
'''
# 添加学习率降低的回调函数，当验证准确率不再提高时降低学习率
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3, min_lr=0.00001, verbose=1, mode='max')

# 添加早停策略，防止过拟合
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1, mode='max')

# 设置合理的训练轮数，但使用早停策略来决定实际训练轮数
history = model.fit(X_train, Y_train, batch_size=128, epochs=15,
                    shuffle=True, verbose=2, validation_split=0.1,
                    callbacks=[reduce_lr, early_stopping])

'''第五步：输出与可视化'''
print("test set")
# 误差评价 ：按batch计算在batch用到的输入数据上模型的误差，并输出测试集准确率
scores = model.evaluate(X_test, Y_test, batch_size=128, verbose=1)
print("The test loss is %f" % scores[0])
print("The accuracy of the model is %f" % scores[1])

# 在此处实现你的可视化功能
#####################################################################################
# 可视化训练曲线
# 获取训练历史数据
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
epochs = range(1, len(loss_values) + 1)

# 绘制损失曲线
plt.figure(figsize=(10, 6))
plt.plot(epochs, loss_values, 'b', label='training loss')
plt.plot(epochs, val_loss_values, 'orange', label='val loss')
plt.title('model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.grid(False)
plt.savefig('loss_curve.png')  # 保存图片
plt.show()

# 绘制准确率曲线
plt.figure(figsize=(10, 6))
plt.plot(epochs, acc_values, 'b', label='training accuracy')
plt.plot(epochs, val_acc_values, 'orange', label='val accuracy')
plt.title('model accuracy')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.legend()
plt.ylim(0.85, 1.01)  # 设置Y轴范围使图形更清晰
plt.grid(False)
plt.savefig('accuracy_curve.png')  # 保存图片
plt.show()
#####################################################################################


#####################################################################################


