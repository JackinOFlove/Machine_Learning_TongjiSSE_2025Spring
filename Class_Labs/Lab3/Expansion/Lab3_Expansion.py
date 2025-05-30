import random
import os
# 如果使用GPU，请注释掉下面这行
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Conv2D, Flatten, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.datasets import cifar10
from keras.utils import to_categorical, plot_model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.applications import VGG16, VGG19
import numpy as np
import matplotlib.pyplot as plt

# 固定随机数种子，提高结果可重复性
tf.random.set_seed(233)
np.random.seed(233)

# 加载Cifar10数据集
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# 数据预处理
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255
Y_train = to_categorical(y_train, 10)
Y_test = to_categorical(y_test, 10)

print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)
print("X_test shape:", X_test.shape)
print("Y_test shape:", Y_test.shape)

# 方法1：使用预训练的VGG16模型，修改为适应CIFAR10
def create_vgg_model_pretrained():
    # 创建适用于CIFAR10的VGG模型（不加载预训练权重）
    base_model = VGG16(
        weights=None,  # 不加载预训练权重，因为输入尺寸不同
        include_top=False,  # 不包括顶层分类器
        input_shape=(32, 32, 3)  # CIFAR10图像尺寸
    )
    
    # 添加自定义分类器
    x = base_model.output
    x = Flatten()(x)
    x = Dense(1024)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(10, activation='softmax')(x)
    
    # 创建完整模型
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model

# 方法2：使用VGG架构但适应CIFAR10的尺寸（更轻量级）
def create_vgg_model_custom():
    model = Sequential()
    
    # 输入层
    model.add(Input(shape=(32, 32, 3)))
    
    # Block 1
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Block 2
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Block 3
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    # Block 4
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    # Block 5 (减少以适应CIFAR10)
    model.add(Flatten())
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    
    return model

# 训练和评估函数
def train_and_evaluate(model, model_name):
    # 编译模型
    adam = Adam(learning_rate=0.0005)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    
    # 显示模型结构
    model.summary()
    
    # 绘制模型结构图
    plot_model(model, to_file=f'{model_name}.png', show_shapes=True, show_layer_names=False, rankdir='TB')
    
    # 回调函数
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    
    # 训练模型
    # 对于CPU训练，可以减少epochs或使用更小的数据集
    # batch_size可以根据GPU内存情况调整
    batch_size = 64  # 如果内存不足，可以减小到32或16
    history = model.fit(X_train, Y_train, 
                        batch_size=batch_size, 
                        epochs=50,
                        validation_split=0.2,
                        shuffle=True, 
                        verbose=2, 
                        callbacks=[reduce_lr, early_stopping])
    
    # 评估模型
    print(f"\nEvaluating {model_name}...")
    scores = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=1)
    print(f"{model_name} - Test loss: {scores[0]}")
    print(f"{model_name} - Test accuracy: {scores[1]}")
    
    # 绘制训练曲线
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    acc_values = history_dict['accuracy']
    val_acc_values = history_dict['val_accuracy']
    epochs_x = range(1, len(loss_values) + 1)
    
    # 损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_x, loss_values, 'b', label='Training loss')
    plt.plot(epochs_x, val_loss_values, 'orange', label='Validation loss')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(False)
    plt.savefig(f'{model_name}_loss.png')
    plt.show()
    
    # 准确率曲线
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_x, acc_values, 'b', label='Training accuracy')
    plt.plot(epochs_x, val_acc_values, 'orange', label='Validation accuracy')
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(False)
    plt.savefig(f'{model_name}_accuracy.png')
    plt.show()
    
    return scores, history

# 主函数
def main():
    # 使用哪种VGG模型 (选择一个)
    use_pretrained = True  # 是否使用预训练VGG架构
    
    if use_pretrained:
        print("\n===== USING PRETRAINED VGG MODEL ON CIFAR10 =====")
        vgg_model = create_vgg_model_pretrained()
        model_name = "VGG16_Pretrained"
    else:
        print("\n===== USING CUSTOM VGG MODEL ON CIFAR10 =====")
        vgg_model = create_vgg_model_custom()
        model_name = "VGG_Custom"
    
    vgg_scores, vgg_history = train_and_evaluate(vgg_model, model_name)
    
    print(f"\n{model_name} - Test Accuracy: {vgg_scores[1]}")

if __name__ == "__main__":
    main()