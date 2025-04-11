import random
import numpy as np
import math

# 由于是二分类问题，所以这里默认数据集label只有0和1，大家也可以挑战更高难度（未知标签情况）

# fold折交叉验证法的第k次（即取fold份数据中的第k份作为测试集,k从1开始）
def Cross_Validation(data, fold, k):
    # 按类别分组数据
    class0 = []
    class1 = []
    
    for d in data:
        if d[0] == 0:
            class0.append(d)
        else:
            class1.append(d)
    
    # 计算每个类别在每个折叠中的数量
    num_class0_per_fold = len(class0) // fold
    num_class1_per_fold = len(class1) // fold
    
    # 准备训练集和测试集
    train_data = []
    test_data = []
    
    # 对类别0进行分割
    start_idx0 = (k-1) * num_class0_per_fold
    end_idx0 = k * num_class0_per_fold if k < fold else len(class0)
    
    test_data.extend(class0[start_idx0:end_idx0])
    train_data.extend(class0[:start_idx0] + class0[end_idx0:])
    
    # 对类别1进行分割
    start_idx1 = (k-1) * num_class1_per_fold
    end_idx1 = k * num_class1_per_fold if k < fold else len(class1)
    
    test_data.extend(class1[start_idx1:end_idx1])
    train_data.extend(class1[:start_idx1] + class1[end_idx1:])
    
    return np.array(train_data), np.array(test_data)

# 测试样本占比为test_ratio的留出法
def Hold_out(data, test_ratio):
    class0 = []
    class1 = []

    # 验证集划分
    for d in data:
        if d[0] == 0:
            class0.append(d)
        else:
            class1.append(d)

    train_data = []
    test_data = []

    # 验证集划分
    for i in range(len(class0)):
        if i < len(class0)*test_ratio:
            test_data.append(class0[i])
        else:
            train_data.append(class0[i])

    for i in range(len(class1)):
        if i < len(class1)*test_ratio:
            test_data.append(class1[i])
        else:
            train_data.append(class1[i])

    return np.array(train_data), np.array(test_data)

# 训练样本抽样times次的自助法
def Bootstrapping(data, times):
    n = len(data)
    # 原始数据集作为测试集
    test_data = np.array(data)
    
    # 通过有放回抽样生成训练集
    train_indices = np.random.randint(0, n, size=times)
    train_data = np.array([data[i] for i in train_indices])
    
    return train_data, test_data