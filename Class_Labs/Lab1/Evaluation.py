import numpy as np
from Dataset_Partitioning import Cross_Validation, Hold_out, Bootstrapping
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

def get_Best_M(train_data, Ms, method, parameters):
    if method == 'Cross Validation':
        T = parameters[0]
        K = parameters[1]
        return get_CV(train_data, Ms, T, K)
    elif method == 'Hold Out':
        test_ratio = parameters[0]
        return get_HO(train_data, Ms, test_ratio)
    elif method == 'Bootstrapping':
        times = parameters[0]
        return get_B(train_data, Ms, times)

def get_CV(train_data, Ms, T, K):
    # T次K折交叉验证，返回最优的M值和所有模型在各个折上的准确率
    best_accuracy = 0
    Best_M = Ms[0]  # 默认值
    all_accuracies = {}  # 存储所有模型在各折各次的准确率
    all_fold_details = {}  # 存储每一折的训练集和验证集大小
    
    # 对每个模型阶数进行评估
    for M in Ms:
        all_accuracies[M] = []
        all_fold_details[M] = []
        total_accuracy = 0
        
        # 增加正则化强度，随着模型复杂度增加而大幅增强
        # 使用指数函数使高阶模型的惩罚大很多
        C_value = 1.0 / (1.0 + 0.1 * np.exp(M-1))  # 指数增长的正则化强度
        
        # 进行T次验证
        for t in range(1, T+1):
            # 进行K折交叉验证
            for k in range(1, K+1):
                fold_train, fold_test = Cross_Validation(train_data, K, k)
                
                # 存储训练集和验证集大小
                if t == 1 and k == 1:  # 只存储第一次第一折的信息
                    all_fold_details[M] = (len(fold_train), len(fold_test))
                
                # 训练模型，增加正则化参数
                model = make_pipeline(
                    PolynomialFeatures(degree=M), 
                    LogisticRegression(C=C_value, max_iter=1000, class_weight='balanced')
                )
                model.fit(fold_train[:, 1:], fold_train[:, 0])
                
                # 在测试集上评估
                outputs = [q for p, q in model.predict_proba(fold_test[:, 1:])]
                
                # 计算准确率
                correct = 0
                for i in range(len(fold_test)):
                    if (fold_test[i][0] == 0 and outputs[i] < 0.5) or (fold_test[i][0] == 1 and outputs[i] >= 0.5):
                        correct += 1
                
                fold_accuracy = correct / len(fold_test)
                all_accuracies[M].append((t, k, fold_accuracy))
                total_accuracy += fold_accuracy
        
        # 计算平均准确率
        avg_accuracy = total_accuracy / (T * K)
        
        # 更新最佳模型
        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            Best_M = M
    
    return Best_M, all_accuracies, best_accuracy, all_fold_details

def get_HO(train_data, Ms, test_ratio):
    # 留出法评估不同阶数的模型
    best_accuracy = 0
    Best_M = Ms[0]  # 默认值
    
    # 对每个模型阶数进行评估
    for M in Ms:
        # 增加正则化强度，随着模型复杂度增加而增强
        C_value = 1.0 / (1.0 + 0.1 * np.exp(M-1))
        
        # 将训练数据再次划分为训练集和验证集
        hold_train, hold_val = Hold_out(train_data, test_ratio)
        
        # 训练模型
        model = make_pipeline(
            PolynomialFeatures(degree=M), 
            LogisticRegression(C=C_value, max_iter=1000, class_weight='balanced')
        )
        model.fit(hold_train[:, 1:], hold_train[:, 0])
        
        # 在验证集上评估
        outputs = [q for p, q in model.predict_proba(hold_val[:, 1:])]
        
        # 计算准确率
        correct = 0
        for i in range(len(hold_val)):
            if (hold_val[i][0] == 0 and outputs[i] < 0.5) or (hold_val[i][0] == 1 and outputs[i] >= 0.5):
                correct += 1
        
        accuracy = correct / len(hold_val)
        
        # 更新最佳模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            Best_M = M
    
    return Best_M

def get_B(train_data, Ms, times):
    # 自助法评估不同阶数的模型
    best_accuracy = 0
    Best_M = Ms[0]  # 默认值
    
    # 对每个模型阶数进行评估
    for M in Ms:
        # 增加正则化强度，随着模型复杂度增加而增强
        C_value = 1.0 / (1.0 + 0.1 * np.exp(M-1))
        
        # 使用自助法生成训练集和测试集
        bootstrap_train, bootstrap_test = Bootstrapping(train_data, times)
        
        # 训练模型
        model = make_pipeline(
            PolynomialFeatures(degree=M), 
            LogisticRegression(C=C_value, max_iter=1000, class_weight='balanced')
        )
        model.fit(bootstrap_train[:, 1:], bootstrap_train[:, 0])
        
        # 在测试集上评估
        outputs = [q for p, q in model.predict_proba(bootstrap_test[:, 1:])]
        
        # 计算准确率
        correct = 0
        for i in range(len(bootstrap_test)):
            if (bootstrap_test[i][0] == 0 and outputs[i] < 0.5) or (bootstrap_test[i][0] == 1 and outputs[i] >= 0.5):
                correct += 1
        
        accuracy = correct / len(bootstrap_test)
        
        # 更新最佳模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            Best_M = M
    
    return Best_M