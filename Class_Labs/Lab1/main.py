import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import warnings
from sklearn.exceptions import ConvergenceWarning

from Dataset_Partitioning import Hold_out, Bootstrapping, Cross_Validation
from Drawing import drawing_data, drawing_model, drawing_PR, drawing_ROC, drawing_models, drawing_PRs, drawing_ROCs
from Dataset import dataset
from Evaluation import get_Best_M

# 忽略特定类型的警告
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# 对M in Ms阶逻辑回归分类器，进行T次K折交叉验证，选择最优模型
# 绘制最佳模型在测试集上的ROC和PR图
# 绘制几个模型在测试集上的ROC和PR图

N = 2000 # 数据集的数量级
Ms = [1, 2, 5] # 从M阶逻辑回归分类器中选择最优
T = 3
K = 5 # T次K折交叉验证

# 数据集生成
data = dataset(N) # 可尝试生成不同的数据集
drawing_data(data, '数据集分布')

# 数据集划分
train_data, test_data = Hold_out(data, 0.5) # 留出法
# train_data, test_data = Bootstrapping(data, 2000) # 自助法

print(f'训练集样本数量：{len(train_data)}')
print(f'测试集样本数量：{len(test_data)}')

# 定义M阶逻辑回归模型
best_M, all_accuracies, best_cv_accuracy, all_fold_details = get_Best_M(train_data, Ms, 'Cross Validation', [T, K])

# 打印各阶模型在不同次数的交叉验证中的准确率
for M in Ms:
    # 打印训练集和验证集样本数量
    train_size, val_size = all_fold_details[M]
    print(f'训练集样本数量：{train_size}')
    print(f'验证集样本数量：{val_size}')
    
    # 打印每一次每一折的准确率
    for t, k, acc in all_accuracies[M]:
        print(f'{M}阶逻辑回归回归模型、第{t}次、第{k}折的准确率为{acc:.2%}')
    
    # 计算此阶模型的平均准确率
    avg_acc = sum([acc for _, _, acc in all_accuracies[M]]) / len(all_accuracies[M])
    print(f'{M}阶逻辑回归回归模型{T}次{K}折交叉检验的平均准确率为{avg_acc:.2%}')
    print('......')

print(f'最佳模型为{best_M}阶逻辑回归回归模型，其在交叉验证法验证集上的平均准确率为{best_cv_accuracy:.2%}')

# 在完整train_data上训练最佳模型
C_value = 1.0 / (1.0 + 0.1 * np.exp(best_M-1))  # 指数增长的正则化强度
best_model = make_pipeline(
    PolynomialFeatures(degree=best_M), 
    LogisticRegression(C=C_value, max_iter=1000, class_weight='balanced')
)
best_model.fit(train_data[:, 1:], train_data[:, 0])

# 结果
output = [q for p, q in best_model.predict_proba(test_data[:, 1:])]
# output取0~1，设定阈值，将其分类0、1两类

# 评价
boolnum = len(test_data)
boolT = 0
for i in range(boolnum):
    if test_data[i][0] == 0:
        if output[i] < 0.5: # 这里以0.5为分界进行分类
            boolT += 1
    elif test_data[i][0] == 1:
        if output[i] >= 0.5:
            boolT += 1

Accuracy = boolT / boolnum # 可尝试其他评价指标
print(f"{best_M}阶逻辑回归回归分类器在测试集上的准确率：{Accuracy:.2%}")

# 绘图
# 最佳模型：
drawing_model(train_data, best_model, str(best_M) + '阶逻辑回归模型在训练集上的结果')
drawing_model(test_data, best_model, str(best_M) + '阶逻辑回归模型在测试集上的结果')
drawing_PR(test_data[:, 0], output, str(best_M) + '阶模型的PR曲线图')
drawing_ROC(test_data[:, 0], output, str(best_M) + '阶模型的ROC曲线图')
# 所有模型：
models = []
outputs = []
for M in Ms:
    # 对每个模型使用相同的正则化策略
    C_value = 1.0 / (1.0 + 0.1 * np.exp(M-1))
    M_model = make_pipeline(
        PolynomialFeatures(degree=M), 
        LogisticRegression(C=C_value, max_iter=1000, class_weight='balanced')
    )
    M_model.fit(train_data[:, 1:], train_data[:, 0])
    models.append(M_model)
    output = [q for p, q in M_model.predict_proba(test_data[:, 1:])]
    outputs.append(output)
drawing_models(models, test_data, Ms, '各阶逻辑回归分类器在测试集上的表现')
drawing_PRs(outputs, test_data, Ms, '各阶逻辑回归分类器PR曲线')
drawing_ROCs(outputs, test_data, Ms, '各阶逻辑回归分类器ROC曲线')
plt.show()

