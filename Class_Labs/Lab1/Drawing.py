import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score, roc_auc_score

# matplotlib画图中中文显示会有问题，需要这两行设置默认字体可以显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def drawing_data(data, title):
    Label = []
    Input1 = []
    Input2 = []

    for d in data:
        Label.append(d[0])
        Input1.append(d[1])
        Input2.append(d[2])

    plt.figure(title)

    for i in range(len(Label)):
        if Label[i] == 1:
            plt.scatter(Input1[i], Input2[i], c='r', marker='+')
        else:
            plt.scatter(Input1[i], Input2[i], c='b', marker='.')

    plt.title(title)
    plt.xlabel('坐标x')
    plt.ylabel('坐标y')
    plt.xlim(-8, 6)
    plt.ylim(-8, 6)

def drawing_model(data, model, title):
    Label = []
    Input1 = []
    Input2 = []

    for d in data:
        Label.append(d[0])
        Input1.append(d[1])
        Input2.append(d[2])

    plt.figure(title)
    xx, yy = np.meshgrid(np.arange(-8, 6, 0.01),
                         np.arange(-8, 6, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    #　plt.contour(xx, yy, Z, colors='k', linewidths=1.5)

    for i in range(len(Label)):
        if Label[i] == 1:
            plt.scatter(Input1[i], Input2[i], c='r', marker='+')
        else:
            plt.scatter(Input1[i], Input2[i], c='b', marker='.')

    plt.title(title)
    plt.xlabel('坐标x')
    plt.ylabel('坐标y')
    plt.xlim(-8, 6)
    plt.ylim(-8, 6)

def drawing_models(models, test_data, Ms, title):
    Label = []
    Input1 = []
    Input2 = []

    for d in test_data:
        Label.append(d[0])
        Input1.append(d[1])
        Input2.append(d[2])

    plt.figure(title)


    for i in range(len(Label)):
        if Label[i] == 1:
            plt.scatter(Input1[i], Input2[i], c='r', marker='+')
        else:
            plt.scatter(Input1[i], Input2[i], c='b', marker='.')

    xx, yy = np.meshgrid(np.arange(-8, 6, 0.01),
                         np.arange(-8, 6, 0.01))
    for M_model in models:
        Z = M_model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        #plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
        plt.contour(xx, yy, Z, colors='k', linewidths=1.5)

    plt.title(title)
    plt.xlabel('坐标x')
    plt.ylabel('坐标y')
    plt.xlim(-8, 6)
    plt.ylim(-8, 6)

"""
    设0为True，1为false
        predict
    gt  0       1
    0   TP      FN
    1   FP      TN

"""

def drawing_PR(Label, Output, title):
    # 绘制单个模型的PR曲线
    y_true = np.array(Label)
    y_scores = np.array(Output)
    
    # 计算精确率、召回率和对应的阈值
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    
    # 计算平均精确率（AP）
    ap = average_precision_score(y_true, y_scores)
    
    # 绘制PR曲线
    plt.figure(title)
    plt.plot(recall, precision, lw=2, label=f'PR曲线')
    
    # 添加图例和标签
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

def drawing_ROC(Label, Output, title):
    # 绘制单个模型的ROC曲线
    y_true = np.array(Label)
    y_scores = np.array(Output)
    
    # 计算ROC曲线的点和AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # 绘制ROC曲线
    plt.figure(title)
    plt.plot(fpr, tpr, lw=2, label=f'ROC曲线')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    # 添加图例和标签
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

def drawing_PRs(outputs, test_data, Ms, title):
    # 绘制多个模型的PR曲线
    y_true = test_data[:, 0]
    
    plt.figure(title)
    
    # 为每个模型绘制PR曲线
    for i, y_scores in enumerate(outputs):
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)
        plt.plot(recall, precision, lw=2, label=f'{Ms[i]}阶逻辑回归分类器PR曲线')
    
    # 添加图例和标签
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

def drawing_ROCs(outputs, test_data, Ms, title):
    # 绘制多个模型的ROC曲线
    y_true = test_data[:, 0]
    
    plt.figure(title)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    # 为每个模型绘制ROC曲线
    for i, y_scores in enumerate(outputs):
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{Ms[i]}阶逻辑回归分类器ROC曲线')
    
    # 添加图例和标签
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
