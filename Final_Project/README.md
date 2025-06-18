# 机器学习期末项目-点过程模型的研究与改进

我们的项目实现了一种改进的时间点过程模型，用于医疗数据中的事件序列建模和规则学习。该模型能够从时序事件数据中学习规则并预测目标事件的发生。

## 项目结构

项目分为三个主要部分：

```
Final_Project/
├── src/
│   ├── Ours/                  # 我们提出的改进后的点过程模型
│   │   ├── Main.ipynb         # 主要训练和评估流程
│   │   └── Modules.py         # 模型实现和各种工具函数
│   ├── Mimic_Dataprocess/     # 医疗数据集预处理
│   │   ├── Coroheart_Raw_Dataprocess.ipynb  # 冠心病数据处理
│   │   ├── Sepsis_Raw_Dataprocess.ipynb     # 脓毒症数据处理
│   │   └── Stroke_Raw_Dataprocess.ipynb     # 中风数据处理
│   └── Reproduction/          # 对比方法复现
│       ├── Clnn/              # CLNN方法复现
│       ├── Cluster/           # Cluster方法复现
│       └── Teller/            # Teller方法复现
├── README.md                  # 项目说明文档
└── REPORT.pdf                 # 详细的项目报告
```

## 环境要求

- Python 3.8+
- PyTorch 1.9+（推荐使用GPU版本）
- pandas
- numpy
- scikit-learn
- tqdm
- CUDA支持（推荐用于加速模型训练）

## 使用方法

### 数据准备

1. 将医疗数据预处理为CSV格式，包含以下列 ：
   - `id`: 患者ID
   - `t`: 事件发生时间
   - `k`: 事件类型（变量名）
   - `v`: 事件值

   数据集若需要联系我提供。
   
2. 处理数据示例
```python
from Modules import read_data

file_path = "your_dataset.csv"
target_name = "Your_Target_Event"
data, var_name_dict = read_data(file_path, target_varibles=target_name, outliers=0.0)
```

### 模型训练

```python
from Modules import optimize, RuleSet, RuleBasedTPP, train_model

# 使用优化方法自动选择最佳规则
best_rules, best_loss = optimize(
    data, var_name_dict, target_name, max_order=2, 
    num_candidates=10, n_calls=20, device="cpu"
)

# 创建规则集
rule_set = RuleSet(data, var_name_dict)
for rule in best_rules:
    rule_set.add_rule(rule)

# 初始化模型
model = RuleBasedTPP(rule_set.var_name_dict, rule_set.rule_name_dict, rule_set.rule_var_ids, device="cpu")

# 训练模型
loss, output = train_model(model, data, rule_set.rule_event_data, target_name, device="cpu", 
                          num_epochs=100, lr=0.01, patience=5)
```

### 性能评估

```python
# 查看规则权重
for rule_name in model.rule_name_dict:
    weight = round(torch.exp(model.rule_weights[model.rule_name_dict[rule_name]]).item(), 4)
    print(f"{rule_name} -> {target_name}, weight = {weight}")

# 评估指标
print("Final NLL:", output[-1][1])
print("Final MAE:", output[-1][2])
print("Final RMSE:", output[-1][3])
```

## 对比方法

本项目也包含对三种先进时间点过程模型的复现：

1. **CLUSTER**

   先使用Cluster_Dataprocess.ipynb将CSV数据集转化为所需的.npy格式，然后运行：

```bash
python Cluster_Steps.py
```

​	每个数据集运行时间为5-10个小时。

2. **CLNN**

​	直接使用处理后的CSV文件即可：

```bash
python Re_Clnn_main.py
```

​	每个数据集运行时间为12-30个小时。

3. **TELLER**

   先使用Teller_Dataprocess.ipynb将CSV数据集转化为所需的.npy格式，然后运行：

```bash
python train_synthetic.py --dataset_name 数据集名 --algorithm 算法名 --time_limit 时间限制
```

​	每个数据集运行时间为2-5天。

## 数据集

本项目使用了三个医疗数据集：

1. **中风数据集**: 包含患者中风风险从中度到重度的变化
2. **脓毒症数据集**: 包含患者脓毒症低尿量程度变化
3. **冠心病数据集**: 包含冠心病患者的临床事件序列

## 更多信息

详细的模型原理、实验结果和分析请参考 `REPORT.pdf`。