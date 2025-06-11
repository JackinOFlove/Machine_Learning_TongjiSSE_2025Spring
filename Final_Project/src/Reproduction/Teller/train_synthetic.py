# 该文件实现了 Teller 论文的复现工作
# python train_synthetic.py --dataset_name 数据集名 --algorithm 算法名 --time_limit 时间限制
# 数据集首先需要通过在 data 文件夹下的 Teller_Dataprocess.ipynb 文件处理得到 .npy 文件
# 本篇论文的主题工作在 logic_learning.py 文件中得到了完整的实现
# train_synthetic.py 文件中实现了模型的训练，包括模型的定义、数据集的输入、模型的训练、模型的评估、模型的保存, 这是我们主要工作的地方
# 我们添加了数据集输入，将数据集输入修改为.npy文件，然后跑通我们的三个数据集
# 我们添加了指标计算，在 train_synthetic.py 中，我们添加了指标计算，包括 MAE\RMSE\NLL
# 我们添加了日志输出和错误输出，可以在 log 文件夹下查看实验过程，可以使用 nohup 或者 screen 实现后台运行
# 我们添加了规则的获取，选择权重较大的规则并且满足时间关系的规则，最后可以在 log 文件夹下查看具体的规则

import datetime
import os
import argparse

import numpy as np
import torch

from logic_learning import Logic_Learning_Model
from utils import redirect_log_file, Timer
from generate_synthetic_data import Logic_Model_Generator

# 获取数据
def get_data(dataset_id, dataset_name, num_sample):
    if dataset_id == 0:
        if dataset_name == "Stroke":
            dataset_path = './data/data-stroke.npy'
        elif dataset_name == "Sepsis":
            dataset_path = './data/data-sepsis.npy'
        elif dataset_name == "Coroheart":
            dataset_path = './data/data-coroheart.npy'
    else:
        dataset_path = './data/data-{}.npy'.format(dataset_id)
    print("dataset_path is ",dataset_path)
    dataset = np.load(dataset_path, allow_pickle='TRUE').item()
    if len(dataset.keys())> num_sample: 
        dataset = {i:dataset[i] for i in range(num_sample)}
    num_sample = len(dataset)
    training_dataset = {i: dataset[i] for i in range(int(num_sample*0.8))}
    testing_dataset = {i: dataset[int(num_sample*0.8)+i] for i in range(int(num_sample*0.2))}

    print("sample num is ", num_sample)
    print("training_dataset size=", len(training_dataset))
    print("testing_dataset size=", len(testing_dataset))
    return training_dataset, testing_dataset

# 获取逻辑模型
def get_logic_model_stroke():
    file_name = ""
    var_name_dict = np.load("data/stroke_var_name_dict.npy", allow_pickle='TRUE').item()
    model = Logic_Model_Generator()
    model.body_predicate_set = list(var_name_dict.values())
    model.head_predicate_set = [var_name_dict["Middle_to_Sever"]]
    model.body_predicate_set.remove(var_name_dict["Middle_to_Sever"])
    model.survival_pred_set = [model.head_predicate_set[0]]
    model.predicate_notation = list(var_name_dict.keys())
    model.num_predicate = len(model.body_predicate_set)    
    return model, file_name

def get_logic_model_sepsis():
    file_name = ""
    var_name_dict = np.load("data/sepsis_var_name_dict.npy", allow_pickle='TRUE').item()
    model = Logic_Model_Generator()
    model.body_predicate_set = list(var_name_dict.values())
    model.head_predicate_set = [var_name_dict["Low Urine"]]
    model.body_predicate_set.remove(var_name_dict["Low Urine"])
    model.survival_pred_set = [model.head_predicate_set[0]]
    model.predicate_notation = list(var_name_dict.keys())
    model.num_predicate = len(model.body_predicate_set)    
    return model, file_name

def get_logic_model_coroheart():
    file_name = ""
    var_name_dict = np.load("data/coroheart_var_name_dict.npy", allow_pickle='TRUE').item()
    model = Logic_Model_Generator()
    model.body_predicate_set = list(var_name_dict.values())
    model.head_predicate_set = [var_name_dict["Goal event"]]
    model.body_predicate_set.remove(var_name_dict["Goal event"])
    model.survival_pred_set = [model.head_predicate_set[0]]
    model.predicate_notation = list(var_name_dict.keys())
    model.num_predicate = len(model.body_predicate_set)    
    return model, file_name

def get_model(dataset_id, dataset_name):
    if dataset_id == 0:
        if dataset_name == "Stroke":
            m, _ = get_logic_model_stroke()
        elif dataset_name == "Sepsis":
            m, _ = get_logic_model_sepsis()
        elif dataset_name == "Coroheart":
            m, _ = get_logic_model_coroheart()
    else:
        from generate_synthetic_data import get_logic_model_1,get_logic_model_2,get_logic_model_3,get_logic_model_4,get_logic_model_5,get_logic_model_6,get_logic_model_7,get_logic_model_8,get_logic_model_9,get_logic_model_10,get_logic_model_11,get_logic_model_12
        logic_model_funcs = [None,get_logic_model_1,get_logic_model_2,get_logic_model_3,get_logic_model_4,get_logic_model_5,get_logic_model_6,get_logic_model_7,get_logic_model_8,get_logic_model_9,get_logic_model_10,get_logic_model_11,get_logic_model_12]
        m, _ = logic_model_funcs[dataset_id]()
    model = m.get_model_for_learn()
    return model

def fit(dataset_id, dataset_name, num_sample, time_limit, worker_num=8, num_epoch=5, algorithm="RAFS"):
    """Train synthetic data set, define hyper-parameters here."""
    t  = datetime.datetime.now()
    print("Start time is", t ,flush=1)
    if not os.path.exists("./model"):
        os.makedirs("./model")    
    model = get_model(dataset_id, dataset_name)
    training_dataset, testing_dataset =  get_data(dataset_id, dataset_name, num_sample)

    #set model hyper params
    model.time_limit = time_limit
    model.num_epoch = num_epoch
    model.worker_num = worker_num
    model.print_time = False
    model.weight_lr = 0.0005
    model.l1_coef = 0.1

    if model.use_exp_kernel:
        model.init_base = 0.01
        model.init_weight = 0.1
    else:
        model.init_base = 0.2
        model.init_weight = 0.1
    
    if algorithm == "Brute":
        #smaller init weight and  smaller lr
        model.init_weight = 0.01
        model.weight_lr = 0.0001

    if dataset_id in [0]:
        model.max_rule_body_length = 2
        model.max_num_rule = 10
        model.weight_threshold = 0.05
        model.strict_weight_threshold= 0.1
    elif dataset_id in [2,8]:
        model.max_rule_body_length = 2
        model.max_num_rule = 15
        model.weight_threshold = 0.05
        model.strict_weight_threshold= 0.1
    elif dataset_id in [3,9]:
        model.max_rule_body_length = 2
        model.max_num_rule = 15
        model.weight_threshold = 0.2
        model.strict_weight_threshold= 0.5
    elif dataset_id in [4,10]:
        model.max_rule_body_length = 3
        model.max_num_rule = 20
        model.weight_threshold = 0.1
        model.strict_weight_threshold= 0.3
    elif dataset_id in [5,11]:
        model.max_rule_body_length = 2
        model.max_num_rule = 20
        model.weight_threshold = 0.1
        model.strict_weight_threshold= 0.3
    elif dataset_id in [1,6,7,12]:
        model.max_rule_body_length = 2
        model.max_num_rule = 15
        model.weight_threshold = 0.1
        model.strict_weight_threshold= 0.3
    else:
        print("Warning: Hyperparameters not set!")

    if dataset_id in [1, 6, 7, 8, 11, 12]:
        model.weight_lr = 0.0001

    if algorithm == "REFS":
        with Timer("REFS") as t:
            model.REFS(model.head_predicate_set[0], training_dataset, testing_dataset, tag = dataset_id)
    elif algorithm == "RAFS":
        with Timer("RAFS") as t:
            model.RAFS(model.head_predicate_set[0], training_dataset, testing_dataset, tag = dataset_id)
    elif algorithm == "Brute":
        with Timer("Brute") as t:
            model.Brute(model.head_predicate_set[0], training_dataset)
    if dataset_id == 0:
        if dataset_name == "Stroke":
            nll, mae, rmse = model.evaluate(head_predicate_idx=48, dataset=testing_dataset)
            with open('log/out/stroke_results.txt', 'w') as file:
                file.write(f"NLL: {nll.item()}\n")
                file.write(f"MAE: {mae.item()}\n")
                file.write(f"RMSE: {rmse.item()}\n")
            print("NLL: ", nll, "MAE: ", mae, "RMSE: ", rmse)
        elif dataset_name == "Sepsis":
            nll, mae, rmse = model.evaluate(head_predicate_idx=66, dataset=testing_dataset)
            with open('log/out/sepsis_results.txt', 'w') as file:
                file.write(f"NLL: {nll.item()}\n")
                file.write(f"MAE: {mae.item()}\n")
                file.write(f"RMSE: {rmse.item()}\n")
            print("NLL: ", nll, "MAE: ", mae, "RMSE: ", rmse)
        elif dataset_name == "Coroheart":
            nll, mae, rmse = model.evaluate(head_predicate_idx=56, dataset=testing_dataset)
            with open('log/out/coroheart_results.txt', 'w') as file:
                file.write(f"NLL: {nll.item()}\n")
                file.write(f"MAE: {mae.item()}\n")
                file.write(f"RMSE: {rmse.item()}\n")
            print("NLL: ", nll, "MAE: ", mae, "RMSE: ", rmse)
    elif dataset_id == 1:
        nll, mae, rmse = model.evaluate(head_predicate_idx=4, dataset=testing_dataset)
        with open('log/out/dataset_1_results.txt', 'w') as file:
            file.write(f"NLL: {nll.item()}\n")
            file.write(f"MAE: {mae.item()}\n")
            file.write(f"RMSE: {rmse.item()}\n")
        print("NLL: ", nll, "MAE: ", mae, "RMSE: ", rmse)
    print("Finish time is", datetime.datetime.now())
 

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_id', type=int, 
        help="an integer between 1 and 12, indicating one of 12 datasets",
        default=0,
        choices=list(range(0,13)))
    parser.add_argument('--dataset_name', type=str, 
        help="an integer between 1 and 12, indicating one of 12 datasets",
        default="Stroke",
        choices=["Stroke", "Sepsis", "Coroheart"])
    parser.add_argument('--algorithm', type=str, 
        help="which seaching scheme to use, possible choices are [RAFS,REFS,Brute].",
        default="RAFS",
        choices=["RAFS","REFS","Brute"])
    parser.add_argument('--time_limit', type=float, 
        help="maximum running time (seconds)",
        default=3600 * 24, # 24 hours
        )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # 设置多进程通信策略，取决于操作系统
    torch.multiprocessing.set_sharing_strategy('file_system') 

    args = get_args()
    # 重定向stdout和stderr到日志文件
    redirect_log_file() 

    fit(dataset_id=args.dataset_id, dataset_name=args.dataset_name, time_limit=args.time_limit, num_sample=60000, worker_num=12, num_epoch=12, algorithm=args.algorithm)
    