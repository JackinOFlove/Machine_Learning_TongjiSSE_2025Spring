# -*- coding: utf-8 -*-
'''
前两个模块
input:
{
    "sequence_1": [(0, "A"), (1, "B"), (2, "A"), (3, "C")],
    "sequence_2": [(0, "B"), (2, "D"), (4, "C")],
}
output:
时钟信号矩阵(标准化)
'''
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import os
from itertools import combinations, permutations
import csv
'''
# Load train, val, test data
def Load_file(pathfile):
    df_train = np.load(pathfile + "train", allow_pickle = True)
    df_val = np.load(pathfile + "dev", allow_pickle = True)
    df_test = np.load(pathfile + "test", allow_pickle = True)
    
    return df_train, df_val, df_test
'''
# 将csv文件转换为字典   
def csv_to_dict(file_path):
    df = pd.read_csv(file_path)
    df.head() 
    data_dict = {}
    current_id = 1  
    data_dict[current_id] = []  
    for _, row in df.iterrows():
        try:
            t = row['t']
            k = row['k']
            
            # 如果事件类型为 'Middle_to_Sever'，则创建一个新的序列
            # 这是 Stroke 数据集的目标事件，Sepsis 和 Corohheart 数据集的目标事件这里还需要修改
            if k == 'Middle_to_Sever':
                # 记录当前事件并切换到新 ID
                data_dict[current_id].append((t, k))
                current_id += 1 
                data_dict[current_id] = []  # 新建一个空列表用于存储下一个 ID 的事件
            else:
                # 否则继续添加事件到当前 ID 下
                data_dict[current_id].append((t, k))
        
        except Exception as e:
            print(f"处理行时发生错误: {row} 错误信息: {e}")
    # 删除最后一个键值对，如果它对应的值是空列表
    if not data_dict[current_id]:
        del data_dict[current_id]
    return data_dict

# 提取事件类型集合和事件流的最大长度
def Filter_set(data_train):
    event_set = set()
    max_length = 0
    
    for events in data_train.values():
        if events:
            events_cur = [event[1] for event in events]
            event_set = event_set | set(events_cur)
            times_cur = events[-1][0]
            max_length = max(max_length, times_cur)
    event_set = list(event_set)
    event_set.sort()
        
    return event_set, max_length, 1.5 * max_length


# 对事件进行初步掩码处理,去除重复事件只保留最近一次
def FirstMask_Event(lbs, clks, small_idx):
    """
    对事件进行初步掩码处理,去除重复事件只保留最近一次
    
    输入:
    - lbs: 事件标签列表
    - clks: 对应的时间戳列表
    - small_idx: 小于当前时间的索引
    
    输出:
    - lbs_new: 去重后的事件标签
    - clks_new: 对应的时间戳
    
    示例:
    输入:
    lbs = ['A','B','A','C']
    clks = [1,2,3,4]
    small_idx = [0,1,2]
    
    输出:
    ['B','A'], [2,3]  # 只保留每个事件最近一次出现
    """
    lbs_new, clks_new = [], []
    for i in small_idx:
       if lbs[i] not in lbs_new:
           lbs_new.append(lbs[i])
           clks_new.append(clks[i])
    return lbs_new, clks_new
    

# 将事件序列转换为时钟信号
def Encode_curclk(data_clks, data_lbs, clk_cur, event_set, target, maxT):
    """
    将事件序列转换为时钟信号
    
    输入:
    - data_clks: 时间戳列表
    - data_lbs: 事件标签列表
    - clk_cur: 当前时间点
    - event_set: 事件类型集合
    - target: 目标事件
    - maxT: 最大时间窗口
    
    输出:
    - clocks_all: 时钟信号矩阵 [1, 事件类型数]
    
    示例:
    输入序列 [(1,'A'), (2,'B'), (4,'A')]
    当前时间 clk_cur = 3
    event_set = ['A','B','C']
    
    输出:
    [[2, 1, maxT]]  # A距离现在2个单位,B距离1个单位,C未出现用maxT表示
    """
    clocks_all = np.ones((1,len(event_set))) * maxT
    small_idx = np.where(np.array(data_clks) < clk_cur)[0]
    data_lbs_mask, data_clks_mask = FirstMask_Event(data_lbs, data_clks, small_idx)
    
    for i in range(len(data_lbs_mask)):
        clocks_all[0, event_set.index(data_lbs_mask[i])] = clk_cur \
        - data_clks_mask[i]
    
    return clocks_all
    

# 将事件序列转换为时钟信号用于训练
def Find_target(target_event, data_train, event_set, T_max, T_reso, Intval_max):
    num=0
    lock=False
    events_clocksdur = []
    events_clocksstep = []
    step_list = []  # 用于保存每个目标事件的 dur
    dur_list=[]
    target_list=[]
    length_dur=[]
    length_step=[]
    Tmax = 50
    i=0
    for data in data_train.values():
        i+=1
        print(data)
        
        if data:
            data_new = data[:]
            data_clks = ([di[0] for di in data_new])
            data_lbs = np.array([di[1] for di in data_new])
            clk_max = max(data_clks)
            clk_min = min(data_clks)
            # 统计当前序列中的事件数量
            length_step.append(len(data_new))

            # 统计时间网格数量
            grid_count = len(np.arange(0, clk_max, T_reso))
            length_dur.append(grid_count)
            print(length_step)
            print(length_dur)
            # print(f"datanew:{data_new}")            
            lb_idx = 0
            for lb_cur in data_lbs:
                if lb_cur == target_event:
                    clk_lbcur = data_clks[lb_idx]
                    clock_all = Encode_curclk(data_clks, data_lbs, clk_lbcur, \
                                event_set, target_event, Intval_max)
                    events_clocksstep.append(clock_all)
                    target_list.append(clk_lbcur)
                    # 获取目标事件的上一个事件
                    if lb_idx > 0:
                        prev_clk = data_clks[lb_idx - 1]  # 上一个事件的时间戳
                        step_list.append(prev_clk)
                
                lb_idx += 1
            #print(prev_clk)
            print(f"clk_min:{clk_min}")
            for clk_cur in np.arange(0, clk_max, T_reso):
                # print(clk_cur)
                clock_all = Encode_curclk(data_clks, data_lbs, \
                            clk_cur, event_set, target_event, Intval_max)
                events_clocksdur.append(clock_all)
                if abs(clk_cur-prev_clk)<0.01 and lock==False:
                    dur_list.append(num)#认为找到了上一个事件的发生的时间
                    lock=True
                num+=1
        lock=False
                        
    return np.concatenate(events_clocksdur, 0), \
        np.concatenate(events_clocksstep, 0),dur_list,step_list,target_list,length_dur,length_step
        
                        
# 对数据进行Z-标准化            
def Normalize_data(X_train, X_val, X_test):
    Ntrain = X_train.shape[0]
    Nval = X_val.shape[0]
    
    ss = StandardScaler(with_mean = True , with_std = True)
    
    Xall = torch.cat((X_train, X_val, X_test),0).reshape([-1,X_val.shape[-1]])
    Xall = ss.fit_transform(Xall).astype(np.float32)
    mean_val = ss.mean_
    std_val = np.sqrt(ss.var_)
    Xall = Xall.reshape([-1,1,X_val.shape[-1]])
    
    X_train, X_val, X_test = Xall[:Ntrain,:,:], Xall[Ntrain:Ntrain+Nval,:,:],\
        Xall[Ntrain+Nval:,:,:]
    
    return X_train, X_val, X_test, mean_val, std_val
    

# 扩展数据维度以用于模型输入
def Expand_dim(X_traindur, X_trainstep, X_valdur, X_valstep, \
               X_testdur, X_teststep):
    X_traindur, X_trainstep = torch.Tensor(X_traindur), torch.Tensor(X_trainstep)
    X_traindur, X_trainstep = X_traindur.unsqueeze(1), X_trainstep.unsqueeze(1)
    X_valdur, X_valstep = torch.Tensor(X_valdur), torch.Tensor(X_valstep)
    X_valdur, X_valstep = X_valdur.unsqueeze(1), X_valstep.unsqueeze(1)
    X_testdur, X_teststep = torch.Tensor(X_testdur), torch.Tensor(X_teststep)
    X_testdur, X_teststep = X_testdur.unsqueeze(1), X_teststep.unsqueeze(1)

    return X_traindur, X_trainstep, X_valdur, X_valstep, \
                   X_testdur, X_teststep


# 将事件序列转换为时钟信号用于训练、验证和测试数据集
def Find_targetall(target_event, data_train, data_val, data_test, event_set, \
                        Integration_maxdur, Integration_reso, T_max):
    X_traindur, X_trainstep,train_list_dur,train_list_step,train_list_target,length_dur_train,length_step_train = Find_target(target_event, data_train, event_set, \
                            Integration_maxdur, Integration_reso, T_max)
    X_valdur, X_valstep,val_list_dur,val_list_step,val_list_target,length_dur_val,length_step_val = Find_target(target_event, data_val, event_set, \
                            Integration_maxdur, Integration_reso, T_max)
    X_testdur, X_teststep,test_list_dur,test_list_step,test_list_target,length_dur_test,length_step_test = Find_target(target_event, data_test, event_set, \
                            Integration_maxdur, Integration_reso, T_max)
    print(f"target中{X_teststep}")
    return X_traindur, X_trainstep, X_valdur, X_valstep, X_testdur, X_teststep,train_list_dur,train_list_step,val_list_dur,val_list_step,test_list_dur,test_list_step,test_list_target,length_dur_test,length_step_test

'''
# Get path to load data and save result
def Get_path(dataset_name):
    if dataset_name == "linkedin":
        load = folder_path + "linkedin/"
        save = "./Results/linkedin/"
    elif dataset_name == "mimic":
        load = folder_path + "mimic/"
        save = "./Results/mimic/"
    elif dataset_name == "stack_overflow":
        load = folder_path + "stack_overflow/"
        save = "./Results/stack_overflow/"
        
    if not os.path.exists(save):
        os.makedirs(save)
    return load, save
'''
def save_nll_to_csv(nll_values, csv_path):
    # 确保目录存在
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # 打开 CSV 文件写入数据
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # 写入每个 NLL 值
        for value in nll_values:
            writer.writerow([value.item()])  # 如果是 PyTorch 张量，需用 .item() 转为 Python 标量
    
    print(f"NLL values saved to {csv_path}")
# 获取数据路径和保存结果
def Get_path(dataset_name):

    if dataset_name == "corohheart":
        load_train = "corohheart_dataset_test.csv"
        load_test="corohheart_dataset_test.csv"
        save = "./Results/"

    if dataset_name == "sepsis":
        load_train = "sepsis_dataset_test.csv"
        load_test="sepsis_dataset_test.csv"
        save = "./Results/"

    if dataset_name == "stroke":
        load_train = "stroke_dataset_train.csv"
        load_test="stroke_dataset_test.csv"
        save = "./Results/"

    if dataset_name == "test":
        load_train = "test.csv"
        load_test="test.csv"
        save = "Results/linkedin/"
    
    if not os.path.exists(save):
        os.makedirs(save)
    return load_train,load_test, save

'''
if __name__ == "__main__":
    load_train,load_test, save = Get_path("test")
    data_train = csv_to_dict(load_train)
    print(data_train[1])
    event_set, Interval_max, T_max = Filter_set(data_train)
    print(event_set)
    print(Interval_max)
    print(T_max)
'''
