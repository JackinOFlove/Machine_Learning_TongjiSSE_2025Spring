# 该文件实现了 CLNN 论文的复现工作
# python Re_Clnn_main.py
# 数据集直接使用在 Mimic_Dataprocess 文件夹中处理得到的 .csv即可
# 该篇论文的框架不全，复现工作比较困难
# 我们添加了数据集输入，然后跑通我们的三个数据集
# 我们在 Order_learn.py 中添加了规则的学习方法，并添加了规则的保存 Learn_rules.txt
# 我们在 data_utils_mask.py 中添加了数据集的输入，并且按照论文中的处理方法，添加了完整的数据集的预处理
# 我们在 Intensity_utils.py 中添加了事件强度的计算，并添加了事件强度的保存
# 我们在调试的时候输出 data.txt 文件和 output.txt 文件，用于调试和查看训练过程
# 我们添加了指标计算，包括 MAE\RMSE\NLL

import torch
import torch.nn as nn
import pandas as pd
from data_utils_mask import * 
import matplotlib.pyplot as plt
from scipy.special import comb, perm
from Order_learn import *
from Intensity_utils import *
import time 
import os  # 添加在文件开头
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.cuda.amp import GradScaler, autocast
import gc
#from memory_profiler import profile

#@profile
def main(dataset_name, Integration_maxdur, Integration_reso, Model_num, \
         learning_rate, Max_epoch):
    print("----- Start training {} Dataset -----".format(dataset_name))
    #pathfile, result_path = Get_path(dataset_name)
    pathfile_train,pathfile_test, result_path = Get_path(dataset_name)
    #data_train, data_val, data_test = Load_file(pathfile)
    data_train = csv_to_dict(pathfile_train)
    data_val = csv_to_dict(pathfile_train)
    data_test = csv_to_dict(pathfile_test)
    event_set, Interval_max, T_max = Filter_set(data_train)  #得到事件类型集合、事件流中的最大时间戳、最大时间戳*1.5
    LLs, N_targets, Dur_targets, LLs_val, LLs_test = {}, {}, {}, {}, {}
    with open('data.txt', 'w') as f:
        for key, events in data_train.items():
            f.write(f'ID: {key}\n')
            for event in events:
                t, k = event
                f.write(f'\tTime: {t}, Event: {k}\n')
            f.write('\n')
    # 检查是否有 GPU
    print(event_set)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 目标事件为Middle_to_Sever
    target_event="Middle_to_Sever"

    print("----- Start training target event - {} -----".format(target_event))
    
    X_traindur, X_trainstep, X_valdur, X_valstep, X_testdur, X_teststep,train_list_dur,train_list_step,val_list_dur,val_list_step,test_list_dur,test_list_step,test_list_target,length_dur_test,length_step_test = \
    Find_targetall(target_event, data_train, data_val, data_test, event_set, \
    Integration_maxdur, Integration_reso, T_max) #找到时钟信号，(按时间步长生成dur/按目标事件生成step)
    print(f"target后{X_teststep}")
    X_traindur, X_trainstep, X_valdur, X_valstep, X_testdur, X_teststep = \
    Expand_dim(X_traindur, X_trainstep, X_valdur, X_valstep, X_testdur, X_teststep)#用于适配深度学习模型
    print(f"dim后{X_teststep}")
    print(f"dim后{X_testdur}")
    print("适配深度学习模型")
    print(event_set)
    model = Multi_model(Model_num, len(event_set), 1)  #多子模型集成

    optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)
    Nsamples_dur, Nsamples_step = len(X_traindur), len(X_trainstep)
    print(Nsamples_dur)
    print(Nsamples_step)
    num_batch = min(128, Nsamples_step)  #训练时将训练集分为num_batch份
    batch_sizedur = int(Nsamples_dur / num_batch)
    batch_sizestep = max(int(Nsamples_step / num_batch), 1)
    print(f"batch_sizestep   {batch_sizestep}")
    Loss_val_best = 1e9
    print("设置好训练参数")
    
    # 设置事件名称 - 将event_set转换为列表传入
    for sub_model in model.model_layers:
        sub_model.set_event_names(list(event_set))
    
    # 创建保存模型的目录
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    
    scaler = GradScaler()
    
    for iternum in range(Max_epoch):
        
        for batch_idx in np.arange(0, num_batch, 1):
            # select batch data
            indices_dur = np.arange(batch_idx * batch_sizedur, \
                                (batch_idx + 1) * batch_sizedur, 1)
            indices_step = np.arange(batch_idx * batch_sizestep, \
                                (batch_idx + 1) * batch_sizestep, 1)
            
            Xbatch_dur = X_traindur[indices_dur,]
            Xbatch_step = X_trainstep[indices_step,]
            print(f"Batch {batch_idx}: Xbatch_step size: {Xbatch_step.size()}")
            # traing batch data
            Losstrain_batch, Lossll_batch, Lossconst_batch, LossIG_batch, _ , _ = optimize_log_likelihood(Xbatch_dur, Xbatch_step, model, Integration_reso, device, "train")
            optimizer.zero_grad()
            Losstrain_batch.backward(retain_graph = True)
            optimizer.step()
            # projected gradient for logical constraints
            with torch.no_grad():
                model.Clamp_proj()
                model.Clamp_alpha()
                # 创建一个列表来存储每次迭代的 intensity
                intensity_list = []
                step_intensity_list=[]
                Nsamples_dur_test, Nsamples_step_test = len(X_testdur), len(X_teststep)
                print(f"Nsample_dur_test{Nsamples_dur_test}")
                print(f"Nsamples_step_test{Nsamples_step_test}")#有多少个事件序列，这里就是多少
                num_batch_test = min(128, Nsamples_step_test)  #训练时将训练集分为2份
                batch_sizedur_test = int(Nsamples_dur_test / num_batch_test)  #平均每一个事件分到多少的dur
                batch_sizestep_test = max(int(Nsamples_step_test / num_batch_test), 1)  #一个事件是一个步长
                print(num_batch_test)
                for batch_idx_test in np.arange(0, num_batch_test, 1):
                    # select batch data
                    print(batch_idx_test*batch_sizedur_test)
                
                    indices_dur_test = np.arange(batch_idx_test * batch_sizedur_test, \
                                        (batch_idx_test + 1) * batch_sizedur_test, 1)
                    indices_step_test = np.arange(batch_idx_test * batch_sizestep_test, \
                                        (batch_idx_test + 1) * batch_sizestep_test, 1)
                    
                    Xbatch_dur_test = X_testdur[indices_dur_test,]
                    Xbatch_step_test = X_teststep[indices_step_test,]
                    print(f"索引{indices_dur_test}")
                    print(indices_step_test)
                    print(batch_idx_test)
                    print(X_teststep)
                    print(Xbatch_step_test)
                    # traing batch data
                    Losstrain_batch, Lossll_batch, Lossconst_batch,\
                        LossIG_batch,intensity,step_intensity = optimize_log_likelihood(Xbatch_dur_test, Xbatch_step_test, \
                                    model, Integration_reso, device, "val")#下面是梯度优化
                    # 将 intensity 添加到列表中
                    intensity_list.append(intensity)
                    step_intensity_list.append(step_intensity)
                # 将列表中的所有 intensity 连接成一个大张量（或数组）
                intensity = torch.cat(intensity_list)
                step_intensity=torch.cat(step_intensity_list)
                with open('stroke-learned_rules.txt', 'a') as f:
                    # 通过 print 将内容同时输出到文件和控制台
                    def print_to_file(*args, **kwargs):
                        print(*args, **kwargs)  # 打印到控制台
                        print(*args, file=f, **kwargs)  # 写入文件
                    print_to_file(intensity)
                    print_to_file(step_intensity)
                    print_to_file("测试集的条件强度长什么样呢")  
                    print_to_file(intensity.shape)    
                    target_intensity=[]
                    for num in test_list_dur:
                        target_intensity.append(intensity[num])
                    print_to_file(target_intensity)
                    times=[]
                    for index,num in enumerate(target_intensity):
                        print(1/num.item())
                        print(test_list_step[index])
                        time=1/num.item()+test_list_step[index]
                        times.append(time)
                    print_to_file(times)
                    # 取较短的长度
                    min_len = min(len(test_list_target), len(times))

                    # 计算 MAE
                    mae = mean_absolute_error(test_list_target[:min_len], times[:min_len])
                    print_to_file(f"MAE: {mae}")

                    # 计算 RMSE
                    rmse = np.sqrt(mean_squared_error(test_list_target[:min_len], times[:min_len]))
                    print_to_file(f"RMSE: {rmse}")
                    print_to_file(f"current_formula = {model.get_readable_formula()}")
                    nlls = compute_nll_per_sequence(step_intensity, intensity, length_step_test,length_dur_test, Integrate_reso)
                    print_to_file(f"nll:{nlls}")
                    # 调用保存函数
                    save_nll_to_csv(nlls, "CLNN-Stroke/nll_values.csv")

            print("一个batch")
        
        # Evaluate model using validation data every 5 iterations
        if iternum % 5 == 0:
            with torch.no_grad():
                Nsamples_dur_val, Nsamples_step_val = len(X_valdur), len(X_valstep)
                print(f"Nsample_dur_val{Nsamples_dur_val}")
                print(f"Nsamples_step_val{Nsamples_step_val}")#有多少个事件序列，这里就是多少
                num_batch_val = min(128, Nsamples_step_val)  #训练时将训练集分为2份
                batch_sizedur_val = int(Nsamples_dur_val / num_batch_val)  #平均每一个事件分到多少的dur
                batch_sizestep_val = max(int(Nsamples_step_val / num_batch_val), 1)  #一个事件是一个步长
                print(num_batch_val)
                for batch_idx_val in np.arange(0, num_batch_val, 1):
                    # select batch data
                    indices_dur_val = np.arange(batch_idx_val * batch_sizedur_val, \
                                        (batch_idx_val + 1) * batch_sizedur_val, 1)
                    indices_step_val = np.arange(batch_idx_val * batch_sizestep_val, \
                                        (batch_idx_val + 1) * batch_sizestep_val, 1)
                    Xbatch_dur_val = X_valdur[indices_dur_val,]
                    Xbatch_step_val = X_valstep[indices_step_val,]
                    print(f"Batch {batch_idx_val}: Xbatch_step_val size: {Xbatch_step_val.size()}")
                    # traing batch data
                    Lossval_iter, _, _, _,intensity,step_intensity = optimize_log_likelihood(Xbatch_dur_val, Xbatch_step_val, model, Integration_reso, device, "val")
                current_formula = model.get_readable_formula()
                print(f"\nIteration {iternum}:")
                print(f"Formula: {current_formula}")
                print(f"Validation Loss: {Lossval_iter.item():.4f}")
                if Lossval_iter.cpu().detach().numpy() < Loss_val_best:
                    print("*** 发现更好的公式! ***")
                    best_formula = current_formula
                    Loss_val_best = Lossval_iter.cpu().detach().numpy() - 0
                    torch.save(model.state_dict(), result_path+"event.pkl")
        
    event_list = sorted(list(event_set))
    bestmodel = Multi_model(Model_num, len(event_list), 1)
    bestmodel.load_state_dict(torch.load(result_path + "event.pkl"))
    # 为bestmodel设置事件名称
    
    for sub_model in bestmodel.model_layers:
        sub_model.set_event_names(event_list)
    # 创建一个列表来存储每次迭代的 intensity
    intensity_list = []
    step_intensity_list=[]
    Nsamples_dur_test, Nsamples_step_test = len(X_testdur), len(X_teststep)
    print(f"Nsample_dur_test{Nsamples_dur_test}")
    print(f"Nsamples_step_test{Nsamples_step_test}")#有多少个事件序列，这里就是多少
    num_batch_test = min(128, Nsamples_step_test)  #训练时将训练集分为2份
    batch_sizedur_test = int(Nsamples_dur_test / num_batch_test)  #平均每一个事件分到多少的dur
    batch_sizestep_test = max(int(Nsamples_step_test / num_batch_test), 1)  #一个事件是一个步长
    print(num_batch_test)
    for batch_idx_test in np.arange(0, num_batch_test, 1):
        # select batch data
        print(batch_idx_test*batch_sizedur_test)

        indices_dur_test = np.arange(batch_idx_test * batch_sizedur_test, \
                            (batch_idx_test + 1) * batch_sizedur_test, 1)
        indices_step_test = np.arange(batch_idx_test * batch_sizestep_test, \
                            (batch_idx_test + 1) * batch_sizestep_test, 1)
        
        Xbatch_dur_test = X_testdur[indices_dur_test,]
        Xbatch_step_test = X_teststep[indices_step_test,]
        print(f"索引{indices_dur_test}")
        print(indices_step_test)
        print(batch_idx_test)
        print(X_teststep)
        print(Xbatch_step_test)
        # 将 intensity 添加到列表中
        intensity_list.append(intensity)
        step_intensity_list.append(step_intensity)
    # 将列表中的所有 intensity 连接成一个大张量（或数组）
    intensity = torch.cat(intensity_list)
    step_intensity=torch.cat(step_intensity_list)
    with open('stroke-learned_rules.txt', 'a') as f:
        # 通过 print 将内容同时输出到文件和控制台
        def print_to_file(*args, **kwargs):
            print(*args, **kwargs)  # 打印到控制台
            print(*args, file=f, **kwargs)  # 写入文件
        print_to_file(intensity)
        print_to_file(step_intensity)
        print_to_file("测试集的条件强度长什么样呢")  
        print_to_file(intensity.shape)    
        target_intensity=[]
        for num in test_list_dur:
            target_intensity.append(intensity[num])
        print_to_file(target_intensity)
        times=[]
        for index,num in enumerate(target_intensity):
            print(1/num.item())
            print(test_list_step[index])
            time=1/num.item()+test_list_step[index]
            times.append(time)
        print_to_file(times)
        # 取较短的长度
        min_len = min(len(test_list_target), len(times))

        # 计算 MAE
        mae = mean_absolute_error(test_list_target[:min_len], times[:min_len])
        print_to_file(f"MAE: {mae}")

        # 计算 RMSE
        rmse = np.sqrt(mean_squared_error(test_list_target[:min_len], times[:min_len]))
        print_to_file(f"RMSE: {rmse}")
        print_to_file(f"current_formula = {model.get_readable_formula()}")
        nlls = compute_nll_per_sequence(step_intensity, intensity, length_step_test,length_dur_test, Integrate_reso)
        print_to_file(f"nll:{nlls}")
        # 调用保存函数
        save_nll_to_csv(nlls, "CLNN-Stroke/nll_values_final.csv")
        
        print("----- Finish training Dataset {} -----".format(dataset_name))


if __name__ == '__main__':
    dataset_name = "stroke"
    Integrate_maxdur, Integrate_reso, Model_num = 10, 0.01, 3
    learn_rate, Max_epoch = 0.001, 2
    main(dataset_name, Integrate_maxdur, Integrate_reso, Model_num, \
         learn_rate, Max_epoch)

