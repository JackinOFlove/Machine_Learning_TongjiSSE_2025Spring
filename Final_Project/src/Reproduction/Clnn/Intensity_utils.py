# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

# Compute log likelihood for event streams
def optimize_log_likelihood(X_dur, X_step, model, resolution, device, eval_type):
    cur_intensity = Intensity_2(model, X_dur) #条件强度
    step_intensity = Intensity_2(model,X_step)
    print(f"cur_itensity{cur_intensity}")
    print(f"step_itensity{step_intensity}")
    print(X_dur)
    print(X_step)
    # Training loss = -LL + constraint loss
    if eval_type == "train":
       loss_step_ll = -log_likelihood(X_dur, X_step, model, resolution,device) #计算模型预测输出的对数似然值
       loss_step_const = model.Const_loss() #计算模型的约束损失
       loss_step_IG = model.IG_loss() #计算模型的信息增益损
       loss_all = loss_step_ll + loss_step_const + loss_step_IG
    else:
    # Evaluation loss = -LL
       loss_step = -log_likelihood(X_dur, X_step, model, resolution,device)
       loss_all = loss_step
       loss_step_ll = loss_step
       loss_step_const = loss_step
       loss_step_IG = loss_step
    return loss_all, loss_step_ll, loss_step_const, loss_step_IG,cur_intensity,step_intensity

# Log likelihood = LL for no target event happening + LL for target event happening
def log_likelihood(X_dur, X_step, model, resolution,device):
    log_likelihood_ret = torch.tensor([0], dtype = torch.float64, device = device)
    intensity_log_sum_ret = Intensity_log_sum(model, X_step)
    log_likelihood_ret += intensity_log_sum_ret
    intensity_integral = Intensity_integral(model, X_dur, resolution)
    log_likelihood_ret -= intensity_integral
        
    return log_likelihood_ret


# LL for time stamps where target event happens
def Intensity_log_sum(model, X_step):
    print(f"\nsum中的X_step:{X_step.shape}")
    cur_intensity = Intensity_1(model, X_step)
    log_sum = torch.sum(torch.log(cur_intensity))
    
    return log_sum


# Integral LL
def Intensity_integral(model, X_dur, resolution):
    print(f"\nintegral中的X_dur:{X_dur.shape}\n")
    cur_intensity = Intensity_2(model, X_dur)
    integral = torch.sum(cur_intensity * resolution)
    
    return integral

# Intensity rate
def Intensity_2(model, X):
    print(f"X_step的形状:{X.shape}")
    model_out = model(X).view([-1])
    print(f"model_out的形状:{model_out.shape}")
    return model_out

def Intensity_1(model, X):
    print(f"X_step的形状:{X.shape}")
    model_out = model(X).view([-1])
    print(f"model_out的形状:{model_out.shape}")
    return model_out

def compute_nll_per_sequence(event_intensities, grid_intensities, event_counts, grid_counts, delta_t):
    """
    计算每个序列的负对数似然值

    参数:
    - event_intensities: 所有事件的条件强度拼接成的 PyTorch 张量
    - grid_intensities: 所有时间网格的条件强度拼接成的 PyTorch 张量
    - event_counts: 每个序列中事件的数量
    - grid_counts: 每个序列中时间点的数量
    - delta_t: 时间网格中两个点之间的间隔大小
    
    返回:
    - nlls: 每个序列的负对数似然值
    """
    nlls = []  # 用于存储每个序列的 NLL
    event_idx = 0  # 事件强度的起始索引
    grid_idx = 0   # 时间网格强度的起始索引

    # 遍历每个序列
    for event_count, grid_count in zip(event_counts, grid_counts):
        # 取出当前序列的事件强度和时间网格强度
        current_event_intensities = event_intensities[event_idx:event_idx + event_count]
        current_grid_intensities = grid_intensities[grid_idx:grid_idx + grid_count]
        print(current_event_intensities)
        print(current_grid_intensities)
        # 更新索引
        event_idx += event_count
        grid_idx += grid_count

        # 第一步：计算事件部分 -∑log λ*(t_i)
        log_likelihood_part = -torch.sum(torch.log(current_event_intensities))

        # 第二步：使用梯形法计算积分部分
        integral_part = delta_t * 0.5 * torch.sum(current_grid_intensities[:-1] + current_grid_intensities[1:])

        # 计算负对数似然并存储
        nll = log_likelihood_part + integral_part
        nlls.append(nll)

    # 返回每个序列的负对数似然值
    return torch.tensor(nlls)