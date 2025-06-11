# 该文件实现了我们的混合事件点过程模型的实现
# Modules.py 文件中实现了模型的所有功能，配合 Main.py 文件可以实现模型的训练、评估、规则获取、指标计算等功能
# 我们的新模型的详细介绍请见报告内容

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pandas as pd
import re
import random
import math
import copy
from tqdm import tqdm
from itertools import product
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import time

# 加载数据
def read_data(file_path, scale=True, outliers=0.0, target_varibles=None):
    """
    读取CSV文件，处理数据，处理异常值，归一化值，将变量名映射为整数。

    参数:
    file_path (str): CSV文件路径，包含数据。
    scale (bool, default=True): 如果为True，则对每个'k'组中的'v'值进行归一化。
    outliers (float, default=0.0): 要剪辑的异常值的分数（0-1）。
    target_varibles (str or None, default=None): 如果提供，则将此变量名移动到变量列表的第一个位置。
    返回:
    data (list of lists): 嵌套列表，每个内列表对应一个由'id'标识的组。
                          每个组中的行包含[时间('t'), 变量索引('k'), 值('v')]。
    var_name_dict (dict): 字典，将变量名映射为整数索引。
    """
    df = pd.read_csv(file_path, header=0, sep=",")
    var_name_list = sorted(set(df['k'].unique()))
    if target_varibles:
        if target_varibles in var_name_list:
            var_name_list.remove(target_varibles)
            var_name_list.insert(0, target_varibles)
    var_name_dict = {val: idx for idx, val in enumerate(var_name_list)}
    unique_k = df['k'].unique()
    for k_value in unique_k:
        subset = df[df['k'] == k_value]
        if k_value.endswith('High'):
            q_upper = subset['v'].quantile(1-outliers)
            df.loc[subset.index, 'v'] = subset['v'].clip(upper=q_upper)
        elif k_value.endswith('Low'):
            q_lower = subset['v'].quantile(outliers)
            df.loc[subset.index, 'v'] = subset['v'].clip(lower=q_lower)
    if scale:
        df['v_normalized'] = df.groupby('k')['v'].transform(lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.0)
        df['v'] = df['v_normalized']
        df = df.drop(columns=['v_normalized'])
    df['k'] = df['k'].map(var_name_dict)
    df = df.sort_values(by=['id', 't', 'k']).reset_index(drop=True)
    data = [
        [[round(row.t,6), int(row.k), round(row.v,6)] for _, row in group.iterrows()]
        for _, group in df.groupby('id')
    ]
    return data, var_name_dict


class RuleSet():
    def __init__(self, event_data, var_name_dict):
        """
        用提供的event数据和变量名字典初始化RuleSet对象。

        参数:
        event_data (list): 一个事件列表，每个事件包含时间、变量索引和值。
        var_name_dict (dict): 一个字典，将变量名映射为唯一标识符。
        """
        self.event_data = event_data
        self.var_name_dict = var_name_dict
        self.rule_name_dict = dict()
        self.rule_name_set = set()
        self.rule_var_ids = set()
        self.var_count = 0
        self.rule_event_data = [[] for _ in range(len(event_data))]
        self.time_tolerance = 0.1
    
    def add_rule(self, cause):
        """
        将规则添加到RuleSet中，处理其cause表达式为RPN并更新事件数据。

        参数:
        cause (str): 规则的cause表达式，可以是一个逻辑组合的变量。
        """
        cause_rpn = self.infix_to_rpn(self.replace_variables(cause))
        cause_var_ids = set([int(token) for token in cause_rpn if token.isdigit()])
        cause_name = self.rpn_to_rule(cause_rpn)
        if cause_name not in self.rule_name_dict:
            self.rule_name_dict[cause_name] = self.var_count
            self.rule_var_ids.update(cause_var_ids)
            cause_rpn_str = " ".join(cause_rpn)
            self.rule_name_set.add(cause_rpn_str)
            for ind, events in enumerate(self.event_data):
                cause_value, cause_time = self.get_new_var(events, cause_rpn)
                self.rule_event_data[ind].extend([t, self.var_count, v] for t, v in zip(cause_time, cause_value))
            self.var_count += 1
            
    def replace_variables(self, rule_expression):
        """
        将规则表达式中的变量名替换为对应的数字ID。

        参数:
        rule_expression (str): 包含变量名的规则表达式。
        返回:
        str: 变量名替换为ID的规则表达式。
        """
        var_name_dict = self.var_name_dict
        def replace_var(match):
            var_name = match.group(0)
            return str(var_name_dict.get(var_name, var_name)) 
        pattern = r'\b(' + '|'.join(re.escape(key) for key in var_name_dict.keys()) + r')\b'
        result = re.sub(pattern, replace_var, rule_expression)
        return result
    
    def infix_to_rpn(self, expression):
        """
        将中缀表达式（例如“A equal B”）转换为逆波兰表达式（RPN）。

        参数:
        expression (str): 要转换的中缀表达式。
        返回:
        list: 一个表示表达式RPN形式的列表。
        """
        precedence = {'equal': 1, 'before': 1, 'and': 1}
        output = []
        operators = []
        tokens = expression.replace('(', ' ( ').replace(')', ' ) ').split()
        last_token = None
        open_parentheses = 0
        for token in tokens:
            if token.lower() in precedence:
                if last_token is None or last_token.lower() in precedence or last_token == '(':
                    raise ValueError(f"Invalid expression: operator {token} in wrong position.")
                while (operators and operators[-1] in precedence and
                    precedence[operators[-1]] >= precedence[token.lower()]):
                    output.append(operators.pop())
                operators.append(token.lower())
            elif token == '(':
                open_parentheses += 1
                operators.append(token)
            elif token == ')':
                open_parentheses -= 1
                if open_parentheses < 0:
                    raise ValueError("Invalid expression: mismatched parentheses.")
                while operators and operators[-1] != '(':
                    output.append(operators.pop())
                operators.pop()
            elif token.isdigit():
                if last_token and last_token.isdigit():
                    raise ValueError("Invalid expression: two consecutive operands.")
                output.append(token)
            else:
                raise ValueError(f'Invalid token {token} in expression.') 
            last_token = token
        if open_parentheses != 0:
            raise ValueError("Invalid expression: mismatched parentheses.")
        if last_token and last_token.lower() in precedence:
            raise ValueError("Invalid expression: cannot end with an operator.")
        while operators:
            op = operators.pop()
            if op == '(' or op == ')':
                raise ValueError("Invalid expression: mismatched parentheses.")
            output.append(op)
        return output
    
    def rpn_to_rule(self, rpn_expression):
        """
        将逆波兰表达式（RPN）转换为人类可读的规则字符串。

        参数:
        rpn_expression (list): 要转换的RPN表达式。
        返回:
        str: 人类可读的规则表达式。
        """
        reverse_var_dict = {str(v): k for k, v in self.var_name_dict.items()}
        stack = []
        for token in rpn_expression:
            if token.isdigit():
                var_name = reverse_var_dict[token]
                stack.append(var_name)
            else:
                if token.lower() in ['before', 'equal', 'and']:
                    operand2 = stack.pop()
                    operand1 = stack.pop()
                    sub_expression = f"({operand1} {token} {operand2})"
                    stack.append(sub_expression)
                else:
                    raise ValueError(f"Invalid token {token} in RPN expression.")
        final_expression = stack.pop()
        if final_expression[0] == "(" and final_expression[-1] == ")":
            return final_expression[1:-1]
        return final_expression
    
    def get_new_var(self, events, rpn_expression):
        """
        评估数据集中每个事件的RPN表达式并返回结果。

        参数:
        events (list): 一个事件列表，每个事件包含时间、变量索引和值。
        rpn_expression (list): 要评估的RPN表达式。
        返回:
        tuple: 一个包含评估变量值和相应时间的元组。
        """
        events_dict = {}
        for var_name_id in self.var_name_dict.values():
            events_dict[str(var_name_id)] = {
                'time': [],
                'value': []
            }
        for t_j, k_j, v_j in events:
            events_dict[str(int(k_j))]['time'].append(t_j)
            events_dict[str(int(k_j))]['value'].append(1)
        def get_variable_info(variables, var_id):
            return variables[var_id]['value'], variables[var_id]['time']
        def before(var1, var2):
            value1, time1 = var1
            value2, time2 = var2
            result_value = []
            result_time = []
            i, j = 0, 0
            while i < len(time1) and j < len(time2):
                if time1[i] < time2[j] - self.time_tolerance:
                    result_value.append(1)
                    result_time.append(max(time1[i], time2[j]))
                    i += 1
                    j += 1
                else:
                    j += 1
            return result_value, result_time
        def equal(var1, var2):
            value1, time1 = var1
            value2, time2 = var2
            result_value = []
            result_time = []
            i, j = 0, 0
            while i < len(time1) and j < len(time2):
                if abs(time1[i] - time2[j]) <= self.time_tolerance:
                    if value1[i] * value2[j] != 0:
                        result_value.append(1)
                        result_time.append(max(time1[i], time2[j]))
                    i += 1
                    j += 1
                elif time1[i] < time2[j]:
                    i += 1
                else:
                    j += 1
            return result_value, result_time
        def and_op(var1, var2): 
            value1, time1 = var1
            value2, time2 = var2
            result_value = []
            result_time = []
            i, j = 0, 0
            while i < len(time1) and j < len(time2):
                if abs(time1[i] - time2[j]) > self.time_tolerance:
                    result_value.append(1)
                    result_time.append(max(time1[i], time2[j]))
                    i += 1
                    j += 1
                else:
                    if i+1 < len(time1) and j+1 < len(time2):
                        if time1[i+1] < time2[j+1]:
                            i += 1
                        else:
                            j += 1
                    elif i+1 < len(time1):
                        i += 1
                    else:
                        j += 1
            return result_value, result_time
        stack = []
        for token in rpn_expression:
            if token == 'before':
                var2 = stack.pop()
                var1 = stack.pop()
                stack.append(before(var1, var2))
            elif token == 'equal':
                var2 = stack.pop()
                var1 = stack.pop()
                stack.append(equal(var1, var2))
            elif token == 'and':
                var2 = stack.pop()
                var1 = stack.pop()
                stack.append(and_op(var1, var2))
            else:
                stack.append(get_variable_info(events_dict, token))
        result_value, result_time = stack.pop()
        return result_value, result_time
    

class RuleBasedTPP(nn.Module):
    def __init__(self, var_name_dict, rule_name_dict, rule_var_ids, device="cpu"):
        """
        初始化Rule-Based Temporal Point Process (RTPP)模型。
        
        参数:
        - var_name_dict (dict): 字典映射变量名到唯一标识符。
        - rule_name_dict (dict): 字典映射规则名到唯一标识符。
        - rule_var_ids (set): 规则表达式中使用的变量ID集合。
        - device (str): 模型执行的设备（例如'cpu'或'cuda'）。
        """
        super(RuleBasedTPP, self).__init__()
        self.var_name_dict = var_name_dict # 字典映射变量名到唯一标识符
        self.rule_name_dict = rule_name_dict # 字典映射规则名到唯一标识符
        self.rule_var_ids = rule_var_ids # 规则表达式中使用的变量ID集合
        self.device = torch.device(device) # 计算设备
        self.K = len(var_name_dict) # 事件类型数量
        self.M = len(rule_name_dict) # 规则事件数量
        self.mu = nn.Parameter(torch.tensor(0.1, dtype=torch.float32, device=self.device))                # 基础强度参数 μ
        self.beta = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device=self.device))              # Softplus参数 β
        self.rule_weights = nn.Parameter(torch.rand(self.M, dtype=torch.float32, device=self.device))     # Rule-guided strength for cause event type m
        self.meas_weights = nn.Parameter(torch.rand(self.K, dtype=torch.float32, device=self.device))     # Measurement-driven strength for each event type k_x
        self.meas_weights_mask = torch.zeros(self.K, dtype=torch.float32, device=self.device, requires_grad=False)  # Mask for measurement weights
        for var_id in self.rule_var_ids:
            self.meas_weights_mask[var_id] = 1.0
        self.meas_weights_mask[0] = 1.0
        
    def forward(self, event_times, event_types, event_meass, rule_times, rule_types, rule_meass):
        """
        执行前向传播，计算损失（负对数似然、时间损失和类型损失）。
        
        参数:
        - event_times (Tensor): 事件的时间。
        - event_types (Tensor): 事件的类型。
        - event_meass (Tensor): 事件的测量。
        - rule_times (Tensor): 规则事件的时间。
        - rule_types (Tensor): 规则事件的类型。
        - rule_meass (Tensor): 规则事件的测量。
        返回:
        - nll + type_loss + time_loss (Tensor): 总损失函数（负对数似然、类型损失和时间损失）。
        """
        # 负对数似然（NLL）损失
        given_times = event_times[event_types == 0]
        lambda_values = self.intensity(given_times=given_times, 
                                       event_times=event_times, event_types=event_types, event_meass=event_meass, 
                                       rule_times=rule_times, rule_types=rule_types, rule_meass=rule_meass)
        log_likelihood = torch.sum(torch.log(lambda_values))
        T_max = torch.max(given_times)
        T_min = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        num_samples = 20
        t_values = torch.linspace(T_min, T_max, num_samples, device=self.device)
        integral_values = self.intensity(given_times=t_values, 
                                         event_times=event_times, event_types=event_types, event_meass=event_meass, 
                                         rule_times=rule_times, rule_types=rule_types, rule_meass=rule_meass)
        integral = torch.trapz(integral_values, t_values)
        nll = - (log_likelihood - integral)
        return nll

    def intensity(self, given_times, event_times, event_types, event_meass, rule_times, rule_types, rule_meass):
        """
        计算每个事件类型在给定时间点的强度函数（λ），考虑事件的历史。

        参数:
        - given_times (Tensor): 在给定时间点评估强度函数的时刻。
        - event_times (Tensor): 历史事件的时间。
        - event_types (Tensor): 历史事件的类型。
        - event_meass (Tensor): 历史事件的测量。
        - rule_times (Tensor): 规则事件的时间。
        - rule_types (Tensor): 规则事件的类型。
        - rule_meass (Tensor): 规则事件的测量。
        返回:
        - total_intensity (Tensor): 在给定时间点计算的每个事件类型的强度。
        """
        # 基础强度组件
        base_intensity = self.mu
        # 规则驱动的强度组件
        rule_intensity = torch.sum(rule_meass * self.time_decay(given_times.view(-1,1)-rule_times) * self.rule_weights[rule_types], dim=1)
        # 数据驱动的强度组件
        meas_intensity = torch.sum(event_meass * self.time_decay(given_times.view(-1,1)-event_times) * self.meas_weights[event_types] * self.meas_weights_mask[event_types], dim=1)
        # 总强度
        #base_intensity = base_intensity.expand_as(given_times)
        sum_intensity =  rule_intensity + meas_intensity + base_intensity#+ meas_intensity#base_intensity + meas_intensity#+ base_intensity#+
        # Softplus函数
        total_intensity = torch.log1p(torch.exp(self.beta * sum_intensity)) / self.beta
        return total_intensity

    def time_decay(self, delta_t):
        """
        使用指数衰减函数计算给定时间差（delta_t）的衰减。

        参数:
        - delta_t (Tensor): 事件之间的时间差。
        返回:
        - decay (Tensor): 基于时间差的衰减值。
        """
        decay = torch.exp(-delta_t)
        decay[delta_t <= 0] = 0.0
        return decay

    def evaluate(self, event_times, event_types, event_meass, rule_times, rule_types, rule_meass, target_name):
        """
        使用负对数似然（NLL）、平均绝对误差（MAE）和均方根误差（RMSE）评估模型性能。
        
        参数:
        - event_times (Tensor): 事件的时间。
        - event_types (Tensor): 事件的类型。
        - event_meass (Tensor): 事件的测量。
        - rule_times (Tensor): 规则事件的时间。
        - rule_types (Tensor): 规则事件的类型。
        - rule_meass (Tensor): 规则事件的测量。
        - target_name (str): 目标变量的名称。
        返回:
        - nll_k (Tensor): 目标变量的负对数似然。
        - mae (Tensor): 预测区间的平均绝对误差。
        - rmse (Tensor): 预测区间的均方根误差。
        """
        target_id = self.var_name_dict[target_name]
        target_indices = (event_types == target_id).nonzero(as_tuple=True)[0]
        lambda_values = self.intensity(given_times=event_times[target_indices], 
                                       event_times=event_times, event_types=event_types, event_meass=event_meass, 
                                       rule_times=rule_times, rule_types=rule_types, rule_meass=rule_meass)
        log_likelihood = torch.sum(torch.log(lambda_values))
        T_max = torch.max(event_times[target_indices])
        T_min = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        num_samples = 20
        t_values = torch.linspace(T_min, T_max, num_samples, device=self.device)
        integral_values = self.intensity(given_times=t_values, 
                                         event_times=event_times, event_types=event_types, event_meass=event_meass, 
                                         rule_times=rule_times, rule_types=rule_types, rule_meass=rule_meass)
        integral = torch.trapz(integral_values, t_values)
        nll_k = - (log_likelihood - integral) # 负对数似然

        target_indices = target_indices[1:] if target_indices[0] == 0 else target_indices
        if len(target_indices) == 0:
            return nll_k, torch.tensor(0.0, dtype=torch.float32, device=self.device), torch.tensor(0.0, dtype=torch.float32, device=self.device)
        prev_indices = target_indices - 1
        target_times = event_times[target_indices]
        prev_times = event_times[prev_indices]
        time_intervals = torch.tensor([self.MC_next_event(prev_time, event_times, event_types, event_meass, rule_times, rule_types, rule_meass) for prev_time in prev_times], device=self.device)
        real_intervals = target_times - prev_times
        interval_diffs = time_intervals - real_intervals
        mae = torch.mean(torch.abs(interval_diffs)) # 平均绝对误差
        mse = torch.mean(torch.square(interval_diffs))
        rmse = mse.sqrt() # 均方根误差
        return nll_k, mae, rmse
    
    def MC_next_event(self, prev_times, event_times, event_types, event_meass, rule_times, rule_types, rule_meass, max_time=6.0, num_samples=2000):
        """
        使用蒙特卡罗方法预测下一个事件的时间。
        
        参数:
        - prev_times (float): 前一个事件的时间。
        - event_times (list): 历史事件发生的时间列表。
        - event_types (list): 历史事件的类型列表。
        - event_meass (list): 历史事件的测量列表。
        - rule_times (list): 规则事件发生的时间列表。
        - rule_types (list): 规则事件的类型列表。
        - rule_meass (list): 规则事件的测量列表。
        - max_time (float): 从上一个事件预测的最大时间范围，默认是10.0。
        - num_samples (int): 生成样本的数量，默认是1000。
        返回:
        - torch.Tensor: 下一个事件的预测时间。
        """
        next_times = []
        for _ in range(num_samples):
            t_current = prev_times
            while True:
                current_intensity = self.intensity(given_times=torch.tensor([t_current], dtype=torch.float32, device=self.device),
                                                   event_times=event_times, event_types=event_types, event_meass=event_meass, 
                                                   rule_times=rule_times, rule_types=rule_types, rule_meass=rule_meass)
                u = torch.rand(1, dtype=torch.float32, device=self.device)
                tau_candidate = -torch.log(u) / current_intensity
                t_candidate = t_current + tau_candidate
                if t_candidate-prev_times > max_time:
                    next_times.append(torch.inf)
                    break
                candidate_intensity = self.intensity(given_times=torch.tensor([t_candidate], dtype=torch.float32, device=self.device),
                                                     event_times=event_times, event_types=event_types, event_meass=event_meass, 
                                                     rule_times=rule_times, rule_types=rule_types, rule_meass=rule_meass)
                if torch.rand(1, dtype=torch.float32, device=self.device) < (candidate_intensity / current_intensity):
                    next_times.append(t_candidate - prev_times)
                    break
                else:
                    t_current = t_candidate
            valid_samples = torch.tensor([t for t in next_times if t < torch.inf])
            if len(valid_samples) == 0:
                return torch.tensor(max_time, dtype=torch.float32, device=self.device)
            return torch.mean(valid_samples)

# 自定义数据集
class EventDataset(Dataset):
    def __init__(self, data, rule_event_data):
        self.data = data
        self.rule_events_data = rule_event_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        events = self.data[idx]
        rule_events = self.rule_events_data[idx]
        return events, rule_events


# 自定义变长序列的collate函数
def collate_fn(batch):
    events, rule_events = zip(*batch)
    return list(events), list(rule_events)


def train_epoch(model, optimizer, train_dataloader, device="cpu"):
    """
    执行单个训练epoch，根据训练数据中的损失更新模型。

    参数:
    - model (nn.Module): 正在训练的模型。
    - optimizer (torch.optim.Optimizer): 用于梯度下降的优化器。
    - train_dataloader (DataLoader): 提供训练数据批次的加载器。
    - device (str): 计算的设备（'cpu'或'cuda'）。
    返回:
    - total_loss (float): 该epoch的累积损失，所有训练批次平均。
    """
    train_size = sum(len(events) for events, _ in train_dataloader)
    model.train()
    total_loss = 0.0
    for events_batch, rule_events_batch in train_dataloader:
        optimizer.zero_grad()
        batch_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        for events, rule_events in zip(events_batch, rule_events_batch):
            event_times, event_types, event_meass = zip(*events)
            event_times = torch.tensor(event_times, dtype=torch.float32, device=device)
            event_types = torch.tensor(event_types, dtype=torch.long, device=device)
            event_meass = torch.tensor(event_meass, dtype=torch.float32, device=device)
            if rule_events == []:
                rule_times = torch.tensor([], dtype=torch.float32, device=device)
                rule_types = torch.tensor([], dtype=torch.long, device=device)
                rule_meass = torch.tensor([], dtype=torch.float32, device=device)
            else:
                rule_times, rule_types, rule_meass = zip(*rule_events)
                rule_times = torch.tensor(rule_times, dtype=torch.float32, device=device)
                rule_types = torch.tensor(rule_types, dtype=torch.long, device=device)
                rule_meass = torch.tensor(rule_meass, dtype=torch.float32, device=device)
            sequence_loss = model.forward(event_times, event_types, event_meass, rule_times, rule_types, rule_meass)
            batch_loss += sequence_loss / len(events_batch)
            total_loss += sequence_loss.item() / train_size
        batch_loss.backward()
        optimizer.step()
    return total_loss


def eval_epoch(model, eval_dataloader, target_name, device="cpu"):
    """
    评估模型在验证或测试数据集上的性能，计算NLL、MAE和RMSE。

    参数:
    - model (nn.Module): 正在评估的模型。
    - eval_dataloader (DataLoader): 提供评估数据批次的加载器。
    - target_name (str): 评估的目标变量名称。
    - device (str): 计算的设备（'cpu'或'cuda'）。
    返回:
    - eval_nll (float): 在评估数据集上的平均负对数似然（NLL）。
    - eval_mae (float): 在评估数据集上的平均绝对误差（MAE）。
    - eval_rmse (float): 在评估数据集上的均方根误差（RMSE）。
    """
    model.eval()
    eval_size = sum(len(events) for events, _ in eval_dataloader)
    with torch.no_grad():
        eval_nll, eval_mae, eval_rmse = 0.0, 0.0, 0.0
        for events_batch, rule_events_batch in eval_dataloader:
            for events, rule_events in zip(events_batch, rule_events_batch):
                event_times, event_types, event_meass = zip(*events)
                event_times = torch.tensor(event_times, dtype=torch.float32, device=device)
                event_types = torch.tensor(event_types, dtype=torch.long, device=device)
                event_meass = torch.tensor(event_meass, dtype=torch.float32, device=device)
                if rule_events == []:
                    rule_times = torch.tensor([], dtype=torch.float32, device=device)
                    rule_types = torch.tensor([], dtype=torch.long, device=device)
                    rule_meass = torch.tensor([], dtype=torch.float32, device=device)
                else:
                    rule_times, rule_types, rule_meass = zip(*rule_events)
                    rule_times = torch.tensor(rule_times, dtype=torch.float32, device=device)
                    rule_types = torch.tensor(rule_types, dtype=torch.long, device=device)
                    rule_meass = torch.tensor(rule_meass, dtype=torch.float32, device=device)
                nll, mae, rmse = model.evaluate(event_times, event_types, event_meass, rule_times, rule_types, rule_meass, target_name)
                eval_nll += nll.item() / eval_size
                eval_mae += mae.item() / eval_size
                eval_rmse += rmse.item() / eval_size
    return eval_nll, eval_mae, eval_rmse


def train_model(model, data, rule_event_data, target_name, device="cpu", num_epochs=100, lr=0.01, patience=5, if_print=False):
    # 训练设置
    train_prop = 0.8
    batch_size = 64
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_eval_loss = float('inf')
    patience_counter = 0
    # 训练过程
    train_size = int(train_prop * len(data))
    test_size = len(data) - train_size
    train_data, test_data = data[:train_size], data[train_size:]
    train_rule_event_data, test_rule_event_data = rule_event_data[:train_size], rule_event_data[train_size:]
    train_dataset = EventDataset(train_data, train_rule_event_data)
    test_dataset = EventDataset(test_data, test_rule_event_data)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    output_list = []
    for epoch in range(num_epochs):
        total_loss = train_epoch(model, optimizer, train_dataloader, device)
        eval_nll, eval_mae, eval_rmse = eval_epoch(model, test_dataloader, target_name, device)
        eval_loss = total_loss
        output_list.append([total_loss, eval_nll, eval_mae, eval_rmse])
        if if_print:
            print(f'Epoch {epoch}, Loss: {total_loss}')
            print(f'Eval NLL: {eval_nll}, Eval MAE: {eval_mae}, Eval RMSE: {eval_rmse}',)
        # Early stopping check
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            if if_print:
                print("Early stopping triggered.")
            break
    return eval_loss, output_list


def generate_candidate_rules(variables, operators=["before","equal","and"], max_order=1):
    all_combinations = []
    for order in range(1, max_order + 1):
        for combination in product(variables, repeat=order):
            if len(set(combination)) == order:  # 确保所有变量都是唯一的
                for ops in product(operators, repeat=order - 1):
                    rule = []
                    for i in range(order):
                        rule.append(combination[i])
                        if i < order - 1:
                            rule.append(ops[i])
                    all_combinations.append(" ".join(rule))
    return all_combinations


def score(data, var_name_dict, rules, target_name, device="cpu"):
    start_time = time.time()
    rule_set = RuleSet(data, var_name_dict)
    for rule in rules:
        rule_set.add_rule(rule)
    model = RuleBasedTPP(rule_set.var_name_dict, rule_set.rule_name_dict, rule_set.rule_var_ids, device=device)
    model.to(device)
    loss, _ = train_model(model, data, rule_set.rule_event_data, target_name, device, num_epochs=100, lr=0.01, patience=5)
    end_time = time.time()
    print(f"Score computation time: {end_time - start_time} seconds")
    return loss


import random
from skopt import gp_minimize
from skopt.space import Categorical, Integer
from skopt.utils import use_named_args
from itertools import product
from skopt.learning.gaussian_process.kernels import Matern
from skopt import forest_minimize
from skopt.callbacks import CheckpointSaver

class EarlyStopping:
    def __init__(self, tol=1e-4, patience=100):
        self.tol = tol
        self.patience = patience
        self.best_score = float('inf')
        self.counter = 0

    def __call__(self, res):
        if res.fun < self.best_score - self.tol:
            self.best_score = res.fun
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            print(f"Early stopping: No improvement for {self.patience} iterations.")
            return True
        return False

def generate_candidate_rules(variables, operators=["before","equal","and"], max_order=1, target_name=None, ref_rules=None):
    """
    根据变量、操作符和最大顺序生成候选规则。
    """
    variables = [var for var in variables if var != target_name]
    all_combinations = []
    for order in range(1, max_order + 1):
        for combination in product(variables, repeat=order):
            if len(set(combination)) == order:  # 确保变量唯一
                for ops in product(operators, repeat=order - 1):
                    rule = []
                    for i in range(order):
                        rule.append(combination[i])
                        if i < order - 1:
                            rule.append(ops[i])
                    if ref_rules is None:
                        all_combinations.append(" ".join(rule))
                    else:
                        for ref_rule in ref_rules:
                            if ref_rule in rule:
                                all_combinations.append(" ".join(rule))
                                break
    return all_combinations


def evaluate_loss(temporary_rules, current_rules, data, var_name_dict, target_name, device="cpu"):
    """
    评估给定规则集的损失。
    """
    rules = current_rules + temporary_rules
    loss = score(data, var_name_dict, rules, target_name, device)
    return float(loss)


def optimize(data, var_name_dict, target_name, max_order=1, num_candidates=10, n_calls=50, device="cpu"):
    """
    使用贝叶斯优化优化规则选择。
    """
    # Step 0: 找出1阶规则
    print("Optimizing rule selection...")
    selected_proportion = 0.6
    rule_set = RuleSet(data, var_name_dict)
    model = RuleBasedTPP(rule_set.var_name_dict, rule_set.rule_name_dict, rule_set.rule_var_ids, device=device)
    model.to(device)
    basic_loss, _ = train_model(model, data, rule_set.rule_event_data, target_name, device, num_epochs=100, lr=0.01, patience=5)
    variables = list(var_name_dict.keys())
    first_order_rules = generate_candidate_rules(variables, max_order=1, target_name=target_name)
    rule_losses = []
    for first_order_rule in tqdm(first_order_rules):
        rule_set = RuleSet(data, var_name_dict)
        rule_set.add_rule(first_order_rule)
        model = RuleBasedTPP(rule_set.var_name_dict, rule_set.rule_name_dict, rule_set.rule_var_ids, device=device)
        model.to(device)
        loss, _ = train_model(model, data, rule_set.rule_event_data, target_name, device, num_epochs=100, lr=0.01, patience=5)
        if loss < basic_loss:
            rule_losses.append((first_order_rule, loss))
    rule_losses.sort(key=lambda x: x[1])
    num_selected = int(selected_proportion * num_candidates)

    if len(rule_losses) >= num_selected:
        selected_first_rules = [rule for rule, _ in rule_losses[:num_selected]]
        print(f"Number of selected first order rules: {num_selected}")
    # if len(rule_losses) >= num_candidates:
    #     selected_first_rules = [rule for rule, _ in rule_losses[:num_candidates]]
    #     rule_set = RuleSet(data, var_name_dict)
    #     for selected_first_rule in selected_first_rules:
    #         rule_set.add_rule(selected_first_rule)
    #     model = RuleBasedTPP(rule_set.var_name_dict, rule_set.rule_name_dict, rule_set.rule_var_ids, device=device)
    #     model.to(device)
    #     loss, _ = train_model(model, data, rule_set.rule_event_data, target_name, device, num_epochs=100, lr=0.01, patience=5, if_print=True)
    #     return selected_first_rules, loss

    if len(rule_losses) == 0:
        print("No rule found.")
        return [], basic_loss
    
    # Step 1: 生成候选规则
    candidate_rules = generate_candidate_rules(variables, max_order=max_order, target_name=target_name, ref_rules=selected_first_rules)
    print(f"Number of candidate rules: {len(candidate_rules)}")
    random.shuffle(candidate_rules)
    rule_indices = list(range(len(candidate_rules)))

    # Step 2: 定义搜索空间
    search_space = [Categorical(rule_indices, name=f"rule_{i}") for i in range(num_candidates)]

    # Step 3: 定义目标函数
    @use_named_args(search_space)
    def objective(**kwargs):
        # 解码规则索引以获取实际规则
        selected_indices = [kwargs[f"rule_{i}"] for i in range(num_candidates)]
        selected_rules = [candidate_rules[idx] for idx in selected_indices]
        return evaluate_loss(selected_rules, [], data, var_name_dict, target_name, device)

    # Step 4: 运行贝叶斯优化
    # result = gp_minimize(
    #     func=objective,
    #     dimensions=search_space,
    #     n_jobs=-1, # 使用核心数
    #     n_calls=n_calls,  # 迭代次数
    #     random_state=24,
    #     verbose=True
    # )

    early_stopping = EarlyStopping(tol=1e-4, patience=100)
    result = forest_minimize(
        func=objective,
        dimensions=search_space,
        n_calls=n_calls,  # 迭代次数
        #n_initial_points=100,  # 随机点数
        n_jobs=-1,        # 使用核心数
        random_state=24,
        callback=[early_stopping],  # 早停
        verbose=True
    )

    # Step 5: 提取最佳规则及其损失
    best_indices = result.x  # 最佳规则索引
    best_rules = [candidate_rules[idx] for idx in best_indices]
    best_loss = result.fun

    return best_rules, best_loss

if __name__ == "__main__":

    # 测试代码
    file_path = "aki_dataset_all.csv"
    target_name = "Phase III"

    data, var_name_dict = read_data(file_path, target_varibles=target_name)
    print(f"The data have {len(data)} samples.")

    start_time = time.time()

    rule_set = RuleSet(data, var_name_dict)
    rules = ["BUN High", "PTT High", "PTT Low", "BUN Low", "PTT High before BUN High", "PTT Low before BUN Low"]
    for rule in rules:
        rule_set.add_rule(rule)
    device = "cpu"

    model = RuleBasedTPP(rule_set.var_name_dict, rule_set.rule_name_dict, rule_set.rule_var_ids, device=device)
    model.to(device)
    
    print(rule_set.rule_event_data[0])
    loss, _ = train_model(model, data, rule_set.rule_event_data, target_name, device, num_epochs=100, lr=0.01, patience=5)