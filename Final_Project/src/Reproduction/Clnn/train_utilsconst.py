# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# paired order predicate
class Predicate_order(nn.Module):
    def __init__(self, T):
        super(Predicate_order, self).__init__()
        self.T = T
        
    def forward(self, x1, x2):
        return torch.sigmoid(x1-x2).unsqueeze(-1)

# conjunction for paired order cell 成对序单元的连接
class OrderConj(nn.Module):
    def __init__(self, T):
        super(OrderConj,self).__init__()        
        self.A1 = torch.nn.Parameter(torch.rand(T, 2), requires_grad = True)
        self.beta1 = torch.nn.Parameter(torch.tensor(1.),requires_grad=True)
        self.T = T
        
    def forward(self, x1, x2):
         self.x12 = torch.cat((x1, x2), -1)
         self.weightbias1 = activate_conj(self.x12, self.A1, self.beta1, dim = -1)
         self.activate1 = clamp(self.weightbias1)
         
         return self.activate1
    
    def Clamp_proj(self):
        self.A1.data = torch.relu(self.A1.data)
        self.beta1.data = clamp_beta(self.beta1.data)
    
    def Const_loss(self):
        const_loss = (torch.sum(self.A1) - 2 * self.beta1 + 1) ** 2
        
        return const_loss
    
    def L1_weight(self):
        L1_loss = torch.norm(self.A1, 1)
        
        return L1_loss


# Paired order cell
class Order_module(nn.Module):
   def __init__(self, T):
       super(Order_module,self).__init__()        
       self.T = T
       self.time_threshold = nn.Parameter(torch.randn(1) * 2.0, requires_grad=True)
       self.pred_ord_layer = Predicate_order(self.T)
       
   def forward(self, x):
        """
        实现双向的时间顺序关系
        x: [batch, seq_len, 2] 包含两个事件的clock signals
        """
        # 计算时间差
        time_diff = x[:,:,0] - x[:,:,1]  # cli - clj
        
        # 根据time_threshold的符号确定关系方向
        order_relation = torch.where(
            self.time_threshold >= 0,
            time_diff - self.time_threshold,  # 正向关系：cli - clj > ulilj
            -time_diff + self.time_threshold  # 反向关系：clj - cli > -ulilj
        )
        
        # 使用sigmoid将结果映射到[0,1]
        return torch.sigmoid(order_relation)
    
   def get_time_threshold(self):
        """返回时间阈值，包括其符号"""
        return self.time_threshold
    
   def get_order_description(self, event1, event2):
        """
        返回可读的顺序关系描述
        Args:
            event1, event2: 事件标签名称
        Returns:
            (predicate_str, is_forward)
        """
        threshold = self.time_threshold.item()
        if threshold >= 0:
            # li在lj之前至少threshold时间单位
            return f"(c_{event1} - c_{event2} > {threshold:.2f})", True
        else:
            # lj在li之前至多|threshold|时间单位
            return f"(c_{event2} - c_{event1} > {-threshold:.2f})", False
    
   def Const_loss(self):
        """约束loss，但不限制符号"""
        # 只约束绝对值不要太大
        return 0.01 * torch.abs(self.time_threshold)
    
   def Clamp_proj(self):
        """限制时间阈值在合理范围内，但允许正负值"""
        with torch.no_grad():
            # 限制在[-24, 24]小时范围内
            self.time_threshold.clamp_(-24.0, 24.0)
    
   def L1_weight(self):
        return self.ordconj_layer.L1_weight()


# Conjunction of paired order cells
class Multiorder_Module(nn.Module):
    def __init__(self, M, T):
        super(Multiorder_Module, self).__init__()
        self.T = T
        self.M = M
        self.A = torch.nn.Parameter(torch.randn(self.T, self.M), \
                                    requires_grad = True)
        self.beta = torch.nn.Parameter(torch.tensor(1.),requires_grad=True)
    
    def forward(self, x):
        self.weightbias1 = activate_conj(x, self.A, self.beta, dim = -1)
        self.activate1 = clamp(self.weightbias1)
        
        return self.activate1
        
    def Const_loss(self):
        const_loss = (torch.sum(self.A) - 2 * self.beta + 1) ** 2
         
        return const_loss
    
    def Clamp_proj(self):
        self.A.data = torch.relu(self.A.data)
        self.beta.data = clamp_beta(self.beta.data)
    
    def L1_weight(self):
        L1_Loss = torch.norm(self.A, 1)
        
        return L1_Loss
    

# Singleton order predicate
class Predicate_time(nn.Module):
    def __init__(self, T, N):
        super(Predicate_time, self).__init__()
        self.T = T
        self.N = N
        # 每个事件的时间窗口参数 ulj
        self.time_windows = nn.Parameter(torch.rand(N) * 5.0, requires_grad=True)
        
    def forward(self, x):
        """
        实现 clj - ulj < 0
        x: [batch, seq_len, N] 包含所有事件的clock signals
        """
        # 对每个事件计算时间窗口判断
        time_checks = []
        for j in range(self.N):
            # clj - ulj < 0
            check = x[:,:,j] - self.time_windows[j]
            time_checks.append(torch.sigmoid(-check))
            
        return torch.stack(time_checks, dim=-1)
    
    def get_time_window(self, idx):
        """返回指定事件的时间窗口参数"""
        return self.time_windows[idx]
        
    def Const_loss(self):
        # 确保时间窗口参数为正
        return torch.sum(torch.relu(-self.time_windows))


# Conjunction of POC and SOC POC与SOC的结合
class TimeorderConj(nn.Module):
    def __init__(self, T, M):
        super(TimeorderConj, self).__init__()        
        self.A1 = torch.nn.Parameter(torch.ones(T, M) * 1/M, requires_grad=True)
        self.beta1 = torch.nn.Parameter(torch.tensor(1.),requires_grad=True)
        self.T = T
        
    def forward(self, x1):
         #print(x1)
         
         self.weightbias1 = activate_conj(x1, self.A1, self.beta1, dim=-1)
         self.activate1 = clamp(self.weightbias1)
         
         return self.activate1
     
    def Const_loss(self):
        const_loss = (torch.sum(self.A1) - 2 * self.beta1 + 1) ** 2
         
        return const_loss
    
    def Clamp_proj(self):
        self.A1.data = torch.relu(self.A1.data)
        self.beta1.data = clamp_beta(self.beta1.data)
        
    def L1_weight(self):
        L1_loss = torch.norm(self.A1, 1)
        
        return L1_loss
        
    
# Disjunction of POC and SOC POC和SOC的分离
class TimeorderDisj(nn.Module):
    def __init__(self, T, M):
        super(TimeorderDisj, self).__init__()        
        self.A1 = torch.nn.Parameter(torch.randn(T, M), requires_grad=True)
        self.beta1 = torch.nn.Parameter(torch.tensor(1.),requires_grad=True)
        self.T = T
        
    def forward(self, x1):
         self.weightbias1 = activate_disj(x1, self.A1, self.beta1, dim=-1)
         self.activate1 = clamp(self.weightbias1)
         
         return self.activate1
     
    def Const_loss(self):
        const_loss = (torch.sum(self.A1) - 2 * self.beta1 + 1) ** 2
         
        return const_loss 
    
    def Clamp_proj(self):
        self.A1.data = torch.relu(self.A1.data)
        self.beta1.data = clamp_beta(self.beta1.data)
    
    def L1_weight(self):
        L1_loss = torch.norm(self.A1, 1)
        
        return L1_loss
    
    
# Architecture cell
class Binary(nn.Module):
    def __init__(self, T, M):
        super(Binary, self).__init__()        
        self.alpha = torch.nn.Parameter(torch.randn(1, M), requires_grad=True)
    
    def forward(self, x):
        self.alpha_sm = F.softmax(self.alpha, dim = -1)
        self.ret = torch.sum(x * self.alpha_sm, dim = -1)
        
        return self.ret

    def IG_loss(self):
        
        Ig_Loss = 1 - torch.sum(self.alpha_sm ** 2)
        
        return Ig_Loss
    
    def clamp_Alpha(self):
        minidx = torch.argmin(self.alpha.data)
        self.alpha.data[0,minidx] = -10



def IG_loss(x):
    
    return 1 - torch.sum(x ** 2)

def activate_conj(x, A, beta, dim):
    return (beta - torch.sum(A * (torch.ones_like(x) - x), dim))

def activate_disj(x, A, beta, dim):
    return (1 - beta + torch.sum(A * x, dim))

def clamp(x):
    return torch.max(torch.zeros_like(x), torch.min(torch.ones_like(x),x))

def clamp_beta(x):
    return torch.max(torch.ones_like(x) * 0.5, x)

