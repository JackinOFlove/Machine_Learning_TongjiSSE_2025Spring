# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import pandas as pd
from train_utilsconst import *
import matplotlib.pyplot as plt
from scipy.special import comb, perm
from itertools import combinations, permutations

# Integrated model for N_model subformulas
class Multi_model(nn.Module):
    def __init__(self, N_model, fea_N, T):
        super(Multi_model, self).__init__()
        self.N_model = N_model
        self.T = T
        self.fea_N = fea_N
        self.model_layers = nn.ModuleList([Model(fea_N, T) for _ \
                                            in range(self.N_model)])
        self.weights = torch.nn.Parameter(torch.randn(1, self.N_model), \
                                          requires_grad=True)
        self.betas = torch.nn.Parameter(torch.rand(1), requires_grad=True)
    
    def get_readable_formula(self):
        """获取所有子模型的可读公式"""
        formulas = []
        weights = torch.softmax(self.weights, dim=1)  # 获取每个子模型的权重
        
        for i, model in enumerate(self.model_layers):
            if hasattr(model, '_get_readable_formula'):
                weight = weights[0, i].item()
                formula = model._get_readable_formula()
                formulas.append((formula, weight))
        
        # 按权重排序
        formulas.sort(key=lambda x: x[1], reverse=True)
        
        # 返回权重最高的公式
        if formulas:
            return f"{formulas[0][0]} (weight: {formulas[0][1]:.3f})"
        else:
            return "No formula available"
    def forward(self, x):
        model_outs = []
        for model_idx in range(self.N_model):
            model_outs.append(self.model_layers[model_idx](x))
        self.rns_multi = torch.cat((model_outs), -1)
        if torch.rand(1).item() < 0.01:  # 以较低概率记录
            with open('output.txt', 'a') as f:
                f.write(f"\nOutput Summary: {self.rns_multi.detach().cpu().numpy()[:10]}\n")
        # 假设 self.weights 是一个张量
        max_idx = torch.argmax(self.weights)  # 找到最大值的索引

        # 创建 weight2，并将最大值位置设为 1，其他为 0
        self.weight2 = torch.zeros_like(self.weights)  # 初始化为 0
        self.weight2[0, max_idx] = 1  # 将最大值位置设为 1

        self.out = torch.exp(torch.sum(self.weight2 * self.rns_multi, -1) + \
                             self.betas)
        with open('output.txt', 'a') as f:
            f.write("\n\n")
            f.write(str(self.out))
        # print("条件强度为",self.out)
        return self.out
    
    
    def Const_loss(self):
        self.const_loss = 0
        for model in self.model_layers:
            self.const_loss += model.Const_loss()
        
        return self.const_loss
    
    def IG_loss(self):
        Ig_Loss = 0
        for model in self.model_layers:
            Ig_Loss += model.Const_loss()
        return Ig_Loss
        
    def Clamp_proj(self):
        for model in self.model_layers:
            model.Clamp_proj()
    
    def Clamp_alpha(self):
        for model in self.model_layers:
            model.Clamp_alpha()
    
# 单个逻辑公式模型
# Model design for a single formula
class Model(nn.Module):
    def __init__(self, N, T):
        super(Model,self).__init__()
        
        # 事件类型数
        self.N = N
        # 时间窗口
        self.T = T
        # 逻辑公式数
        self.Ordprednum = self.N * (self.N - 1)
        # 逻辑公式层
        self.ordmdl_layers = nn.ModuleList([Order_module(self.T) for _ \
                                            in range(self.Ordprednum)])
        self.multiord_layer = Multiorder_Module(self.Ordprednum, self.T)
        self.predpar_layer = Predicate_time(self.T, self.N)
        self.conjpar_layer = TimeorderConj(self.T, self.N + 1)
        self.disjpar_layer = TimeorderDisj(self.T, self.N + 1)
        self.binary_layer = Binary(self.T, 2)
        
        # 添加事件类型名称映射
        self.event_names = None  # 需要从外部设置
        
    # 添加设置事件名称的方法
    def set_event_names(self, names):
        """设置事件类型名称
        Args:
            names: 事件类型名称列表，如 ['A', 'B', 'C']
        """
        self.event_names = names
        
    def forward(self, x):
        rns = []
        i = 0
        
        # 打印基础顺序关系
        print("\n=== 基础顺序关系 ===")
        for cur_perm in permutations(list(range(self.N)), 2):
            result = self.ordmdl_layers[i](x[:,:,cur_perm])
            # 只在event_names存在时打印具体事件名称
            if self.event_names:
                event1 = self.event_names[cur_perm[0]]
                event2 = self.event_names[cur_perm[1]]
            rns.append(result.unsqueeze(-1))    
            i += 1
        
        if rns:
            self.rn_ord1toN = torch.cat((rns), -1)
            self.rn_ords = self.multiord_layer(self.rn_ord1toN).unsqueeze(-1)
            print("\n=== 组合后的顺序关系 ===")
            print(self.rn_ords.shape)
            
            self.rn_clk = self.predpar_layer(x)
            print("\n=== 时间谓词结果 ===")
            print(self.rn_clk.shape)
            
            self.rn_ordclk = torch.cat((self.rn_ords, self.rn_clk), -1)
            
            # 打印逻辑与和逻辑或的结果
            self.ret_conj = self.conjpar_layer(self.rn_ordclk).unsqueeze(-1)
            self.ret_disj = self.disjpar_layer(self.rn_ordclk).unsqueeze(-1)
            print("\n=== 逻辑运算结果 ===")
            print(f"逻辑与ret_conj: {self.ret_conj.shape}")
            print(f"逻辑或ret_disj: {self.ret_disj.shape}")
            
            self.ret_cdcat = torch.cat((self.ret_conj, self.ret_disj),-1)
            self.ret = self.binary_layer(self.ret_cdcat)
            print("\n=== 最终选择的逻辑公式 ===")
            print(f"结果ret: {self.ret.shape}")
            
            # 只在event_names存在时打印可读公式
            if self.event_names:
                formula = self._get_readable_formula()
                with open('output.txt', 'a') as f:
                    f.write(f"\n最终逻辑公式: {formula}")
                print(f"\n最终逻辑公式: {formula}")
        return self.ret
        
    def _get_readable_formula(self):
        """生成符合wCL格式的逻辑公式，使用学习到的时间阈值"""
        if not hasattr(self, 'event_names'):
            return "Event names not set"
            
        threshold = 0  # 关系强度阈值
        max_relations = 10  # 最大显示的关系数量
        
        # 收集所有谓词和权重
        predicates = []
        
        #  收集成对顺序谓词
        i = 0
        for idx1, event1 in enumerate(self.event_names):
            for idx2, event2 in enumerate(self.event_names):
                if idx1 != idx2:
                    value = self.rn_ord1toN[:, -1, i].mean().item()
                    if value > threshold:
                        # 获取该关系的时间阈值
                        time_threshold = self.ordmdl_layers[i].get_time_threshold().item()
                        pred = f"(c_{event1} - c_{event2} > {time_threshold:.2f})"
                        
                        # 给与P/F相关的关系更高的优先级
                        if "Middle_to_Sever" in [event1, event2]:
                            value += 0.03
                        predicates.append((pred, value))
                    i += 1
        
        # 按权重排序并只保留前max_relations个
        predicates.sort(key=lambda x: x[1], reverse=True)
        predicates = predicates[:max_relations]
        
        # 构建最终公式
        if not predicates:
            return "No significant predicates found"
        
        # 确定使用AND还是OR
        if self.ret_conj.mean().item() > self.ret_disj.mean().item():
            operator = " ∧ "
        else:
            operator = " ∨ "
            
        formula_parts = [f"{pred}{value:.2f}" for pred, value in predicates]
        formula = operator.join(formula_parts)
        
        return f"φ = {formula}"
    
    def get_weights(self):
        return self.ordmdl_layer.get_order_weights()
    
    
    def Const_loss(self):
        const_loss = []
        for layer in self.ordmdl_layers:
            const_loss.append(layer.Const_loss())
        self.const_loss = torch.sum(torch.stack(const_loss)) + \
        self.multiord_layer.Const_loss() + self.conjpar_layer.Const_loss() + \
        self.disjpar_layer.Const_loss() 
        
        return self.const_loss
    

    def Clamp_proj(self):
        for layer in self.ordmdl_layers:
            layer.Clamp_proj()
        self.multiord_layer.Clamp_proj()
        self.conjpar_layer.Clamp_proj()
        self.disjpar_layer.Clamp_proj()
    
    
    def Clamp_alpha(self):
        self.binary_layer.clamp_Alpha()
        
    
    def IG_loss(self):
        loss = 0
        loss += self.binary_layer.IG_loss()
        
        return loss        
'''
class Binary(nn.Module):
    def __init__(self, T, output_dim):
        super(Binary, self).__init__()
        self.T = T
        self.output_dim = output_dim
        # 初始化alpha参数，用于二元选择
        self.alpha = torch.nn.Parameter(torch.rand(output_dim), requires_grad=True)
    
    def forward(self, x):
        # 实现二元选择的逻辑
        return x @ F.softmax(self.alpha, dim=0)
    
    def get_weights(self):
        """返回二元选择的权重"""
        return self.alpha.detach().numpy()
    
    def clamp_Alpha(self):
        """确保alpha参数在合理范围内"""
        with torch.no_grad():
            self.alpha.clamp_(min=0.0)
'''
    
    
    
    
    
    
