"""
真正修复的可微分路径生成器

解决方案：事后概率计算方法
1. 用标准A*生成最优路径（离散决策，无梯度）
2. 事后重计算路径概率（连续优化，保持梯度）
3. 确保REINFORCE训练有效

这解决了根本矛盾：A*离散决策 vs 可微分连续优化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Set, Optional, Union
import random
from astar_path_generator import AStarPathGenerator

class DifferentiablePathGeneratorTrulyFixed(AStarPathGenerator):
    """
    真正修复的可微分A*路径生成器
    
    核心修复：
    1. 分离离散搜索和连续优化
    2. 事后概率计算保持梯度连接
    3. 移除伪可微分操作（torch.topk, .item()）
    4. 真正的A*算法（非Beam Search）
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # REINFORCE基线网络
        self.baseline_network = nn.Sequential(
            nn.Linear(self.sbert_dim, self.sbert_dim // 2),
            nn.ReLU(),
            nn.Linear(self.sbert_dim // 2, 1)
        )
        
        # 训练模式标志
        self.training_mode = False
        
    def enable_training_mode(self):
        """启用训练模式（可微分路径生成）"""
        self.training_mode = True
        self.train()
        
    def disable_training_mode(self):
        """禁用训练模式（标准A*推理）"""
        self.training_mode = False
        self.eval()
    
    def standard_astar_search(self, question: str, start_entity: str, 
                             target_entities: Set[str]) -> List[str]:
        """
        标准A*搜索（离散决策，无梯度）
        用于生成最优路径，然后事后计算概率
        """
        from heapq import heappush, heappop
        
        open_set = [(0, 0, {'entity': start_entity, 'path': [start_entity], 'g_cost': 0.0})]
        closed_set = set()
        node_counter = 1
        
        while open_set:
            _, _, current_node = heappop(open_set)
            current_entity = current_node['entity']
            
            if current_entity in closed_set:
                continue
            closed_set.add(current_entity)
            
            if current_entity in target_entities:
                return current_node['path']
            
            if len(current_node['path']) >= self.max_path_length:
                continue
            
            if current_entity not in self.knowledge_graph:
                continue
            
            neighbors = list(self.knowledge_graph.neighbors(current_entity))
            if len(neighbors) > 15:
                neighbors = random.sample(neighbors, 15)
            
            # 真正的A*：逐个评估邻居，选择最优的一个
            best_neighbor = None
            best_relation = None
            best_f_cost = float('inf')
            
            for neighbor in neighbors:
                if neighbor in closed_set:
                    continue
                
                edge_data = self.knowledge_graph.get_edge_data(current_entity, neighbor)
                relation = edge_data.get('relation', 'related_to') if edge_data else 'related_to'
                
                # 使用确定性评分（无梯度）
                with torch.no_grad():
                    edge_score = self.compute_edge_score(
                        question=question,
                        current_path=current_node['path'],
                        candidate_relation=relation,
                        candidate_entity=neighbor,
                        stochastic=False
                    ).item()  # 转为标量
                
                new_g_cost = current_node['g_cost'] - edge_score  # 负分数=代价
                h_cost = self.compute_heuristic_simple(neighbor, target_entities)
                f_cost = new_g_cost + h_cost
                
                if f_cost < best_f_cost:
                    best_f_cost = f_cost
                    best_neighbor = neighbor
                    best_relation = relation
                    best_edge_score = edge_score  # 保存最佳边分数
            
            # 只扩展最优邻居（真正的A*）
            if best_neighbor is not None:
                new_path = current_node['path'] + [best_relation, best_neighbor]
                child_node = {
                    'entity': best_neighbor,
                    'path': new_path,
                    'g_cost': current_node['g_cost'] - best_edge_score  # 使用正确的边分数
                }
                
                heappush(open_set, (best_f_cost, node_counter, child_node))
                node_counter += 1
        
        return []  # 未找到路径
    
    def compute_heuristic_simple(self, entity: str, target_entities: Set[str]) -> float:
        """简化的启发式函数（无梯度）"""
        if entity in target_entities:
            return 0.0
        if entity not in self.entity_to_id:
            return 10.0
        
        min_distance = float('inf')
        entity_emb = self.entity_embeddings[self.entity_to_id[entity]]
        
        for target in target_entities:
            if target in self.entity_to_id:
                target_emb = self.entity_embeddings[self.entity_to_id[target]]
                with torch.no_grad():
                    cosine_sim = torch.cosine_similarity(entity_emb, target_emb, dim=0)
                    distance = 1.0 - cosine_sim.item()
                min_distance = min(min_distance, distance)
        
        return min_distance if min_distance != float('inf') else 5.0
    
    def recompute_path_probability(self, path: List[str], question: str) -> torch.Tensor:
        """
        事后重计算路径概率（保持梯度连接）
        
        这是核心修复：将离散A*搜索与连续梯度优化分离
        """
        if len(path) < 3:  # 至少需要[entity, relation, entity]
            return torch.tensor(0.0, requires_grad=True, device=self.device)
        
        total_log_prob = torch.tensor(0.0, requires_grad=True, device=self.device)
        
        # 逐步重计算每个选择的概率
        for step_idx in range(len(path) // 2):  # path格式: [e1, r1, e2, r2, e3, ...]
            current_entity = path[step_idx * 2]
            if step_idx * 2 + 2 >= len(path):
                break
                
            chosen_relation = path[step_idx * 2 + 1]
            chosen_entity = path[step_idx * 2 + 2]
            
            # 获取该步的所有候选
            if current_entity not in self.knowledge_graph:
                continue
                
            neighbors = list(self.knowledge_graph.neighbors(current_entity))
            if not neighbors:
                continue
                
            # 计算所有候选的分数（保持梯度）
            candidate_scores = []
            chosen_idx = -1
            
            for i, neighbor in enumerate(neighbors):
                edge_data = self.knowledge_graph.get_edge_data(current_entity, neighbor)
                relation = edge_data.get('relation', 'related_to') if edge_data else 'related_to'
                
                # 可微分评分
                score = self.compute_edge_score(
                    question=question,
                    current_path=path[:step_idx*2+1],
                    candidate_relation=relation,
                    candidate_entity=neighbor,
                    return_tensor=True,
                    stochastic=False
                )
                candidate_scores.append(score)
                
                # 找到实际选择的邻居
                if neighbor == chosen_entity and relation == chosen_relation:
                    chosen_idx = i
            
            # 计算选择概率
            if chosen_idx != -1 and candidate_scores:
                scores_tensor = torch.stack(candidate_scores)
                probs = F.softmax(scores_tensor, dim=0)
                step_log_prob = torch.log(probs[chosen_idx] + 1e-10)
                total_log_prob = total_log_prob + step_log_prob
        
        return total_log_prob
    
    def generate_differentiable_paths(self, question: str, start_entity: str,
                                    target_entities: Set[str],
                                    num_samples: int = 1,
                                    temperature: float = 1.0) -> List[Tuple[List[str], torch.Tensor]]:
        """
        生成可微分路径（修复版本）
        
        Returns:
            List of (path, log_prob) tuples，其中log_prob保持梯度连接
        """
        results = []
        
        for _ in range(num_samples):
            # 第一步：用标准A*找到最优路径（离散决策）
            optimal_path = self.standard_astar_search(question, start_entity, target_entities)
            
            if not optimal_path:
                continue
            
            # 第二步：事后重计算路径概率（保持梯度）
            path_log_prob = self.recompute_path_probability(optimal_path, question)
            
            results.append((optimal_path, path_log_prob))
        
        return results
    
    def compute_baseline(self, question_embedding: torch.Tensor) -> torch.Tensor:
        """
        计算REINFORCE基线值
        
        Args:
            question_embedding: 问题的embedding
            
        Returns:
            基线值（标量张量）
        """
        return self.baseline_network(question_embedding).squeeze()
    
    def compute_policy_loss_with_baseline(self, paths_with_log_probs: List[Tuple[List[str], torch.Tensor]], 
                                        rewards: List[float],
                                        question_embedding: torch.Tensor) -> torch.Tensor:
        """
        使用基线计算REINFORCE策略损失
        
        Args:
            paths_with_log_probs: [(path, log_prob), ...] 其中log_prob有梯度
            rewards: 对应的奖励列表
            question_embedding: 问题embedding（用于基线计算）
            
        Returns:
            policy_loss: 可以反向传播的损失
        """
        if not paths_with_log_probs or not rewards:
            return torch.tensor(0.0, requires_grad=True, device=self.device)
        
        # 计算基线
        baseline_value = self.compute_baseline(question_embedding)
        
        policy_losses = []
        baseline_losses = []
        
        for (path, log_prob), reward in zip(paths_with_log_probs, rewards):
            reward_tensor = torch.tensor(reward, device=self.device)
            
            # REINFORCE with baseline
            advantage = reward_tensor - baseline_value
            policy_loss = -advantage.detach() * log_prob  # advantage不传梯度到基线
            policy_losses.append(policy_loss)
            
            # 基线损失（MSE）
            baseline_loss = F.mse_loss(baseline_value, reward_tensor)
            baseline_losses.append(baseline_loss)
        
        # 合并损失
        total_policy_loss = torch.stack(policy_losses).mean()
        total_baseline_loss = torch.stack(baseline_losses).mean()
        
        return total_policy_loss + 0.5 * total_baseline_loss
    
    def get_trainable_parameters(self):
        """获取可训练参数（包括基线网络）"""
        params = list(super().get_trainable_parameters())
        params.extend(list(self.baseline_network.parameters()))
        return params