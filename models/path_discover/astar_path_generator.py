import torch
import torch.nn as nn
import torch.nn.functional as F
import heapq
import networkx as nx
from typing import Dict, List, Tuple, Set, Optional, Union
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import numpy as np
from collections import defaultdict
import random

@dataclass
class SearchNode:
    """A*搜索节点"""
    entity: str
    path: List[str]  # 完整路径 [entity, relation, entity, relation, ...]
    g_cost: float    # 从起点到当前节点的实际代价 (负数，越大越好)
    h_cost: float    # 启发式代价（预估到终点的代价）
    f_cost: float    # f = g + h
    log_prob: float = 0.0  # 累积对数概率 log P(τ)，用于REINFORCE
    parent: Optional['SearchNode'] = None
    
    def __lt__(self, other):
        """用于优先队列排序"""
        return self.f_cost < other.f_cost

class RelationPredictorNetwork(nn.Module):
    """s_rel: 关系预测模型 - 条件策略网络"""
    
    def __init__(self, hidden_dim: int = 256, sbert_dim: int = 384):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 问题编码器（使用预训练SBERT）
        self.question_encoder = nn.Linear(sbert_dim, hidden_dim)
        
        # 路径上下文编码器
        self.path_encoder = nn.LSTM(sbert_dim, hidden_dim // 2, batch_first=True, bidirectional=True)
        
        # 关系预测头
        self.relation_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # 输出单个分数
        )
    
    def forward(self, question_emb: torch.Tensor, path_embs: torch.Tensor, relation_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            question_emb: (batch, sbert_dim) 问题embedding
            path_embs: (batch, seq_len, sbert_dim) 路径上实体的embeddings
            relation_emb: (batch, sbert_dim) 候选关系embedding
        Returns:
            score: (batch,) 关系分数
        """
        batch_size = question_emb.size(0)
        
        # 编码问题
        q_encoded = self.question_encoder(question_emb)  # (batch, hidden_dim)
        
        # 编码路径上下文
        if path_embs.size(1) > 0:
            path_output, (h_n, c_n) = self.path_encoder(path_embs)
            # 双向LSTM：连接前向和后向的最后隐状态
            path_encoded = torch.cat([h_n[0], h_n[1]], dim=-1)  # (batch, hidden_dim)
        else:
            # 如果没有路径历史，使用零向量
            path_encoded = torch.zeros(batch_size, self.hidden_dim, device=question_emb.device)
        
        # 融合问题和路径信息
        combined = torch.cat([q_encoded, path_encoded], dim=-1)  # (batch, hidden_dim * 2)
        
        
        # 预测关系分数
        rel_score = self.relation_predictor(combined).squeeze(-1)  # (batch,)
        
        return rel_score

class TransitionModel(nn.Module):
    """s_tran: 转移模型 - 一阶马尔可夫模型"""
    
    def __init__(self, relation_vocab_size: int, embedding_dim: int = 128):
        super().__init__()
        self.relation_embeddings = nn.Embedding(relation_vocab_size, embedding_dim)
        self.transition_matrix = nn.Linear(embedding_dim, relation_vocab_size)
        
    def forward(self, prev_relation_id: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prev_relation_id: (batch,) 前一个关系的ID
        Returns:
            transition_scores: (batch, vocab_size) 转移到各个关系的分数
        """
        prev_rel_emb = self.relation_embeddings(prev_relation_id)  # (batch, emb_dim)
        transition_scores = self.transition_matrix(prev_rel_emb)  # (batch, vocab_size)
        return transition_scores

class LocalSemanticModel(nn.Module):
    """s_loc: 局部语义模型 - 语义相似度匹配"""
    
    def __init__(self, hidden_dim: int = 256, sbert_dim: int = 384):
        super().__init__()
        
        # 问题-实体匹配网络
        self.question_entity_matcher = nn.Sequential(
            nn.Linear(sbert_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 问题-关系匹配网络
        self.question_relation_matcher = nn.Sequential(
            nn.Linear(sbert_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, question_emb: torch.Tensor, entity_emb: torch.Tensor, 
                relation_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            question_emb: (batch, sbert_dim)
            entity_emb: (batch, sbert_dim)
            relation_emb: (batch, sbert_dim)
        Returns:
            semantic_score: (batch,)
        """
        # 问题-实体匹配
        qe_input = torch.cat([question_emb, entity_emb], dim=-1)
        entity_match = self.question_entity_matcher(qe_input).squeeze(-1)
        
        # 问题-关系匹配
        qr_input = torch.cat([question_emb, relation_emb], dim=-1)
        relation_match = self.question_relation_matcher(qr_input).squeeze(-1)
        
        # 组合语义分数
        semantic_score = (entity_match + relation_match) / 2
        return semantic_score

class TripletConfidenceModel(nn.Module):
    """s_tri: 三元组置信模型 - 数据质量评估"""
    
    def __init__(self, hidden_dim: int = 128, sbert_dim: int = 384):
        super().__init__()
        
        self.confidence_estimator = nn.Sequential(
            nn.Linear(sbert_dim * 3, hidden_dim * 2),  # head + relation + tail
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 输出0-1之间的置信度
        )
    
    def forward(self, head_emb: torch.Tensor, relation_emb: torch.Tensor, 
                tail_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            head_emb: (batch, sbert_dim)
            relation_emb: (batch, sbert_dim)
            tail_emb: (batch, sbert_dim)
        Returns:
            confidence: (batch,) 三元组置信度
        """
        triplet_input = torch.cat([head_emb, relation_emb, tail_emb], dim=-1)
        confidence = self.confidence_estimator(triplet_input).squeeze(-1)
        return confidence

class AStarPathGenerator(nn.Module):
    """
    A*路径生成器 - 基于可学习评分函数的图搜索
    
    核心思想：
    1. 使用A*搜索作为路径生成的骨架
    2. 四个可学习的评分函数提供智能指导：
       - s_rel: 关系预测（策略网络）
       - s_tran: 关系转移（马尔可夫模型）
       - s_loc: 语义匹配
       - s_tri: 三元组置信度
    3. edge_score = w1*s_rel + w2*s_tran + w3*s_loc + w4*s_tri
    4. 支持多hop路径生成
    """
    
    def __init__(self, 
                 sbert_model_name: str = 'all-MiniLM-L6-v2',
                 hidden_dim: int = 256,
                 max_path_length: int = 6,
                 beam_width: int = 5,
                 entity_embedding_path: Optional[str] = None):
        super().__init__()
        
        self.max_path_length = max_path_length
        self.beam_width = beam_width
        self.hidden_dim = hidden_dim
        
        # 初始化SBERT并冻结参数
        self.sbert = SentenceTransformer(sbert_model_name)
        self.sbert_dim = self.sbert.get_sentence_embedding_dimension()
        self._freeze_sbert()
        
        # 四个核心评分模型
        self.s_rel = RelationPredictorNetwork(hidden_dim, self.sbert_dim)
        self.s_tran = TransitionModel(1000, 128)  # 假设1000个关系
        self.s_loc = LocalSemanticModel(hidden_dim, self.sbert_dim)  
        self.s_tri = TripletConfidenceModel(128, self.sbert_dim)
        
        # 可学习的权重参数
        self.score_weights = nn.Parameter(torch.tensor([1.0, 0.5, 1.0, 0.3]))
        
        # 实体embeddings
        self.entity_embeddings = {}
        if entity_embedding_path:
            self.load_entity_embeddings(entity_embedding_path)
        
        # 知识图谱和关系映射（外部提供）
        self.knowledge_graph = None
        self.relation_to_id = {}
        self.id_to_relation = {}
    
    def _freeze_sbert(self):
        """冻结SBERT参数"""
        for param in self.sbert.parameters():
            param.requires_grad = False
        print("[INFO] SBERT parameters frozen in A* Generator")
    
    def load_entity_embeddings(self, path: str):
        """加载预训练的实体embeddings"""
        embeddings = torch.load(path, map_location='cpu')
        self.entity_embeddings = embeddings
        print(f"Loaded {len(embeddings)} entity embeddings")
    
    def set_knowledge_graph(self, kg: nx.Graph, relation_vocab: Dict[str, int]):
        """设置知识图谱和关系词表"""
        self.knowledge_graph = kg
        self.relation_to_id = relation_vocab
        self.id_to_relation = {v: k for k, v in relation_vocab.items()}
    
    def get_entity_embedding(self, entity: str) -> torch.Tensor:
        """获取实体embedding"""
        device = next(self.parameters()).device
        if entity in self.entity_embeddings:
            return self.entity_embeddings[entity].to(device)
        else:
            # 如果没有预训练embedding，使用SBERT编码
            with torch.no_grad():
                emb = self.sbert.encode([entity], convert_to_tensor=True, device=device)
            return emb[0].to(device)
    
    def get_relation_embedding(self, relation: str) -> torch.Tensor:
        """获取关系embedding"""
        device = next(self.parameters()).device
        with torch.no_grad():
            emb = self.sbert.encode([relation], convert_to_tensor=True, device=device)
        return emb[0].to(device)
    
    def compute_edge_score(self, 
                          question: str,
                          current_path: List[str],
                          candidate_relation: str,
                          candidate_entity: str,
                          stochastic: bool = False,
                          return_tensor: bool = False) -> Union[float, torch.Tensor]:
        """
        计算边的评分：edge_score = w1*s_rel + w2*s_tran + w3*s_loc + w4*s_tri
        
        Args:
            question: 输入问题
            current_path: 当前路径 [entity, relation, entity, ...]
            candidate_relation: 候选关系
            candidate_entity: 候选实体
            stochastic: 是否使用随机化（训练时用于探索）
            return_tensor: 是否返回张量保持梯度（训练时用）
        """
        device = next(self.parameters()).device
        
        # 编码问题 - 训练时不能用no_grad
        if return_tensor:
            question_emb = self.sbert.encode([question], convert_to_tensor=True, device=device).to(device)
        else:
            with torch.no_grad():
                question_emb = self.sbert.encode([question], convert_to_tensor=True, device=device).to(device)
        
        # 编码当前路径中的实体
        if len(current_path) > 0:
            path_entities = [current_path[i] for i in range(0, len(current_path), 2)]  # 只取实体
            if return_tensor:
                path_embs = torch.stack([self.get_entity_embedding(e).to(device) for e in path_entities]).to(device)
            else:
                with torch.no_grad():
                    path_embs = torch.stack([self.get_entity_embedding(e).to(device) for e in path_entities]).to(device)
            path_embs = path_embs.unsqueeze(0)  # (1, seq_len, emb_dim)
        else:
            path_embs = torch.zeros(1, 0, self.sbert_dim, device=device)
        
        # 编码候选关系和实体
        if return_tensor:
            rel_emb = self.get_relation_embedding(candidate_relation).to(device).unsqueeze(0)
            entity_emb = self.get_entity_embedding(candidate_entity).to(device).unsqueeze(0)
        else:
            with torch.no_grad():
                rel_emb = self.get_relation_embedding(candidate_relation).to(device).unsqueeze(0)  
                entity_emb = self.get_entity_embedding(candidate_entity).to(device).unsqueeze(0)
        
        # 计算四个分量 - 关键修复：训练时不用no_grad
        if return_tensor:
            # 训练模式：保持梯度连接
            s_rel_score = self.s_rel(question_emb, path_embs, rel_emb).squeeze()
            
            # s_tran: 转移分数
            if len(current_path) >= 3:  # 至少有一个关系
                prev_rel = current_path[-2]  # 前一个关系
                prev_rel_id = self.relation_to_id.get(prev_rel, 0)
                prev_rel_tensor = torch.tensor([prev_rel_id], device=device)
                curr_rel_id = self.relation_to_id.get(candidate_relation, 0)
                s_tran_scores = self.s_tran(prev_rel_tensor)
                s_tran_score = s_tran_scores[0, curr_rel_id]
            else:
                s_tran_score = torch.tensor(0.0, device=device)
            
            # s_loc: 语义匹配分数
            s_loc_score = self.s_loc(question_emb, entity_emb, rel_emb).squeeze()
            
            # s_tri: 三元组置信度
            if len(current_path) > 0:
                head_emb = self.get_entity_embedding(current_path[-1]).to(device).unsqueeze(0)
            else:
                head_emb = question_emb  # 起始节点用问题表示
            s_tri_score = self.s_tri(head_emb, rel_emb, entity_emb).squeeze()
            
            # 加权组合 - 保持张量形式
            scores = torch.stack([s_rel_score, s_tran_score, s_loc_score, s_tri_score])
            weighted_score = torch.dot(self.score_weights, scores)
            
            # 训练时不添加随机噪声（会破坏梯度）
            return weighted_score
            
        else:
            # 推理模式：使用no_grad优化性能
            with torch.no_grad():
                s_rel_score = self.s_rel(question_emb, path_embs, rel_emb).item()
                
                # s_tran: 转移分数
                if len(current_path) >= 3:  # 至少有一个关系
                    prev_rel = current_path[-2]  # 前一个关系
                    prev_rel_id = self.relation_to_id.get(prev_rel, 0)
                    prev_rel_tensor = torch.tensor([prev_rel_id], device=device)
                    curr_rel_id = self.relation_to_id.get(candidate_relation, 0)
                    s_tran_scores = self.s_tran(prev_rel_tensor)
                    s_tran_score = s_tran_scores[0, curr_rel_id].item()
                else:
                    s_tran_score = 0.0
                
                # s_loc: 语义匹配分数
                s_loc_score = self.s_loc(question_emb, entity_emb, rel_emb).item()
                
                # s_tri: 三元组置信度
                if len(current_path) > 0:
                    head_emb = self.get_entity_embedding(current_path[-1]).to(device).unsqueeze(0)
                else:
                    head_emb = question_emb  # 起始节点用问题表示
                s_tri_score = self.s_tri(head_emb, rel_emb, entity_emb).item()
            
            # 加权组合
            scores = torch.tensor([s_rel_score, s_tran_score, s_loc_score, s_tri_score], device=self.score_weights.device)
            weighted_score = torch.dot(self.score_weights, scores).item()
            
            # 推理时可以添加随机噪声
            if stochastic:
                noise = random.gauss(0, 0.1)
                weighted_score += noise
            
            return weighted_score
    
    def optimistic_upper_bound(self, current_entity: str, target_entities: Set[str]) -> float:
        """
        启发式函数：估算从当前实体到目标实体的最优可能代价
        这是A*算法中的h(n)函数
        """
        if not self.knowledge_graph or current_entity not in self.knowledge_graph:
            return float('inf')
        
        min_distance = float('inf')
        for target in target_entities:
            if target in self.knowledge_graph:
                try:
                    distance = nx.shortest_path_length(self.knowledge_graph, current_entity, target)
                    min_distance = min(min_distance, distance)
                except nx.NetworkXNoPath:
                    continue
        
        # 转换为代价（距离越短代价越低）
        if min_distance == float('inf'):
            return float('inf')
        else:
            return -min_distance  # 负数表示好的启发值
    
    def generate_paths(self, 
                      question: str,
                      start_entity: str, 
                      target_entities: Set[str],
                      max_paths: int = 5,
                      stochastic: bool = False,
                      record_probs: bool = False) -> List[Tuple[List[str], float, float]]:
        """
        使用A*搜索生成从start_entity到target_entities的路径
        
        Args:
            question: 输入问题
            start_entity: 起始实体
            target_entities: 目标实体集合
            max_paths: 最多返回的路径数
            stochastic: 是否启用随机化探索
            record_probs: 是否记录路径概率(REINFORCE训练时使用)
            
        Returns:
            List of (path, score, log_prob) tuples，按分数降序排列
        """
        if not self.knowledge_graph:
            raise ValueError("Knowledge graph not set. Call set_knowledge_graph() first.")
        
        if start_entity not in self.knowledge_graph:
            return []
        
        # 初始化A*搜索
        open_set = []  # 优先队列
        closed_set = set()  # 已访问节点
        found_paths = []  # 找到的完整路径
        
        # 创建起始节点
        start_node = SearchNode(
            entity=start_entity,
            path=[start_entity],
            g_cost=0.0,
            h_cost=self.optimistic_upper_bound(start_entity, target_entities),
            f_cost=0.0 + self.optimistic_upper_bound(start_entity, target_entities),
            log_prob=0.0  # 起始节点概率为1 (log(1)=0)
        )
        
        heapq.heappush(open_set, (start_node.f_cost, start_node))
        
        # A*搜索主循环
        while open_set and len(found_paths) < max_paths:
            current_f, current_node = heapq.heappop(open_set)
            
            # 检查是否到达目标
            if current_node.entity in target_entities:
                path_score = current_node.g_cost  # 现在g_cost就是累积分数(正数)
                path_log_prob = current_node.log_prob if record_probs else 0.0
                found_paths.append((current_node.path.copy(), path_score, path_log_prob))
                continue
            
            # 检查路径长度限制
            if len(current_node.path) >= self.max_path_length:
                continue
            
            # 标记为已访问
            node_key = (current_node.entity, len(current_node.path))
            if node_key in closed_set:
                continue
            closed_set.add(node_key)
            
            # 扩展邻居节点
            neighbors = list(self.knowledge_graph.neighbors(current_node.entity))
            
            # 计算所有邻居的分数
            neighbor_candidates = []
            for neighbor in neighbors:
                edge_data = self.knowledge_graph.get_edge_data(current_node.entity, neighbor)
                relation = edge_data.get('relation', 'related_to') if edge_data else 'related_to'
                
                score = self.compute_edge_score(
                    question, 
                    current_node.path, 
                    relation, 
                    neighbor,
                    stochastic=stochastic
                )
                neighbor_candidates.append((score, neighbor, relation))
            
            # 选择策略：确定性 vs 随机化
            if stochastic and record_probs:
                # 随机化探索：按softmax概率采样
                scores = torch.tensor([score for score, _, _ in neighbor_candidates])
                probs = F.softmax(scores / 0.5, dim=0)  # temperature=0.5控制探索程度
                
                # 限制选择数量
                num_select = min(self.beam_width, len(neighbor_candidates))
                selected_indices = torch.multinomial(probs, num_select, replacement=False)
                
                selected_neighbors = []
                for idx in selected_indices:
                    score, neighbor, relation = neighbor_candidates[idx]
                    log_prob = torch.log(probs[idx]).item()
                    selected_neighbors.append((score, neighbor, relation, log_prob))
                    
            elif len(neighbors) > self.beam_width:
                # 确定性选择：top-k
                neighbor_candidates.sort(reverse=True)
                selected_neighbors = [(score, neighbor, relation, 0.0) 
                                    for score, neighbor, relation in neighbor_candidates[:self.beam_width]]
            else:
                # 全部选择
                selected_neighbors = [(score, neighbor, relation, 0.0) 
                                    for score, neighbor, relation in neighbor_candidates]
            
            # 创建子节点
            for edge_score, neighbor_entity, relation, step_log_prob in selected_neighbors:
                new_path = current_node.path + [relation, neighbor_entity]
                new_g_cost = current_node.g_cost + edge_score  # 累积分数(现在统一为正数越大越好)
                new_h_cost = self.optimistic_upper_bound(neighbor_entity, target_entities)
                new_f_cost = -(new_g_cost) + new_h_cost  # A*最小化f，所以用负的g_cost
                new_log_prob = current_node.log_prob + step_log_prob  # 累积对数概率
                
                child_node = SearchNode(
                    entity=neighbor_entity,
                    path=new_path,
                    g_cost=new_g_cost,
                    h_cost=new_h_cost,
                    f_cost=new_f_cost,
                    log_prob=new_log_prob,
                    parent=current_node
                )
                
                heapq.heappush(open_set, (child_node.f_cost, child_node))
        
        # 按分数排序返回
        found_paths.sort(key=lambda x: x[1], reverse=True)
        return found_paths[:max_paths]
    
    def forward(self, questions: List[str], start_entities: List[str], 
                target_entities_list: List[Set[str]], **kwargs) -> List[List[Tuple[List[str], float]]]:
        """
        批量生成路径（前向传播接口）
        
        Args:
            questions: 问题列表
            start_entities: 起始实体列表
            target_entities_list: 目标实体集合列表
            
        Returns:
            每个样本的路径列表 (path, score) tuples
        """
        results = []
        for question, start_entity, target_entities in zip(questions, start_entities, target_entities_list):
            paths_with_probs = self.generate_paths(question, start_entity, target_entities, **kwargs)
            # 为了保持向后兼容，只返回(path, score)，不返回log_prob
            paths = [(path, score) for path, score, _ in paths_with_probs]
            results.append(paths)
        return results
    
    # ==================== 预留的训练接口 ====================
    
    def get_trainable_parameters(self):
        """获取可训练参数（用于优化器）"""
        trainable_params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
        return trainable_params
    
    def compute_policy_loss(self, paths: List[List[str]], rewards: torch.Tensor) -> torch.Tensor:
        """
        计算策略梯度损失（REINFORCE）
        
        Args:
            paths: 生成的路径列表
            rewards: 判别器给出的奖励信号
            
        Returns:
            policy_loss: 策略损失
        """
        # 预留接口 - 实际实现需要记录路径生成的概率
        # L_G = -R * log P(τ)
        # 这里需要在generate_paths时记录每步的概率
        pass
    
    def update_from_discriminator_feedback(self, 
                                         paths: List[List[str]], 
                                         rewards: torch.Tensor,
                                         optimizer: torch.optim.Optimizer):
        """
        根据判别器反馈更新生成器参数
        
        Args:
            paths: 生成的路径
            rewards: 判别器奖励
            optimizer: 优化器
        """
        # 预留接口 - GAN-RL训练循环的一部分
        pass
    
    def enable_stochastic_exploration(self):
        """启用随机探索模式（训练时使用）"""
        self.stochastic_mode = True
    
    def disable_stochastic_exploration(self):
        """禁用随机探索模式（推理时使用）"""
        self.stochastic_mode = False