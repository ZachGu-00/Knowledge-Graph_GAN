import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import networkx as nx
import numpy as np
import random
from typing import List, Set, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class BeamNode:
    """Beam Search中的节点，代表一个部分路径"""
    path: List[str]  # 完整路径：[entity1, relation1, entity2, relation2, ...]
    score: float     # 累积log概率得分
    log_prob: Union[float, torch.Tensor]  # 累积对数概率（训练时保持张量）
    current_entity: str  # 当前所在实体
    
    def __post_init__(self):
        if len(self.path) > 0:
            self.current_entity = self.path[-1]


class BeamSearchPathGenerator(nn.Module):
    """
    基于Beam Search的路径生成器
    
    核心思想：
    - 使用可学习的edge_score函数作为策略网络
    - Beam Search进行宽度优先探索
    - 训练时随机采样，推理时贪婪选择
    """
    
    def __init__(self, 
                 entity_embedding_path: str = "embeddings/entity_embeddings.pt",
                 max_path_length: int = 6,
                 beam_width: int = 5,
                 sbert_model_name: str = 'all-MiniLM-L6-v2'):
        super().__init__()
        
        self.max_path_length = max_path_length
        self.beam_width = beam_width
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 知识图谱
        self.knowledge_graph = None
        
        # 语义编码器（冻结）
        self.sbert_model = SentenceTransformer(sbert_model_name)
        for param in self.sbert_model.parameters():
            param.requires_grad = False
            
        # 实体嵌入
        self.entity_embeddings = None
        if entity_embedding_path:
            self.load_entity_embeddings(entity_embedding_path)
            
        # 可学习的策略网络：关系选择网络
        self.p_rel_network = nn.Sequential(
            nn.Linear(384 + 384 + 128, 256),  # question + path + context
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)  # 输出单个分数
        )
        
        # 训练模式标志
        self.exploration_mode = False
        self.temperature = 1.0
        self.training_mode = False       # 是否为训练模式（保持梯度）
        
    def load_knowledge_graph(self, graph_path: str):
        """加载知识图谱"""
        import pickle
        with open(graph_path, 'rb') as f:
            self.knowledge_graph = pickle.load(f)
        print(f"Knowledge graph loaded: {len(self.knowledge_graph.nodes)} entities")
        
    def load_entity_embeddings(self, embedding_path: str):
        """加载实体嵌入"""
        try:
            embeddings = torch.load(embedding_path, map_location=self.device)
            self.entity_embeddings = embeddings
            print(f"Entity embeddings loaded: {len(embeddings)} entities")
        except Exception as e:
            print(f"Failed to load entity embeddings: {e}")
            
    def get_entity_embedding(self, entity: str) -> torch.Tensor:
        """获取实体嵌入"""
        if self.entity_embeddings and entity in self.entity_embeddings:
            return self.entity_embeddings[entity].to(self.device)
        else:
            # 使用SBERT编码实体名称作为后备
            with torch.no_grad():
                embedding = self.sbert_model.encode(entity, convert_to_tensor=True)
                return embedding.to(self.device)
    
    def compute_edge_score(self, 
                          question: str, 
                          current_path: List[str], 
                          relation: str, 
                          target_entity: str) -> float:
        """
        核心策略网络：计算从当前路径经过关系到目标实体的得分
        
        这是整个生成器的可学习部分！
        """
        # 1. 问题语义编码
        with torch.no_grad():
            question_emb = self.sbert_model.encode(question, convert_to_tensor=True).to(self.device)
        
        # 2. 路径语义编码
        if len(current_path) > 0:
            path_text = " -> ".join(current_path)
            with torch.no_grad():
                path_emb = self.sbert_model.encode(path_text, convert_to_tensor=True).to(self.device)
        else:
            path_emb = torch.zeros(384).to(self.device)
        
        # 3. 当前决策上下文编码
        current_entity = current_path[-1] if len(current_path) > 0 else current_path[0] if current_path else ""
        decision_text = f"{current_entity} -> {relation} -> {target_entity}"
        
        # 简化的上下文特征（降维到128）
        entity_emb = self.get_entity_embedding(target_entity)[:128]  # 截取前128维
        
        # 4. 特征融合
        features = torch.cat([
            question_emb,  # 384维：问题语义
            path_emb,      # 384维：路径语义  
            entity_emb     # 128维：实体特征
        ], dim=0)
        
        # 5. 策略网络预测 - ensure same device
        features = features.to(self.device)
        score = self.p_rel_network(features.unsqueeze(0)).squeeze()
        
        # 训练模式下保持梯度，推理模式下返回纯数字
        if hasattr(self, 'training_mode') and self.training_mode:
            return score  # 保持梯度的张量
        else:
            return score.item()  # 转为纯数字
    
    def get_neighbors(self, entity: str) -> List[Tuple[str, str]]:
        """获取实体的所有邻居 (relation, neighbor_entity)"""
        if not self.knowledge_graph or entity not in self.knowledge_graph:
            return []
            
        neighbors = []
        for neighbor in self.knowledge_graph.neighbors(entity):
            # 获取所有边（可能有多个关系）
            edge_data = self.knowledge_graph.get_edge_data(entity, neighbor)
            for key, data in edge_data.items():
                # Handle both string and dict formats
                if isinstance(data, dict):
                    relation = data.get('relation', key)
                else:
                    # If data is a string, use it as the relation
                    relation = str(data) if data else key
                neighbors.append((relation, neighbor))
        
        return neighbors
    
    def beam_search_generate(self, 
                           question: str, 
                           start_entity: str,
                           max_steps: Optional[int] = None,
                           temperature: float = 1.0) -> List[BeamNode]:
        """
        Beam Search路径生成
        
        Args:
            question: 问题文本
            start_entity: 起始实体
            max_steps: 最大步数（实体对数）
            temperature: 温度参数，>1增加随机性
            
        Returns:
            最终beam中的所有路径节点
        """
        if max_steps is None:
            max_steps = self.max_path_length // 2
            
        # 初始化beam，训练模式下使用张量
        if self.training_mode:
            initial_log_prob = torch.tensor(0.0, requires_grad=True, device=self.device)
        else:
            initial_log_prob = 0.0
            
        current_beam = [BeamNode(
            path=[start_entity],
            score=0.0,
            log_prob=initial_log_prob,
            current_entity=start_entity
        )]
        
        # 逐步扩展
        for step in range(max_steps):
            next_candidates = []
            
            # 扩展当前beam中的每个路径
            for beam_node in current_beam:
                current_entity = beam_node.current_entity
                neighbors = self.get_neighbors(current_entity)
                
                if not neighbors:  # 死胡同
                    next_candidates.append(beam_node)
                    continue
                
                # 对每个邻居计算得分
                for relation, neighbor_entity in neighbors:
                    # 计算edge_score
                    edge_score = self.compute_edge_score(
                        question, beam_node.path, relation, neighbor_entity
                    )
                    
                    # 创建新路径
                    new_path = beam_node.path + [relation, neighbor_entity]
                    
                    # 处理score（总是用纯数字）
                    if isinstance(edge_score, torch.Tensor):
                        new_score = beam_node.score + edge_score.item()
                    else:
                        new_score = beam_node.score + edge_score
                    
                    # 处理log_prob（训练时保持梯度）
                    if self.training_mode and isinstance(edge_score, torch.Tensor):
                        # 训练模式：保持梯度流
                        if isinstance(beam_node.log_prob, torch.Tensor):
                            new_log_prob = beam_node.log_prob + edge_score
                        else:
                            # 将float转换为有梯度的张量
                            base_tensor = torch.tensor(beam_node.log_prob, 
                                                     requires_grad=True, device=self.device)
                            new_log_prob = base_tensor + edge_score
                    else:
                        # 推理模式：使用纯数字
                        if isinstance(edge_score, torch.Tensor):
                            new_log_prob = beam_node.log_prob + edge_score.item()
                        else:
                            new_log_prob = beam_node.log_prob + edge_score
                    
                    new_node = BeamNode(
                        path=new_path,
                        score=new_score,
                        log_prob=new_log_prob,
                        current_entity=neighbor_entity
                    )
                    
                    next_candidates.append(new_node)
            
            # 剪枝：保留最佳的beam_width个路径
            if self.exploration_mode and self.training:
                # 训练模式：随机采样
                current_beam = self._sample_from_candidates(next_candidates, temperature)
            else:
                # 推理模式：贪婪选择
                current_beam = self._select_top_candidates(next_candidates)
                
            if not current_beam:
                break
                
        return current_beam
    
    def _sample_from_candidates(self, candidates: List[BeamNode], temperature: float) -> List[BeamNode]:
        """训练时的随机采样剪枝"""
        if len(candidates) <= self.beam_width:
            return candidates
            
        # 计算采样概率
        scores = torch.tensor([node.score for node in candidates], dtype=torch.float)
        probs = F.softmax(scores / temperature, dim=0)
        
        # 多项式采样
        num_samples = min(self.beam_width, len(candidates))
        try:
            indices = torch.multinomial(probs, num_samples, replacement=False)
            return [candidates[i] for i in indices.tolist()]
        except RuntimeError:
            # 如果采样失败，回退到贪婪选择
            return self._select_top_candidates(candidates)
    
    def _select_top_candidates(self, candidates: List[BeamNode]) -> List[BeamNode]:
        """推理时的贪婪选择剪枝"""
        if len(candidates) <= self.beam_width:
            return candidates
            
        # 按得分降序排列，取top-k
        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates[:self.beam_width]
    
    def generate_paths(self, 
                      question: str,
                      start_entity: str, 
                      target_entities: Set[str] = None,
                      max_paths: int = 5,
                      stochastic: bool = False) -> List[Tuple[List[str], float, float]]:
        """
        生成路径的主要接口（与原有接口兼容）
        
        Returns:
            List of (path, score, log_prob) tuples
        """
        # 设置探索模式
        old_exploration_mode = self.exploration_mode
        self.exploration_mode = stochastic
        
        try:
            # 生成路径
            beam_nodes = self.beam_search_generate(
                question=question,
                start_entity=start_entity,
                temperature=1.5 if stochastic else 1.0
            )
            
            # 转换格式
            results = []
            for node in beam_nodes[:max_paths]:
                results.append((node.path, node.score, node.log_prob))
            
            return results
            
        finally:
            # 恢复探索模式
            self.exploration_mode = old_exploration_mode
    
    def generate_differentiable_paths(self, 
                                    question: str,
                                    start_entity: str, 
                                    target_entities: Set[str] = None,
                                    num_samples: int = 1,
                                    temperature: float = 1.5) -> List[Tuple[List[str], torch.Tensor]]:
        """
        生成可微分路径（用于REINFORCE训练）
        
        Returns:
            List of (path, log_prob_tensor) tuples
        """
        # 启用训练模式保持梯度
        self.exploration_mode = True
        self.training_mode = True  # 关键：启用训练模式
        self.temperature = temperature
        
        results = []
        
        for _ in range(num_samples):
            # 使用温度采样生成一条路径，现在log_prob保持梯度
            beam_nodes = self.beam_search_generate(
                question=question,
                start_entity=start_entity,
                temperature=temperature
            )
            
            if beam_nodes:
                # 随机选择一条路径（如果有多条）
                selected_node = random.choice(beam_nodes)
                path = selected_node.path
                
                # 现在selected_node.log_prob已经是有梯度的张量了！
                if isinstance(selected_node.log_prob, torch.Tensor):
                    log_prob = selected_node.log_prob
                else:
                    # 后备方案：重新计算路径概率（保持梯度）
                    log_prob = self._recompute_path_log_probability(path, question)
                
                results.append((path, log_prob))
        
        self.exploration_mode = False
        self.training_mode = False  # 重置训练模式
        return results
    
    def _recompute_path_log_probability(self, path: List[str], question: str) -> torch.Tensor:
        """
        重新计算路径的对数概率（保持梯度）
        用于后备方案
        """
        if len(path) <= 1:
            return torch.tensor(0.0, requires_grad=True, device=self.device)
        
        total_log_prob = torch.tensor(0.0, requires_grad=True, device=self.device)
        
        # 逐步重新计算每一步的概率
        for i in range(0, len(path) - 2, 2):
            current_entity = path[i]
            chosen_relation = path[i + 1]
            chosen_entity = path[i + 2]
            
            # 获取所有可能的邻居
            neighbors = self.get_neighbors(current_entity)
            if not neighbors:
                continue
            
            # 计算所有选择的分数
            scores = []
            chosen_idx = -1
            
            for j, (relation, entity) in enumerate(neighbors):
                score = self.compute_edge_score(
                    question, path[:i+1], relation, entity
                )
                scores.append(score)
                
                if relation == chosen_relation and entity == chosen_entity:
                    chosen_idx = j
            
            # 计算概率分布和选择的概率
            if chosen_idx != -1 and scores:
                scores_tensor = torch.stack(scores)
                probs = F.softmax(scores_tensor, dim=0)
                step_log_prob = torch.log(probs[chosen_idx] + 1e-10)
                total_log_prob = total_log_prob + step_log_prob
        
        return total_log_prob
    
    def persistent_query_exploration(self, 
                                   question: str, 
                                   start_entity: str,
                                   answer_entities: Set[str],
                                   max_attempts: int = 20) -> Tuple[List[str], int, bool]:
        """
        持续探索直到找到正确答案
        
        Args:
            question: 问题
            start_entity: 起始实体
            answer_entities: 正确答案实体集合
            max_attempts: 最大尝试次数
            
        Returns:
            (best_path, attempts_used, found_correct)
        """
        # print(f"[BEAM-EXPLORE] Persistent exploration for: {question[:50]}...")  # 简化输出
        
        best_path = []
        best_score = float('-inf')
        
        for attempt in range(1, max_attempts + 1):
            # print(f"   [ATTEMPT] {attempt}: Generating with temperature=1.5...")  # 简化输出
            
            # 生成路径
            beam_nodes = self.beam_search_generate(
                question=question,
                start_entity=start_entity,
                temperature=1.5  # 高温度增加探索性
            )
            
            for node in beam_nodes:
                path = node.path
                if len(path) == 0:
                    continue
                    
                final_entity = path[-1]
                is_correct = final_entity in answer_entities
                
                # 记录最佳路径
                if node.score > best_score:
                    best_score = node.score
                    best_path = path
                
                # 如果找到正确答案，立即返回
                if is_correct:
                    # print(f"   [SUCCESS] Found correct path in {attempt} attempts: {' -> '.join(path[-4:])}")  # 简化输出
                    return path, attempt, True
                else:
                    # print(f"   [RETRY] Wrong path: {' -> '.join(path[-4:])} != {list(answer_entities)}")  # 简化输出
                    pass
        
        # print(f"   [FAILED] No correct path found after {max_attempts} attempts")  # 简化输出
        # print(f"   [BEST] Returning highest-scoring path: {' -> '.join(best_path[-4:])}")  # 简化输出
        
        return best_path, max_attempts, False
    
    # ==================== 训练相关方法 ====================
    
    def enable_training_mode(self):
        """启用训练模式"""
        self.train()
        self.exploration_mode = True
        
    def disable_training_mode(self):
        """禁用训练模式"""
        self.eval()
        self.exploration_mode = False
    
    def get_trainable_parameters(self):
        """获取可训练参数"""
        return self.p_rel_network.parameters()
    
    def enable_training_mode(self):
        """启用训练模式（保持梯度）"""
        self.training_mode = True
        self.p_rel_network.train()
        # 确保网络在正确的设备上
        self.p_rel_network.to(self.device)
    
    def disable_training_mode(self):
        """禁用训练模式（推理模式）"""
        self.training_mode = False
        self.p_rel_network.eval()
    
    def compute_policy_loss(self, paths: List[List[str]], rewards: torch.Tensor) -> torch.Tensor:
        """
        计算REINFORCE策略损失
        
        Args:
            paths: 生成的路径列表  
            rewards: 对应的奖励
            
        Returns:
            策略损失
        """
        if len(paths) == 0:
            return torch.tensor(0.0, requires_grad=True).to(self.device)
            
        total_loss = 0.0
        
        for path, reward in zip(paths, rewards):
            if len(path) < 2:
                continue
                
            # 重新计算路径的log概率
            path_log_prob = 0.0
            
            for i in range(0, len(path) - 2, 2):
                current_entity = path[i]
                relation = path[i + 1]  
                next_entity = path[i + 2]
                
                # 计算这一步的log概率
                current_path = path[:i+1]
                edge_score = self.compute_edge_score("", current_path, relation, next_entity)
                path_log_prob += edge_score
                
            # REINFORCE损失：-R * log P(τ)
            policy_loss = -reward * path_log_prob
            total_loss += policy_loss
            
        return total_loss / len(paths) if paths else torch.tensor(0.0)
    
    # ==================== 兼容性方法 ====================
    
    def enable_stochastic_exploration(self):
        """启用随机探索（兼容性接口）"""
        self.exploration_mode = True
        
    def disable_stochastic_exploration(self):
        """禁用随机探索（兼容性接口）"""
        self.exploration_mode = False


# ==================== 工厂函数 ====================

def create_beam_search_generator(entity_embedding_path: str = "embeddings/entity_embeddings.pt",
                               max_path_length: int = 6,
                               beam_width: int = 5) -> BeamSearchPathGenerator:
    """创建Beam Search生成器的工厂函数"""
    generator = BeamSearchPathGenerator(
        entity_embedding_path=entity_embedding_path,
        max_path_length=max_path_length,
        beam_width=beam_width
    )
    return generator


if __name__ == "__main__":
    # 简单测试
    generator = create_beam_search_generator()
    print("BeamSearchPathGenerator created successfully!")
    
    # 测试路径生成接口
    sample_paths = generator.generate_paths(
        question="What is the capital of France?",
        start_entity="France",
        max_paths=3,
        stochastic=True
    )
    
    print(f"Generated {len(sample_paths)} sample paths")
    for i, (path, score, log_prob) in enumerate(sample_paths):
        print(f"  Path {i+1}: {' -> '.join(path)}, Score: {score:.3f}")