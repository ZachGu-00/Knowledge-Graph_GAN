"""
改进版GAA对抗训练 - 加入问题感知、关系编码和历史追踪
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import json
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Optional, Tuple
from sentence_transformers import SentenceTransformer
import pickle
from collections import deque
from tqdm import tqdm
import time


class ImprovedGNNCore(nn.Module):
    """改进的GNN - 问题感知、关系编码、历史追踪"""
    
    def __init__(self, node_dim=384, hidden_dim=512, num_layers=4, 
                 num_relations=9, dropout=0.2):
        super(ImprovedGNNCore, self).__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # 1. 问题编码器 - 将问题映射到隐藏空间
        self.question_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 2. 关系编码器 - 学习每种关系的嵌入
        self.relation_embeddings = nn.Embedding(num_relations + 1, 64)  # +1 for unknown
        self.relation_names = [
            'starred_actors', 'directed_by', 'written_by', 'has_genre',
            'has_tags', 'has_plot', 'in_language', 'release_year', 
            'has_imdb_rating', 'unknown'
        ]
        self.relation_to_idx = {name: i for i, name in enumerate(self.relation_names)}
        
        # 3. 节点变换 - 融合节点特征和问题信息
        self.node_transform = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),  # node + question
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 4. 注意力机制 - 节点对问题的相关性
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=8, dropout=dropout, batch_first=True
        )
        
        # 5. 关系感知的GNN层
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gnn_layers.append(nn.ModuleDict({
                'message': nn.Linear(hidden_dim * 2 + 64, hidden_dim),  # node + neighbor + relation
                'node_update': nn.Sequential(  # 改名避免冲突
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            }))
        
        # 6. 历史编码器 - 简化为平均池化
        self.history_pooling = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 7. 扩展掩码处理 - 标记已扩展节点
        self.expansion_mask_transform = nn.Linear(1, hidden_dim // 4)
        
        # 8. 双输出头 - 考虑历史信息
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # features + history
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.expansion_policy_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # features + history
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def encode_relations(self, kg, edges):
        """编码边的关系类型"""
        edge_types = []
        for u, v in edges:
            edge_data = kg.get_edge_data(u, v)
            if edge_data and 'relation' in edge_data:
                rel_name = edge_data['relation']
                rel_idx = self.relation_to_idx.get(rel_name, len(self.relation_names) - 1)
            else:
                rel_idx = len(self.relation_names) - 1  # unknown
            edge_types.append(rel_idx)
        
        if edge_types:
            return torch.tensor(edge_types, dtype=torch.long)
        return None
    
    def forward(self, node_features, edge_index, question_features, 
                kg=None, expansion_mask=None, history=None):
        """
        Args:
            node_features: [N, node_dim] 节点特征
            edge_index: [2, E] 边索引
            question_features: [1, node_dim] 问题特征
            kg: NetworkX图，用于获取关系类型
            expansion_mask: [N] 已扩展节点掩码 (1=已扩展, 0=未扩展)
            history: [S, hidden_dim] 扩展历史序列
        
        Returns:
            node_values: [N, 1] 节点价值分数
            expansion_policies: [N, 1] 扩展策略分数
        """
        N = node_features.size(0)
        device = node_features.device
        
        # 1. 编码问题
        q_emb = self.question_encoder(question_features)  # [1, hidden_dim]
        q_expanded = q_emb.expand(N, -1)  # [N, hidden_dim]
        
        # 2. 融合节点特征和问题信息
        node_question_combined = torch.cat([node_features, q_expanded], dim=1)  # [N, node_dim + hidden_dim]
        x = self.node_transform(node_question_combined)  # [N, hidden_dim]
        
        # 3. 节点-问题注意力
        x_attended, attention_weights = self.cross_attention(
            x.unsqueeze(0),  # [1, N, hidden_dim]
            q_emb.unsqueeze(0),  # [1, 1, hidden_dim] - 只需要一个unsqueeze
            q_emb.unsqueeze(0)   # [1, 1, hidden_dim] - 只需要一个unsqueeze
        )
        x = x + x_attended.squeeze(0)  # 残差连接
        
        # 4. 获取关系编码
        edge_embeddings = None
        if kg is not None and edge_index.size(1) > 0:
            edges = [(edge_index[0, i].item(), edge_index[1, i].item()) 
                    for i in range(edge_index.size(1))]
            edge_types = self.encode_relations(kg, edges)
            if edge_types is not None:
                edge_embeddings = self.relation_embeddings(edge_types.to(device))  # [E, 64]
        
        # 5. 关系感知的消息传递
        for layer_idx, layer in enumerate(self.gnn_layers):
            if edge_index.size(1) > 0:
                row, col = edge_index
                neighbor_features = x[col]  # [E, hidden_dim]
                source_features = x[row]    # [E, hidden_dim]
                
                # 组合特征：源节点 + 邻居节点 + 关系
                if edge_embeddings is not None:
                    combined = torch.cat([source_features, neighbor_features, edge_embeddings], dim=1)
                else:
                    # 如果没有关系编码，用零填充
                    zero_relations = torch.zeros(source_features.size(0), 64, device=device)
                    combined = torch.cat([source_features, neighbor_features, zero_relations], dim=1)
                
                # 消息传递
                messages = layer['message'](combined)  # [E, hidden_dim]
                
                # 聚合消息（按目标节点）
                aggregated = torch.zeros_like(x)
                for i in range(N):
                    mask = (row == i)
                    if mask.any():
                        aggregated[i] = messages[mask].mean(0)
                    else:
                        aggregated[i] = x[i]  # 自循环
                
                # 更新节点
                x = layer['node_update'](aggregated)
            else:
                # 无边时的自变换
                self_combined = torch.cat([x, x, torch.zeros(N, 64, device=device)], dim=1)
                x = layer['node_update'](layer['message'](self_combined))
        
        # 6. 处理扩展历史 - 使用平均池化
        history_context = torch.zeros(1, self.hidden_dim, device=device)
        if history is not None and len(history) > 0:
            # history: [S, hidden_dim] 扩展历史序列
            history_tensor = torch.stack(history)  # [S, hidden_dim]
            history_mean = history_tensor.mean(dim=0, keepdim=True)  # [1, hidden_dim]
            history_context = self.history_pooling(history_mean)  # [1, hidden_dim]
        
        history_expanded = history_context.expand(N, -1)  # [N, hidden_dim]
        
        # 7. 处理扩展掩码（已扩展节点降低expansion分数）
        mask_penalty = torch.zeros(N, self.hidden_dim // 4, device=device)
        if expansion_mask is not None:
            mask_input = expansion_mask.unsqueeze(1).float()  # [N, 1]
            mask_penalty = self.expansion_mask_transform(mask_input)  # [N, hidden_dim//4]
            # 填充到完整维度
            mask_penalty = F.pad(mask_penalty, (0, self.hidden_dim - self.hidden_dim // 4))
        
        # 8. 组合所有信息用于输出头
        final_features = torch.cat([x, history_expanded], dim=1)  # [N, hidden_dim * 2]
        
        # 9. 双头输出
        node_values = self.value_head(final_features)  # [N, 1]
        
        # 扩展策略要考虑掩码惩罚
        expansion_features = final_features - mask_penalty.repeat(1, 2)  # 降低已扩展节点分数
        expansion_policies = self.expansion_policy_head(expansion_features)  # [N, 1]
        
        # 强制已扩展节点的expansion_policy为0
        if expansion_mask is not None:
            expansion_policies = expansion_policies * (1 - expansion_mask.unsqueeze(1))
        
        return node_values, expansion_policies


class ImprovedIterativeTrainer:
    """改进的迭代训练器 - 问题感知、关系编码、历史追踪"""
    
    def __init__(self, kg: nx.Graph, discriminator_path: str, device: str = 'cuda'):
        self.kg = kg
        self.device = device
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 改进的GNN
        self.gnn = ImprovedGNNCore(
            node_dim=384, 
            hidden_dim=512, 
            num_layers=4, 
            num_relations=9,
            dropout=0.2
        ).to(device)
        
        # 加载判别器（简化版用于测试）
        self.discriminator = self.load_dummy_discriminator()
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.gnn.parameters(), 
            lr=0.001, 
            weight_decay=1e-5
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        # 经验回放池 - 用于判别器训练
        self.positive_samples = []  # 正确路径
        self.negative_samples = []  # 判别器给高分但错误的路径
        
        # 训练统计
        self.training_stats = {
            'questions_attempted': 0,
            'total_attempts': 0,
            'successful_questions': 0,
            'discriminator_calls': 0,
            'question_details': []
        }
    
    def load_dummy_discriminator(self):
        """加载测试用判别器"""
        class DummyDiscriminator:
            def forward_single_path(self, query, path_str):
                import random
                return {
                    'path_score': random.random(),
                    'confidence': random.random() * 0.5
                }
        return DummyDiscriminator()
    
    def encode_question(self, question: str) -> torch.Tensor:
        """编码问题文本"""
        with torch.no_grad():
            q_emb = self.sentence_model.encode([question])
        return torch.FloatTensor(q_emb).to(self.device)
    
    def encode_nodes(self, nodes: List[str]) -> torch.Tensor:
        """编码节点文本"""
        node_texts = [str(node).replace('_', ' ') for node in nodes]
        with torch.no_grad():
            embeddings = self.sentence_model.encode(node_texts)
        return torch.FloatTensor(embeddings).to(self.device)
    
    def get_1hop_neighbors(self, entity: str) -> Set[str]:
        """获取1跳邻居"""
        if entity not in self.kg:
            return {entity}
        neighbors = set(self.kg.neighbors(entity))
        neighbors.add(entity)
        return neighbors
    
    def create_edge_index(self, nodes: Set[str]) -> torch.Tensor:
        """创建边索引"""
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        edges = []
        
        subgraph = self.kg.subgraph(nodes)
        for u, v in subgraph.edges():
            if u in node_to_idx and v in node_to_idx:
                edges.append([node_to_idx[u], node_to_idx[v]])
                edges.append([node_to_idx[v], node_to_idx[u]])
        
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            
        return edge_index.to(self.device)
    
    def find_shortest_paths(self, start: str, targets: List[str]) -> Dict[str, List[str]]:
        """找最短路径"""
        paths = {}
        for target in targets:
            if target in self.kg:
                try:
                    path = nx.shortest_path(self.kg, start, target)
                    formatted_path = []
                    for i in range(len(path) - 1):
                        formatted_path.append(path[i])
                        edge_data = self.kg.get_edge_data(path[i], path[i+1])
                        relation = edge_data.get('relation', 'unknown') if edge_data else 'unknown'
                        formatted_path.append(relation)
                    formatted_path.append(path[-1])
                    paths[target] = formatted_path
                except:
                    continue
        return paths
    
    def attempt_question_improved(self, question_data: Dict) -> Dict:
        """改进的问题尝试 - 使用问题感知和历史追踪"""
        question = question_data['question']
        question_entity = question_data.get('question_entity', '')
        true_answers = question_data.get('answer_entities', [])
        
        # 编码问题
        question_features = self.encode_question(question)
        
        attempt_log = {
            'question_id': question_data.get('id', f'q_{self.training_stats["questions_attempted"]}'),
            'attempts': [],
            'final_result': 'failed',
            'total_attempts': 0
        }
        
        if not question_entity or not true_answers:
            return attempt_log
        
        attempt_count = 0
        max_attempts = 20
        expansion_history = []  # 追踪扩展历史
        
        while attempt_count < max_attempts:
            attempt_count += 1
            self.training_stats['total_attempts'] += 1
            
            attempt_result = {
                'attempt_number': attempt_count,
                'expansion_steps': [],
                'final_predictions': {},
                'correct_answer_found': False
            }
            
            # 初始化子图和扩展状态
            current_subgraph = {question_entity}
            expanded_nodes = set()
            step_history = []
            
            # 多步扩展，使用改进的GNN
            for expansion_step in range(3):  # 最多3步扩展
                # 获取当前子图
                if len(current_subgraph) < 2:
                    base_neighbors = self.get_1hop_neighbors(question_entity)
                    current_subgraph.update(base_neighbors)
                
                node_list = list(current_subgraph)
                node_features = self.encode_nodes(node_list)
                edge_index = self.create_edge_index(current_subgraph)
                
                # 创建扩展掩码
                expansion_mask = torch.zeros(len(node_list), device=self.device)
                for i, node in enumerate(node_list):
                    if node in expanded_nodes:
                        expansion_mask[i] = 1.0
                
                # 准备历史张量
                history_tensor = step_history if step_history else None
                
                # GNN前向传播 - 使用改进的架构
                self.gnn.train()
                node_values, expansion_policies = self.gnn(
                    node_features, 
                    edge_index, 
                    question_features,
                    kg=self.kg,
                    expansion_mask=expansion_mask,
                    history=history_tensor
                )
                
                # 记录预测
                step_predictions = {}
                for i, node in enumerate(node_list):
                    step_predictions[node] = {
                        'value_score': float(node_values[i].item()),
                        'expansion_score': float(expansion_policies[i].item())
                    }
                
                # 选择扩展节点（基于expansion_policy）
                expansion_candidates = []
                for i, node in enumerate(node_list):
                    if expansion_mask[i] == 0:  # 未扩展
                        expansion_candidates.append((expansion_policies[i].item(), i, node))
                
                if not expansion_candidates:
                    break
                
                # 选择最佳扩展节点
                expansion_candidates.sort(reverse=True)
                best_score, best_idx, best_node = expansion_candidates[0]
                
                # 扩展该节点
                new_neighbors = self.get_1hop_neighbors(best_node)
                newly_added = new_neighbors - current_subgraph
                current_subgraph.update(new_neighbors)
                expanded_nodes.add(best_node)
                
                # 更新历史 - 需要将384维转换为512维
                # 使用GNN的中间表示作为历史
                node_hidden = node_values.new_zeros(self.gnn.hidden_dim)  # 创建512维向量
                # 简单处理：填充前384维
                node_hidden[:node_features.size(1)] = node_features[best_idx]
                step_history.append(node_hidden)
                
                attempt_result['expansion_steps'].append({
                    'step': expansion_step + 1,
                    'expanded_node': best_node,
                    'expansion_score': best_score,
                    'new_neighbors': len(newly_added),
                    'predictions': step_predictions
                })
                
                if len(newly_added) == 0:
                    break
            
            # 最终预测
            final_node_list = list(current_subgraph)
            final_node_features = self.encode_nodes(final_node_list)
            final_edge_index = self.create_edge_index(current_subgraph)
            
            final_expansion_mask = torch.zeros(len(final_node_list), device=self.device)
            for i, node in enumerate(final_node_list):
                if node in expanded_nodes:
                    final_expansion_mask[i] = 1.0
            
            final_values, _ = self.gnn(
                final_node_features,
                final_edge_index,
                question_features,
                kg=self.kg,
                expansion_mask=final_expansion_mask,
                history=step_history if step_history else None
            )
            
            # 选择最佳答案
            value_scores = [(final_values[i].item(), final_node_list[i]) 
                          for i in range(len(final_node_list))]
            value_scores.sort(reverse=True)
            
            best_candidate = value_scores[0][1] if value_scores else None
            
            # 检查是否找到正确答案
            if best_candidate:
                is_correct = any(ans.lower() in best_candidate.lower() or 
                               best_candidate.lower() in ans.lower() 
                               for ans in true_answers)
                
                if is_correct:
                    attempt_result['correct_answer_found'] = True
                    attempt_log['final_result'] = 'success'
                    attempt_log['attempts'].append(attempt_result)
                    
                    # 保存成功路径
                    self.successful_paths.append({
                        'question': question,
                        'path': attempt_result['expansion_steps'],
                        'answer': best_candidate
                    })
                    break
            
            # 计算损失并更新
            self.compute_and_update_loss(
                final_node_list, final_values, true_answers, 
                expansion_history=attempt_result['expansion_steps']
            )
            
            attempt_log['attempts'].append(attempt_result)
        
        attempt_log['total_attempts'] = attempt_count
        
        if attempt_log['final_result'] == 'success':
            self.training_stats['successful_questions'] += 1
        
        return attempt_log
    
    def compute_and_update_loss(self, node_list, node_values, true_answers, expansion_history=None):
        """计算并更新损失 - 改进expansion_policy监督信号"""
        # value head目标：最终答案
        value_targets = []
        for node in node_list:
            is_answer = any(ans.lower() in node.lower() or node.lower() in ans.lower() 
                           for ans in true_answers)
            value_targets.append(1.0 if is_answer else 0.0)
        
        # expansion policy目标：如果有成功路径，中间节点应该得高分
        expansion_targets = value_targets.copy()  # 默认与value相同
        if hasattr(self, 'positive_samples') and self.positive_samples:
            # 从成功案例学习哪些节点应该扩展
            last_sample = self.positive_samples[-1] if self.positive_samples else None
            if last_sample and 'intermediate_nodes' in last_sample:
                for i, node in enumerate(node_list):
                    if node in last_sample['intermediate_nodes']:
                        expansion_targets[i] = 1.0  # 中间节点应该被扩展
        
        value_target_tensor = torch.FloatTensor(value_targets).to(self.device).unsqueeze(1)
        expansion_target_tensor = torch.FloatTensor(expansion_targets).to(self.device).unsqueeze(1)
        
        # 答案损失 - 使用正确的目标张量
        answer_loss = F.binary_cross_entropy(node_values, value_target_tensor)
        
        # 排序损失（确保正确答案排名更高）
        positive_mask = value_target_tensor.squeeze() > 0.5
        negative_mask = ~positive_mask
        
        ranking_loss = 0
        if positive_mask.any() and negative_mask.any():
            pos_scores = node_values[positive_mask]
            neg_scores = node_values[negative_mask]
            
            # 边界损失
            for pos_score in pos_scores:
                for neg_score in neg_scores:
                    ranking_loss += F.relu(0.2 - (pos_score - neg_score))
            
            ranking_loss = ranking_loss / (len(pos_scores) * len(neg_scores))
        
        # 总损失
        total_loss = answer_loss + 0.5 * ranking_loss
        
        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.gnn.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return total_loss.item()
    
    def run_improved_training(self, qa_data: List[Dict], samples_per_epoch: int = 400) -> Dict:
        """运行改进的训练"""
        print("="*80)
        print("Starting Improved Iterative Training")
        print("Features: Question-aware, Relation encoding, History tracking")
        print(f"Samples per epoch: {samples_per_epoch}")
        print("="*80)
        
        # 按hop类型分组数据
        hop1_questions = [q for q in qa_data if q.get('type', '').startswith('1hop')]
        hop2_questions = [q for q in qa_data if q.get('type', '').startswith('2hop')]
        hop3_questions = [q for q in qa_data if q.get('type', '').startswith('3hop')]
        
        print(f"Available data:")
        print(f"  1-hop: {len(hop1_questions)} questions")
        print(f"  2-hop: {len(hop2_questions)} questions")
        print(f"  3-hop: {len(hop3_questions)} questions")
        print()
        
        # 训练结果汇总
        all_results = {}
        
        # 计算总体进度
        total_questions_to_train = 0
        hop_configs = []
        for hop_type, hop_questions in [('1-hop', hop1_questions), ('2-hop', hop2_questions), ('3-hop', hop3_questions)]:
            if hop_questions:
                num_samples = min(samples_per_epoch, len(hop_questions))
                total_questions_to_train += num_samples
                hop_configs.append((hop_type, hop_questions, num_samples))
        
        print(f"\nTotal questions to train: {total_questions_to_train}")
        overall_start_time = time.time()
        questions_processed = 0
        
        # 按难度递进训练：先1-hop，再2-hop，最后3-hop
        for hop_type, hop_questions, num_samples in sorted(hop_configs, key=lambda x: int(x[0][0])):
            print(f"\n{'='*60}")
            print(f"Training on {hop_type} questions")
            print(f"Samples: {num_samples}/{len(hop_questions)} available")
            print(f"Overall progress: {questions_processed}/{total_questions_to_train} ({questions_processed/total_questions_to_train*100:.1f}%)")
            elapsed = time.time() - overall_start_time
            if questions_processed > 0:
                eta = elapsed / questions_processed * (total_questions_to_train - questions_processed)
                print(f"Estimated time remaining: {eta/60:.1f} minutes")
            print(f"{'='*60}")
            
            # 选择samples_per_epoch个样本作为一个epoch
            epoch_questions = random.sample(hop_questions, min(samples_per_epoch, len(hop_questions)))
            
            # 训练统计
            hop_stats = {
                'total_questions': len(epoch_questions),
                'successful': 0,
                'total_attempts': 0,
                'discriminator_calls': 0,
                'discriminator_errors': 0,  # 高分但错误的情况
                'subgraph_sizes': [],
                'expansion_steps': [],
                'attempt_distribution': {}
            }
            
            # 批量训练 - 使用进度条
            start_time = time.time()
            with tqdm(total=len(epoch_questions), 
                     desc=f"{hop_type} Training", 
                     unit="questions",
                     ncols=100,
                     bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
                
                for i, question_data in enumerate(epoch_questions):
                    self.training_stats['questions_attempted'] += 1
                    
                    # 简化的尝试日志，不保存详细信息
                    attempt_result = self.attempt_question_simplified(question_data)
                    
                    # 更新统计
                    if attempt_result['success']:
                        hop_stats['successful'] += 1
                    hop_stats['total_attempts'] += attempt_result['attempts']
                    hop_stats['subgraph_sizes'].append(attempt_result['final_subgraph_size'])
                    hop_stats['expansion_steps'].append(attempt_result['expansion_count'])
                    
                    # 记录尝试次数分布
                    attempts_bucket = min(attempt_result['attempts'] // 5 * 5, 20)  # 0-5, 5-10, 10-15, 15-20, 20+
                    hop_stats['attempt_distribution'][f"{attempts_bucket}-{attempts_bucket+5}"] = \
                        hop_stats['attempt_distribution'].get(f"{attempts_bucket}-{attempts_bucket+5}", 0) + 1
                    
                    # 判别器相关统计
                    hop_stats['discriminator_calls'] += attempt_result.get('disc_calls', 0)
                    hop_stats['discriminator_errors'] += attempt_result.get('disc_errors', 0)
                    
                    # 更新进度条描述
                    current_success_rate = hop_stats['successful'] / (i + 1) if i > 0 else 0
                    pbar.set_postfix({
                        'Success': f"{current_success_rate:.1%}",
                        'Avg_Attempts': f"{hop_stats['total_attempts']/(i+1):.1f}"
                    })
                    pbar.update(1)
            
            elapsed_time = time.time() - start_time
            questions_processed += len(epoch_questions)
            print(f"  Training time: {elapsed_time:.1f} seconds ({elapsed_time/len(epoch_questions):.2f}s per question)")
            
            # 计算汇总指标
            hop_summary = {
                'hop_type': hop_type,
                'total_questions': hop_stats['total_questions'],
                'success_rate': hop_stats['successful'] / hop_stats['total_questions'],
                'avg_attempts': hop_stats['total_attempts'] / hop_stats['total_questions'],
                'avg_subgraph_size': np.mean(hop_stats['subgraph_sizes']),
                'avg_expansion_steps': np.mean(hop_stats['expansion_steps']),
                'discriminator_accuracy': 1 - (hop_stats['discriminator_errors'] / max(hop_stats['discriminator_calls'], 1)),
                'attempt_distribution': hop_stats['attempt_distribution']
            }
            
            all_results[hop_type] = hop_summary
            
            # 打印汇总
            print(f"\n{hop_type} Training Summary:")
            print(f"  Success rate: {hop_summary['success_rate']:.1%}")
            print(f"  Avg attempts: {hop_summary['avg_attempts']:.1f}")
            print(f"  Avg subgraph size: {hop_summary['avg_subgraph_size']:.0f} nodes")
            print(f"  Avg expansion steps: {hop_summary['avg_expansion_steps']:.1f}")
            print(f"  Discriminator accuracy: {hop_summary['discriminator_accuracy']:.1%}")
            print(f"  Positive samples collected: {len(self.positive_samples)}")
            print(f"  Negative samples collected: {len(self.negative_samples)}")
            
            # 每个hop类型训练完后，用收集的样本训练判别器适配层
            if len(self.positive_samples) > 10 and len(self.negative_samples) > 10:
                print(f"\n  Training discriminator adapter with {len(self.positive_samples)} positive and {len(self.negative_samples)} negative samples...")
                self.train_discriminator_adapter()
                # 清空样本池
                self.positive_samples = []
                self.negative_samples = []
        
        # 整体训练汇总
        overall_summary = {
            'session_time': datetime.now().isoformat(),
            'training_mode': 'hop_separated_training',
            'samples_per_epoch': samples_per_epoch,
            'model_params': sum(p.numel() for p in self.gnn.parameters()),
            'hop_results': all_results,
            'overall_metrics': {
                'total_questions': sum(r['total_questions'] for r in all_results.values()),
                'avg_success_rate': np.mean([r['success_rate'] for r in all_results.values()]),
                'avg_attempts': np.mean([r['avg_attempts'] for r in all_results.values()]),
                'avg_subgraph_size': np.mean([r['avg_subgraph_size'] for r in all_results.values()]),
                'avg_discriminator_accuracy': np.mean([r['discriminator_accuracy'] for r in all_results.values()])
            }
        }
        
        # 确保保存：1. 汇总日志 2. 模型检查点
        self.save_summary_log(overall_summary)
        self.save_model_checkpoint(overall_summary)
        
        print("\n" + "="*80)
        print("Training artifacts saved:")
        print("  - Training summary JSON in training_logs/")
        print("  - Generator checkpoint in checkpoints/improved_gnn/")
        print("  - Discriminator adapter in checkpoints/improved_gnn/")
        print("="*80)
        
        return overall_summary
    
    def attempt_question_simplified(self, question_data: Dict) -> Dict:
        """简化的问题尝试 - 只返回统计信息"""
        question = question_data['question']
        question_entity = question_data.get('question_entity', '')
        true_answers = question_data.get('answer_entities', [])
        
        result = {
            'success': False,
            'attempts': 0,
            'final_subgraph_size': 0,
            'expansion_count': 0,
            'disc_calls': 0,
            'disc_errors': 0
        }
        
        if not question_entity or not true_answers:
            return result
        
        question_features = self.encode_question(question)
        max_attempts = 20
        
        for attempt in range(max_attempts):
            result['attempts'] += 1
            
            # 执行扩展和预测（简化版）
            current_subgraph = {question_entity}
            expanded_nodes = set()
            expansion_count = 0
            
            # 多步扩展
            for step in range(3):
                if len(current_subgraph) < 2:
                    current_subgraph.update(self.get_1hop_neighbors(question_entity))
                
                node_list = list(current_subgraph)
                node_features = self.encode_nodes(node_list)
                edge_index = self.create_edge_index(current_subgraph)
                
                expansion_mask = torch.zeros(len(node_list), device=self.device)
                for i, node in enumerate(node_list):
                    if node in expanded_nodes:
                        expansion_mask[i] = 1.0
                
                self.gnn.train()
                node_values, expansion_policies = self.gnn(
                    node_features, edge_index, question_features,
                    kg=self.kg, expansion_mask=expansion_mask
                )
                
                # 选择扩展节点
                best_expand = None
                best_score = -1
                for i, node in enumerate(node_list):
                    if expansion_mask[i] == 0 and expansion_policies[i].item() > best_score:
                        best_score = expansion_policies[i].item()
                        best_expand = node
                
                if best_expand:
                    new_neighbors = self.get_1hop_neighbors(best_expand)
                    current_subgraph.update(new_neighbors)
                    expanded_nodes.add(best_expand)
                    expansion_count += 1
                else:
                    break
            
            result['final_subgraph_size'] = len(current_subgraph)
            result['expansion_count'] = expansion_count
            
            # 最终预测 - 需要重新计算以确保尺寸匹配
            final_node_list = list(current_subgraph)
            final_node_features = self.encode_nodes(final_node_list)
            final_edge_index = self.create_edge_index(current_subgraph)
            
            final_mask = torch.zeros(len(final_node_list), device=self.device)
            for i, node in enumerate(final_node_list):
                if node in expanded_nodes:
                    final_mask[i] = 1.0
            
            # 重新前向传播获取最终预测
            final_values, _ = self.gnn(
                final_node_features, final_edge_index, question_features,
                kg=self.kg, expansion_mask=final_mask
            )
            
            # 选择答案
            best_idx = torch.argmax(final_values).item()
            best_candidate = final_node_list[best_idx] if final_node_list else None
            
            if best_candidate:
                is_correct = any(ans.lower() in best_candidate.lower() or 
                               best_candidate.lower() in ans.lower() 
                               for ans in true_answers)
                
                # 判别器检查
                result['disc_calls'] += 1
                disc_result = self.discriminator.forward_single_path(question, best_candidate)
                disc_score = disc_result.get('path_score', 0.5)
                
                # 收集样本用于判别器训练
                if is_correct:
                    # 正样本：正确的路径（包含中间节点作为扩展策略的正例）
                    self.positive_samples.append({
                        'question': question,
                        'path': best_candidate,
                        'score': disc_score,
                        'intermediate_nodes': list(expanded_nodes)  # 中间节点用于训练expansion_policy
                    })
                    result['success'] = True
                    break
                elif disc_score > 0.7:
                    # 负样本：判别器给高分但错误
                    self.negative_samples.append({
                        'question': question,
                        'path': best_candidate,
                        'score': disc_score
                    })
                    result['disc_errors'] += 1
            
            # 每个query都更新GNN参数
            loss = self.compute_and_update_loss(final_node_list, final_values, true_answers)
            result['loss'] = loss
        
        return result
    
    def train_discriminator_adapter(self):
        """完整实现判别器适配层训练"""
        if not hasattr(self, 'disc_adapter'):
            # 创建判别器适配层 - 外接参数矩阵
            self.disc_adapter = nn.Sequential(
                nn.Linear(384, 128),  # 输入是path embedding
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Sigmoid()
            ).to(self.device)
            self.disc_optimizer = torch.optim.Adam(self.disc_adapter.parameters(), lr=0.001)
        
        num_updates = min(100, len(self.positive_samples), len(self.negative_samples))
        total_loss = 0
        
        for _ in range(num_updates):
            # 采样正负样本
            pos_sample = random.choice(self.positive_samples)
            neg_sample = random.choice(self.negative_samples)
            
            # 编码路径为特征（简化：使用问题和答案的编码）
            pos_features = self.encode_nodes([pos_sample['path']])[0]
            neg_features = self.encode_nodes([neg_sample['path']])[0]
            
            # 适配层预测
            pos_score = self.disc_adapter(pos_features)
            neg_score = self.disc_adapter(neg_features)
            
            # 二分类损失
            pos_target = torch.ones_like(pos_score)
            neg_target = torch.zeros_like(neg_score)
            
            pos_loss = F.binary_cross_entropy(pos_score, pos_target)
            neg_loss = F.binary_cross_entropy(neg_score, neg_target)
            loss = pos_loss + neg_loss
            
            # 反向传播
            self.disc_optimizer.zero_grad()
            loss.backward()
            self.disc_optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / num_updates if num_updates > 0 else 0
        accuracy = self.evaluate_discriminator_adapter()
        print(f"    Discriminator adapter training: Loss={avg_loss:.4f}, Accuracy={accuracy:.1%}")
        
        # 保存适配层
        checkpoint_dir = Path("checkpoints/improved_gnn")
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        disc_path = checkpoint_dir / f"disc_adapter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        torch.save({
            'adapter_state_dict': self.disc_adapter.state_dict(),
            'optimizer_state_dict': self.disc_optimizer.state_dict(),
            'samples': {'positive': len(self.positive_samples), 'negative': len(self.negative_samples)},
            'accuracy': accuracy
        }, disc_path)
    
    def evaluate_discriminator_adapter(self):
        """评估判别器适配层准确率"""
        if not hasattr(self, 'disc_adapter'):
            return 0
        
        correct = 0
        total = 0
        
        # 测试正样本
        for sample in self.positive_samples[:20]:
            features = self.encode_nodes([sample['path']])[0]
            score = self.disc_adapter(features).item()
            if score > 0.5:
                correct += 1
            total += 1
        
        # 测试负样本
        for sample in self.negative_samples[:20]:
            features = self.encode_nodes([sample['path']])[0]
            score = self.disc_adapter(features).item()
            if score < 0.5:
                correct += 1
            total += 1
        
        return correct / total if total > 0 else 0
    
    def save_model_checkpoint(self, summary: Dict):
        """保存生成器模型检查点"""
        checkpoint_dir = Path("checkpoints/improved_gnn")
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存生成器
        generator_path = checkpoint_dir / f"generator_{timestamp}.pth"
        torch.save({
            'model_state_dict': self.gnn.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_params': sum(p.numel() for p in self.gnn.parameters()),
            'training_summary': summary,
            'timestamp': timestamp
        }, generator_path)
        
        print(f"\nGenerator checkpoint saved to: {generator_path}")
        
        # 如果有判别器适配层，也保存
        if hasattr(self, 'disc_adapter'):
            disc_path = checkpoint_dir / f"disc_adapter_final_{timestamp}.pth"
            torch.save({
                'adapter_state_dict': self.disc_adapter.state_dict(),
                'optimizer_state_dict': self.disc_optimizer.state_dict()
            }, disc_path)
            print(f"Discriminator adapter saved to: {disc_path}")
    
    def save_summary_log(self, summary: Dict):
        """保存简化的汇总日志 - 只保存总体统计，不保存每个问题的细节"""
        log_dir = Path("training_logs")
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"training_summary_{timestamp}.json"
        
        def convert_for_json(obj):
            if isinstance(obj, dict):
                return {key: convert_for_json(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            elif hasattr(obj, 'item'):
                return obj.item()
            elif str(type(obj).__name__).startswith('int'):
                return int(obj)
            elif str(type(obj).__name__).startswith('float'):
                return float(obj)
            else:
                return obj
        
        summary_converted = convert_for_json(summary)
        
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(summary_converted, f, indent=2, ensure_ascii=False)
        
        print(f"Training summary saved to: {log_path}")


def main():
    print("Improved GAA Training System")
    print("Question-aware, Relation-aware, History-tracking")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print()
    
    try:
        # 1. 加载数据
        print("1. Loading data...")
        with open('graph/knowledge_graph.pkl', 'rb') as f:
            kg = pickle.load(f)
        print(f"  Knowledge graph: {kg.number_of_nodes()} nodes, {kg.number_of_edges()} edges")
        
        with open('query/qa_with_paths_cleaned.json', 'r', encoding='utf-8') as f:
            content = f.read().strip()
            qa_entries = content.split('\n\n')
            all_data = []
            for entry in qa_entries:
                if entry.strip():
                    try:
                        data = json.loads(entry.strip())
                        all_data.append(data)
                    except:
                        continue
        print(f"  QA data: {len(all_data)} samples")
        print()
        
        # 2. 初始化改进的训练器
        print("2. Initializing Improved Trainer...")
        discriminator_path = "checkpoints/enhanced_pathranker/best_hits1_model.pth"
        trainer = ImprovedIterativeTrainer(kg, discriminator_path, device)
        
        total_params = sum(p.numel() for p in trainer.gnn.parameters())
        print(f"  Improved GNN Parameters: {total_params:,} ({total_params/1000:.1f}K)")
        print(f"  Features: Question encoding, Relation encoding, History tracking")
        print()
        
        # 3. 运行改进的训练
        print("3. Starting improved training...")
        print("Training configuration: 400 samples per hop type")
        training_session = trainer.run_improved_training(
            qa_data=all_data,
            samples_per_epoch=400
        )
        
        # 4. 打印结果
        print("="*80)
        print("Training Complete!")
        print("="*80)
        
        if 'overall_metrics' in training_session:
            metrics = training_session['overall_metrics']
            print(f"\nOverall Performance:")
            print(f"  Total questions: {metrics['total_questions']}")
            print(f"  Avg success rate: {metrics['avg_success_rate']:.1%}")
            print(f"  Avg attempts: {metrics['avg_attempts']:.1f}")
            print(f"  Avg subgraph size: {metrics['avg_subgraph_size']:.0f} nodes")
            print(f"  Avg discriminator accuracy: {metrics['avg_discriminator_accuracy']:.1%}")
        
        print("\nDetailed results by hop type:")
        for hop_type, results in training_session.get('hop_results', {}).items():
            print(f"\n{hop_type}:")
            print(f"  Success rate: {results['success_rate']:.1%}")
            print(f"  Avg attempts: {results['avg_attempts']:.1f}")
            print(f"  Avg subgraph: {results['avg_subgraph_size']:.0f} nodes")
        
        print("\nTraining completed successfully!")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()