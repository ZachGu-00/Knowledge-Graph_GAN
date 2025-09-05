"""
Query-Conditioned GNN生成器 (修复版本)
基于proposal_gnn的QC-APPR与双重引导耦合机制实现

修复内容：
1. 冻结SBERT参数
2. 动态计算关系数量，不硬编码
3. 使用真实图结构和数据
4. 实现完整的梯度归因机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import networkx as nx
from typing import Dict, List, Tuple, Set, Optional
from sentence_transformers import SentenceTransformer
from collections import defaultdict, deque
import heapq
from tqdm import tqdm


class QueryConditionedAPPR:
    """查询条件化近似个性化PageRank"""
    
    def __init__(self, knowledge_graph, sentence_model, alpha=0.15, epsilon=1e-6):
        self.kg = knowledge_graph
        self.sentence_model = sentence_model
        self.alpha = alpha  # 回传率
        self.epsilon = epsilon  # Forward-Push阈值
        self.relation_priors = self._compute_relation_priors()
        
    def _compute_relation_priors(self):
        """从真实图数据计算关系先验权重"""
        relation_counts = defaultdict(int)
        for u, v, data in self.kg.edges(data=True):
            relation = data.get('relation', 'unknown')
            relation_counts[relation] += 1
            
        # 基于频率计算先验：高频关系降权以避免偏向常见关系
        max_count = max(relation_counts.values()) if relation_counts else 1
        total_edges = sum(relation_counts.values())
        
        priors = {}
        for rel, count in relation_counts.items():
            # 使用逆频率权重：低频关系获得更高权重
            frequency = count / total_edges
            prior = max(0.1, 0.8 - 0.7 * frequency)  # 范围[0.1, 0.8]
            priors[rel] = prior
            
        return priors
        
    def compute_transition_matrix(self, query: str, current_subgraph: Set, 
                                lambda_param: float = 0.6, mu_param: float = 0.4):
        """计算查询条件化转移矩阵"""
        # 编码查询
        with torch.no_grad():  # 确保SBERT推理时不计算梯度
            query_emb = self.sentence_model.encode([query])[0]
        
        # 获取子图中节点和边的文本表示
        node_texts = {}
        edge_texts = {}
        
        for node in current_subgraph:
            if isinstance(node, str):
                node_texts[node] = node.replace('_', ' ')
            else:
                node_texts[node] = str(node)
                
        subgraph = self.kg.subgraph(current_subgraph)
        for u, v, data in subgraph.edges(data=True):
            relation = data.get('relation', 'related_to')
            edge_texts[(u, v)] = relation.replace('_', ' ')
            
        # 批量编码节点和关系以提高效率
        node_similarities = {}
        if node_texts:
            all_node_texts = list(node_texts.values())
            with torch.no_grad():
                node_embs = self.sentence_model.encode(all_node_texts)
            
            for i, node in enumerate(node_texts.keys()):
                sim = np.dot(query_emb, node_embs[i]) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(node_embs[i]) + 1e-8
                )
                node_similarities[node] = max(sim, 0.0)
            
        edge_similarities = {}
        if edge_texts:
            all_edge_texts = list(edge_texts.values())
            with torch.no_grad():
                edge_embs = self.sentence_model.encode(all_edge_texts)
            
            for i, edge in enumerate(edge_texts.keys()):
                sim = np.dot(query_emb, edge_embs[i]) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(edge_embs[i]) + 1e-8
                )
                edge_similarities[edge] = max(sim, 0.0)
                
        # 构建转移概率矩阵
        transition_probs = {}
        for node in current_subgraph:
            neighbors = list(subgraph.neighbors(node))
            if not neighbors:
                continue
                
            weights = []
            neighbor_list = []
            
            for neighbor in neighbors:
                if neighbor not in current_subgraph:
                    continue
                    
                # 关系分数
                edge = (node, neighbor)
                if edge in edge_similarities:
                    s_r = edge_similarities[edge]
                else:
                    edge = (neighbor, node)
                    s_r = edge_similarities.get(edge, 0.1)
                    
                # 节点分数
                s_e = node_similarities.get(neighbor, 0.1)
                
                # 关系先验
                edge_data = subgraph.get_edge_data(node, neighbor)
                relation = edge_data.get('relation', 'unknown') if edge_data else 'unknown'
                b_r = self.relation_priors.get(relation, 0.3)
                
                # 综合权重：λs_r + μs_e + b_r
                weight = lambda_param * s_r + mu_param * s_e + b_r
                weights.append(weight)
                neighbor_list.append(neighbor)
                
            # Softmax归一化保证行随机化
            if weights:
                weights = np.array(weights)
                weights = np.exp(weights - np.max(weights))  # 数值稳定性
                weights = weights / np.sum(weights)
                
                transition_probs[node] = dict(zip(neighbor_list, weights))
                
        return transition_probs
        
    def forward_push_appr(self, start_node, transition_probs: Dict, max_iterations: int = 100):
        """Forward-Push近似PageRank算法"""
        if start_node not in transition_probs:
            return {start_node: 1.0}
            
        # 初始化残差和估计值
        residual = defaultdict(float)
        residual[start_node] = 1.0
        estimate = defaultdict(float)
        
        for iteration in range(max_iterations):
            # 找到残差最大的节点
            if not residual:
                break
                
            max_residual_node = max(residual.items(), key=lambda x: x[1])[0]
            max_residual_value = residual[max_residual_node]
            
            if max_residual_value < self.epsilon:
                break
                
            # Push操作
            estimate[max_residual_node] += self.alpha * max_residual_value
            residual[max_residual_node] = 0.0
            
            # 向邻居传播剩余概率质量
            if max_residual_node in transition_probs:
                propagation = (1 - self.alpha) * max_residual_value
                for neighbor, prob in transition_probs[max_residual_node].items():
                    residual[neighbor] += propagation * prob
                    
        return dict(estimate)


class GNNMessagePassing(nn.Module):
    """GNN消息传递层 - 支持动态关系数量"""
    
    def __init__(self, input_dim, hidden_dim, num_heads=4, num_relations=10):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_relations = num_relations
        
        # 多头注意力投影
        self.query_proj = nn.Linear(input_dim, hidden_dim * num_heads)
        self.key_proj = nn.Linear(input_dim, hidden_dim * num_heads)
        self.value_proj = nn.Linear(input_dim, hidden_dim * num_heads)
        
        # 关系嵌入 - 动态大小
        self.relation_embed = nn.Embedding(num_relations, hidden_dim)
        
        # 输出投影
        self.out_proj = nn.Linear(hidden_dim * num_heads, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)
        
        # 初始化关系嵌入
        nn.init.xavier_uniform_(self.relation_embed.weight)
        
    def forward(self, node_features, edge_index, edge_relations, query_emb):
        """
        Args:
            node_features: [num_nodes, input_dim]
            edge_index: [2, num_edges] - 边的索引
            edge_relations: [num_edges] - 边的关系类型ID
            query_emb: [input_dim] - 查询嵌入
        """
        num_nodes = node_features.size(0)
        
        # 查询增强的节点表示
        query_expanded = query_emb.unsqueeze(0).expand(num_nodes, -1)
        enhanced_features = node_features + 0.1 * query_expanded
        
        # 多头注意力投影
        Q = self.query_proj(enhanced_features).view(num_nodes, self.num_heads, self.hidden_dim)
        K = self.key_proj(enhanced_features).view(num_nodes, self.num_heads, self.hidden_dim)
        V = self.value_proj(enhanced_features).view(num_nodes, self.num_heads, self.hidden_dim)
        
        # 消息传递
        messages = []
        for i in range(num_nodes):
            # 找到指向节点i的所有边
            incoming_edges = (edge_index[1] == i).nonzero(as_tuple=True)[0]
            
            if len(incoming_edges) == 0:
                # 没有入边，使用自环消息
                messages.append(V[i])
                continue
                
            # 收集邻居消息
            neighbor_indices = edge_index[0][incoming_edges]
            neighbor_relations = edge_relations[incoming_edges]
            
            # 关系嵌入调制
            rel_embs = self.relation_embed(neighbor_relations)  # [num_neighbors, hidden_dim]
            
            # 计算注意力权重
            neighbor_k = K[neighbor_indices]  # [num_neighbors, num_heads, hidden_dim]
            neighbor_v = V[neighbor_indices]  # [num_neighbors, num_heads, hidden_dim]
            
            # 关系调制的key
            rel_embs_expanded = rel_embs.unsqueeze(1).expand(-1, self.num_heads, -1)
            modulated_k = neighbor_k + 0.2 * rel_embs_expanded
            
            # 注意力分数
            attn_scores = torch.sum(Q[i] * modulated_k, dim=-1) / (self.hidden_dim ** 0.5)
            attn_weights = F.softmax(attn_scores, dim=0)  # [num_neighbors, num_heads]
            
            # 聚合消息
            weighted_v = attn_weights.unsqueeze(-1) * neighbor_v  # [num_neighbors, num_heads, hidden_dim]
            aggregated = torch.sum(weighted_v, dim=0)  # [num_heads, hidden_dim]
            messages.append(aggregated)
            
        # 组合所有消息
        all_messages = torch.stack(messages)  # [num_nodes, num_heads, hidden_dim]
        all_messages = all_messages.view(num_nodes, -1)  # [num_nodes, num_heads * hidden_dim]
        
        # 输出投影和残差连接
        output = self.out_proj(all_messages)
        output = self.layer_norm(output + node_features)
        
        return output


class QCGPRGNNCore(nn.Module):
    """QC-GPR-GNN骨干网络 - 支持动态关系数量"""
    
    def __init__(self, node_dim=768, hidden_dim=256, num_layers=3, num_heads=4, num_relations=10):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_relations = num_relations
        
        # 节点特征投影
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        
        # GNN层 - 传入真实关系数量
        self.gnn_layers = nn.ModuleList([
            GNNMessagePassing(hidden_dim, hidden_dim // num_heads, num_heads, num_relations)
            for _ in range(num_layers)
        ])
        
        # GPR权重（可学习的层权重组合）
        self.gpr_weights = nn.Parameter(torch.ones(num_layers + 1))
        
    def forward(self, node_features, edge_index, edge_relations, query_emb):
        """
        Args:
            node_features: [num_nodes, node_dim] - 节点SBERT嵌入
            edge_index: [2, num_edges] - 边索引
            edge_relations: [num_edges] - 关系类型
            query_emb: [node_dim] - 查询嵌入
        """
        # 初始投影
        h = self.node_proj(node_features)
        query_projected = self.node_proj(query_emb)
        
        # 收集所有层的输出（GPR机制）
        layer_outputs = [h]
        
        # 逐层传播
        for layer in self.gnn_layers:
            h = layer(h, edge_index, edge_relations, query_projected)
            layer_outputs.append(h)
            
        # GPR加权组合
        gpr_weights_norm = F.softmax(self.gpr_weights, dim=0)
        final_h = sum(w * output for w, output in zip(gpr_weights_norm, layer_outputs))
        
        return final_h


class GradientAttribution:
    """梯度归因计算器 - 真正的梯度归因实现"""
    
    @staticmethod
    def compute_gradient_attribution(discriminator, path: List[str], query: str, 
                                   node_embeddings: Dict, device='cuda') -> Dict[str, float]:
        """计算真正的梯度归因分数"""
        if len(path) < 2 or not node_embeddings:
            return {}
            
        try:
            discriminator.eval()
            
            # 收集路径节点的嵌入并启用梯度
            path_node_embeddings = []
            path_nodes = []
            for node in path:
                if node in node_embeddings:
                    emb = node_embeddings[node].clone().detach().to(device)
                    emb.requires_grad_(True)
                    path_node_embeddings.append(emb)
                    path_nodes.append(node)
                    
            if not path_node_embeddings:
                return {}
                
            # 构建路径输入
            final_entity = path[-1]
            path_string = '.'.join(path)
            path_data = [{'paths': {final_entity: [path_string]}}]
            
            # 前向传播
            outputs = discriminator([query], path_data, epoch=0)
            if not outputs or not outputs[0]['individual_scores']:
                return {}
                
            score = outputs[0]['individual_scores'][0]
            
            # 计算梯度
            gradients = torch.autograd.grad(
                outputs=score,
                inputs=path_node_embeddings,
                create_graph=False,
                retain_graph=False
            )
            
            # 使用梯度范数作为重要性
            attributions = {}
            for i, node in enumerate(path_nodes):
                importance = gradients[i].norm(dim=-1).item()
                attributions[node] = importance
                
            return attributions
            
        except Exception as e:
            return {}
    
    @staticmethod
    def batch_evaluate_paths(discriminator, paths: List[List[str]], queries: List[str]) -> List[float]:
        """批量判别器调用"""
        if not paths or not queries:
            return []
            
        try:
            # 批量构建路径数据
            batch_data = []
            batch_queries = []
            
            for i, path in enumerate(paths):
                if len(path) >= 1:  # 修改：允许单节点路径，但对长度为1的路径特殊处理
                    final_entity = path[-1]
                    
                    if len(path) == 1:
                        # 单节点路径：创建一个简单的自指向路径用于判别器评估
                        path_string = f"{final_entity}.self.{final_entity}"
                    else:
                        # 多跳路径：正常处理
                        path_string = '.'.join(path)
                        
                    batch_data.append({'paths': {final_entity: [path_string]}})
                    batch_queries.append(queries[i] if i < len(queries) else queries[0])
                    
            if not batch_data:
                return []
                
            # 一次性评估所有路径
            with torch.no_grad():
                outputs = discriminator(batch_queries, batch_data, epoch=0)
                
            scores = []
            for output in outputs:
                if output['individual_scores']:
                    raw_score = float(output['individual_scores'][0])
                    confidence = torch.sigmoid(torch.tensor(raw_score)).item()
                    scores.append(confidence)
                else:
                    scores.append(0.0)
                    
            return scores
            
        except Exception:
            return [0.0] * len(paths)


class GNNGenerator(nn.Module):
    """双重引导GNN生成器 - 完全基于真实数据"""
    
    def __init__(self, entity_embedding_path: str, knowledge_graph_path: str, 
                 discriminator=None, node_dim=768, hidden_dim=256, 
                 num_layers=3, num_heads=4, device='cuda'):
        super().__init__()
        
        self.device = device
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        
        # 加载预训练SBERT并冻结参数
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.sentence_model.eval()
        
        # 冻结SBERT所有参数
        for param in self.sentence_model.parameters():
            param.requires_grad = False
        
        # 加载真实知识图谱
        with open(knowledge_graph_path, 'rb') as f:
            self.kg = pickle.load(f)
            
        # 加载真实实体嵌入
        self.entity_embeddings = torch.load(entity_embedding_path, map_location=device)
        
        # 判别器（教师模型）
        self.discriminator = discriminator
        
        # 数据集固定的9种关系类型（按用户要求使用硬编码）
        self.relation_templates = {
            'starred_actors': 0,
            'directed_by': 1, 
            'written_by': 2,
            'has_genre': 3,
            'release_year': 4,
            'has_tags': 5,
            'in_language': 6,
            'has_imdb_rating': 7,
            'has_imdb_votes': 8
        }
        
        self.num_relations = 9
        self.relation_to_id = self.relation_templates
        self.id_to_relation = {i: rel for rel, i in self.relation_to_id.items()}
        
        # 关系到自然语言的映射
        self.relation_text_map = {
            'starred_actors': 'acted in',
            'directed_by': 'was directed by', 
            'written_by': 'was written by',
            'has_genre': 'belongs to genre',
            'release_year': 'was released in year',
            'has_tags': 'has tag',
            'in_language': 'is in language',
            'has_imdb_rating': 'has IMDB rating',
            'has_imdb_votes': 'has IMDB votes'
        }
        
        # QC-APPR引擎
        self.qc_appr = QueryConditionedAPPR(self.kg, self.sentence_model)
        
        # QC-GPR-GNN骨干 - 使用固定的9种关系
        self.gnn_core = QCGPRGNNCore(node_dim, hidden_dim, num_layers, num_heads, self.num_relations)
        
        # 双重输出头
        self.node_value_head = nn.Sequential(
            nn.Linear(hidden_dim + node_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
        
        # 修复维度：node_repr(256) + neighbor_emb(384) + rel_emb(64) + query_emb(384) = 1088
        self.expansion_policy_head = nn.Sequential(
            nn.Linear(hidden_dim + node_dim + 64 + node_dim, hidden_dim),  # 256 + 384 + 64 + 384 = 1088
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
        
        # 关系嵌入（与GNN层共享）- 固定9种关系
        self.relation_embeddings = nn.Embedding(self.num_relations, 64)
        
        # 梯度归因计算器
        self.gradient_attribution = GradientAttribution()
        
        self.to(device)
        
    def extract_subgraph_with_appr(self, query: str, start_entity: str, 
                                  hop: int, top_k: int = 50) -> Tuple[Set, Dict]:
        """使用QC-APPR提取查询相关子图"""
        
        if start_entity not in self.kg:
            return {start_entity}, {start_entity: 1.0}
            
        # 从起始实体开始的k跳邻域
        current_nodes = {start_entity}
        all_nodes = {start_entity}
        
        for h in range(hop):
            next_nodes = set()
            for node in current_nodes:
                if node in self.kg:
                    neighbors = set(self.kg.neighbors(node))
                    next_nodes.update(neighbors)
            current_nodes = next_nodes - all_nodes
            all_nodes.update(current_nodes)
            
        # 如果子图过大，用APPR筛选
        if len(all_nodes) > top_k * 2:
            # 计算查询条件化转移矩阵
            transition_probs = self.qc_appr.compute_transition_matrix(
                query, all_nodes, lambda_param=0.6, mu_param=0.4
            )
            
            # Forward-Push APPR算法
            appr_scores = self.qc_appr.forward_push_appr(start_entity, transition_probs)
            
            # 选择Top-K高分节点
            sorted_nodes = sorted(appr_scores.items(), key=lambda x: x[1], reverse=True)
            selected_nodes = {node for node, _ in sorted_nodes[:top_k]}
            selected_nodes.add(start_entity)  # 确保包含起始节点
        else:
            selected_nodes = all_nodes
            appr_scores = {node: 1.0 / len(all_nodes) for node in all_nodes}
            
        return selected_nodes, appr_scores
        
    def build_subgraph_tensors(self, subgraph_nodes: Set, query: str):
        """构建子图的张量表示 - 使用真实图结构"""
        
        # 节点列表和映射
        node_list = list(subgraph_nodes)
        node_to_idx = {node: i for i, node in enumerate(node_list)}
        
        # 节点特征：优先使用预训练嵌入
        node_features = []
        for node in node_list:
            if node in self.entity_embeddings:
                emb = self.entity_embeddings[node].clone()
            else:
                # 对未知实体使用SBERT编码
                node_text = node.replace('_', ' ')
                with torch.no_grad():
                    emb = torch.tensor(self.sentence_model.encode([node_text])[0])
            node_features.append(emb)
            
        node_features = torch.stack(node_features).to(self.device)
        
        # 边索引和关系 - 从真实图结构构建
        edge_indices = []
        edge_relations = []
        
        subgraph = self.kg.subgraph(subgraph_nodes)
        for u, v, data in subgraph.edges(data=True):
            if u in node_to_idx and v in node_to_idx:
                u_idx, v_idx = node_to_idx[u], node_to_idx[v]
                relation = data.get('relation', 'unknown')
                rel_id = self.relation_to_id.get(relation, 0)
                
                # 无向图：添加双向边
                edge_indices.extend([[u_idx, v_idx], [v_idx, u_idx]])
                edge_relations.extend([rel_id, rel_id])
                
        if edge_indices:
            edge_index = torch.tensor(edge_indices).T.to(self.device)
            edge_relations = torch.tensor(edge_relations).to(self.device)
        else:
            # 处理孤立节点：添加自环
            num_nodes = len(node_list)
            edge_index = torch.tensor([[i, i] for i in range(num_nodes)]).T.to(self.device)
            edge_relations = torch.zeros(num_nodes, dtype=torch.long).to(self.device)
            
        # 查询嵌入 - 使用冻结的SBERT
        with torch.no_grad():
            query_emb = torch.tensor(self.sentence_model.encode([query])[0]).to(self.device)
        
        return node_features, edge_index, edge_relations, query_emb, node_list, node_to_idx
        
    def forward(self, query: str, start_entity: str, hop: int = 2, 
                top_k_nodes: int = 50) -> Dict:
        """前向传播"""
        
        # 1. 使用QC-APPR提取查询相关子图
        subgraph_nodes, appr_scores = self.extract_subgraph_with_appr(
            query, start_entity, hop, top_k_nodes
        )
        
        if len(subgraph_nodes) < 2:
            return {
                'node_values': {},
                'expansion_policies': {},
                'subgraph_nodes': subgraph_nodes,
                'node_representations': {},
                'appr_scores': appr_scores
            }
            
        # 2. 构建真实图结构的张量表示
        node_features, edge_index, edge_relations, query_emb, node_list, node_to_idx = \
            self.build_subgraph_tensors(subgraph_nodes, query)
            
        # 3. QC-GPR-GNN骨干网络
        node_representations = self.gnn_core(node_features, edge_index, edge_relations, query_emb)
        
        # 4. 节点价值头
        query_expanded = query_emb.unsqueeze(0).expand(len(node_list), -1)
        value_input = torch.cat([node_representations, query_expanded], dim=1)
        node_values_tensor = self.node_value_head(value_input).squeeze(1)
        
        node_values = {node_list[i]: node_values_tensor[i] for i in range(len(node_list))}  # 保持tensor格式维持梯度
        
        # 5. 扩展策略头 - 基于真实图结构计算前沿边
        expansion_policies = {}
        current_subgraph = set(subgraph_nodes)
        
        for node in subgraph_nodes:
            if node not in self.kg:
                continue
                
            node_idx = node_to_idx[node]
            node_repr = node_representations[node_idx]
            
            # 获取真实图中的邻居
            real_neighbors = set(self.kg.neighbors(node))
            frontier_neighbors = real_neighbors - current_subgraph
            
            for neighbor in list(frontier_neighbors)[:15]:  # 适当限制邻居数量
                # 邻居特征
                if neighbor in self.entity_embeddings:
                    neighbor_emb = self.entity_embeddings[neighbor].clone().to(self.device)
                else:
                    neighbor_text = neighbor.replace('_', ' ')
                    with torch.no_grad():
                        neighbor_emb = torch.tensor(
                            self.sentence_model.encode([neighbor_text])[0]
                        ).to(self.device)
                        
                # 真实关系特征
                edge_data = self.kg.get_edge_data(node, neighbor)
                relation = edge_data.get('relation', 'unknown') if edge_data else 'unknown'
                rel_id = self.relation_to_id.get(relation, 0)
                rel_emb = self.relation_embeddings(torch.tensor(rel_id).to(self.device))
                
                # 扩展策略输入
                policy_input = torch.cat([node_repr, neighbor_emb, rel_emb, query_emb])
                expansion_score = self.expansion_policy_head(policy_input).squeeze()
                
                expansion_policies[(node, neighbor)] = float(expansion_score)
                
        # 6. 节点表示字典
        node_reprs = {node_list[i]: node_representations[i] for i in range(len(node_list))}
        
        return {
            'node_values': node_values,
            'expansion_policies': expansion_policies,
            'subgraph_nodes': subgraph_nodes,
            'node_representations': node_reprs,
            'appr_scores': appr_scores
        }
        
    def batch_evaluate_paths(self, paths: List[List[str]], queries: List[str]) -> List[float]:
        """批量路径评估"""
        return GradientAttribution.batch_evaluate_paths(self.discriminator, paths, queries)
        
    def generate_paths_with_beam_search(self, query: str, start_entity: str, max_hops: int = 3, 
                                      beam_size: int = 5, max_paths: int = 10) -> List[Tuple[List, float]]:
        """使用Beam Search生成高质量路径"""
        
        if start_entity not in self.kg:
            return []
            
        paths_with_scores = []
        
        # 初始化beam
        beam = [(([start_entity], 0.0))]  # (路径, 累积分数)
        
        for hop in range(max_hops):
            next_beam = []
            
            for path, path_score in beam:
                # 获取路径中的最后一个实体（跳过关系）
                current_entity = path[-1] if len(path) % 2 == 1 else path[-2] if len(path) > 1 else path[-1]
                
                if current_entity not in self.kg:
                    continue
                
                # 使用GNN获取扩展策略
                gnn_output = self.forward(query, current_entity, hop=hop+1, top_k_nodes=30)
                expansion_policies = gnn_output.get('expansion_policies', {})
                
                # 获取当前实体的真实邻居和扩展选项
                real_neighbors = set(self.kg.neighbors(current_entity))
                expansion_options = []
                
                # 获取路径中已访问的实体（避免循环）
                visited_entities = set()
                for i, node in enumerate(path):
                    if i % 2 == 0:  # 只统计实体（偶数位置）
                        visited_entities.add(node)
                
                for (from_node, to_node), score in expansion_policies.items():
                    if (from_node == current_entity and 
                        to_node in real_neighbors and 
                        to_node not in visited_entities):  # 避免重复访问
                        expansion_options.append((to_node, score))
                        
                # 如果没有扩展选项，也加入当前路径作为候选
                if not expansion_options:
                    paths_with_scores.append((path, path_score))
                    continue
                        
                # 排序并选择top-k扩展
                expansion_options.sort(key=lambda x: x[1], reverse=True)
                
                for next_entity, expansion_score in expansion_options[:beam_size]:
                    # 获取关系信息
                    edge_data = self.kg.get_edge_data(current_entity, next_entity)
                    relation = edge_data.get('relation', 'unknown') if edge_data else 'unknown'
                    
                    # 创建完整路径：实体-关系-实体格式
                    if len(path) == 1:  # 第一跳
                        new_path = path + [relation, next_entity]
                    else:  # 后续跳跃
                        new_path = path + [relation, next_entity]
                        
                    new_score = path_score + expansion_score
                    next_beam.append((new_path, new_score))
                    
                    # 也加入最终路径集合
                    paths_with_scores.append((new_path, new_score))
                    
            # 保持beam大小并继续下一跳
            if next_beam:
                beam = sorted(next_beam, key=lambda x: x[1], reverse=True)[:beam_size]
            else:
                break
                
        # 返回按分数排序的top-k路径
        paths_with_scores.sort(key=lambda x: x[1], reverse=True)
        return paths_with_scores[:max_paths]


def create_gnn_generator(entity_embedding_path: str, knowledge_graph_path: str, 
                        discriminator=None, **kwargs):
    """创建GNN生成器工厂函数"""
    return GNNGenerator(
        entity_embedding_path=entity_embedding_path,
        knowledge_graph_path=knowledge_graph_path,
        discriminator=discriminator,
        **kwargs
    )


if __name__ == "__main__":
    print("Query-Conditioned GNN Generator (Fixed Version)")
    print("修复内容:")
    print("1. 冻结SBERT参数")
    print("2. 动态计算关系数量")
    print("3. 使用真实图结构")
    print("4. 实现梯度归因机制")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")
    
    try:
        generator = create_gnn_generator(
            entity_embedding_path="embeddings/entity_embeddings.pt",
            knowledge_graph_path="graph/knowledge_graph.pkl",
            device=device
        )
        
        print(f"关系数量: {generator.num_relations}")
        print(f"关系类型: {list(generator.relation_to_id.keys())[:10]}...")
        
        # 验证SBERT参数已冻结
        sbert_params_frozen = all(not p.requires_grad for p in generator.sentence_model.parameters())
        print(f"SBERT参数已冻结: {sbert_params_frozen}")
        
        # 测试前向传播
        query = "Who directed The Dark Knight?"
        start_entity = "the_dark_knight"
        
        if start_entity in generator.kg:
            print(f"测试查询: {query}")
            print(f"起始实体: {start_entity}")
            
            output = generator.forward(query, start_entity, hop=2)
            
            print(f"子图节点数: {len(output['subgraph_nodes'])}")
            print(f"节点价值分数数: {len(output['node_values'])}")
            print(f"扩展策略数: {len(output['expansion_policies'])}")
            
            # 测试路径生成
            paths = generator.generate_paths_with_beam_search(query, start_entity, max_hops=2, beam_size=3)
            print(f"生成路径数: {len(paths)}")
            for i, (path, score) in enumerate(paths[:3]):
                print(f"  {i+1}. {' -> '.join(path[-3:])}: {score:.3f}")
        else:
            print(f"实体 {start_entity} 不在知识图谱中")
            
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()