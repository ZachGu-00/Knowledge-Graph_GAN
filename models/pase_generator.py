"""
PASE: 渐进式自适应子图扩展生成器
Progressive Adaptive Subgraph Expansion Generator

基于proposal实现的轻量化、无可学习参数的图结构对抗生成器

核心创新:
1. 零样本Query关联度注入 (Zero-shot Relevance Injection)
2. 免参数信号传播GNN (Parameter-Free Propagation GNN)  
3. 判别器指导的自适应决策 (Discriminator-Guided Adaptive Decision)
"""

import torch
import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
import pickle
import json
from pathlib import Path


class PASEGenerator:
    """
    PASE生成器 - 轻量化无参数图结构对抗生成器
    
    与传统生成器不同，本生成器：
    1. 不包含任何可学习参数
    2. 完全依赖预训练语义模型(SBERT)
    3. 使用判别器指导进行自适应探索
    """
    
    def __init__(self, 
                 sbert_model_name: str = 'all-MiniLM-L6-v2',
                 entity_embedding_path: str = None,
                 discriminator=None,
                 T_base: float = 0.8,
                 delta: float = 0.1,
                 alpha: float = 0.7,
                 max_hops: int = 4):
        """
        初始化PASE生成器
        
        Args:
            sbert_model_name: SBERT模型名称
            entity_embedding_path: 预计算实体嵌入路径
            discriminator: 判别器模型实例
            T_base: 基础置信度阈值
            delta: 阈值衰减因子
            alpha: 自身信息保留比例
            max_hops: 最大跳数
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 语义编码器(冻结)
        self.sbert = SentenceTransformer(sbert_model_name)
        for param in self.sbert.parameters():
            param.requires_grad = False
        print(f"SBERT模型加载完成: {sbert_model_name}")
        
        # 实体嵌入缓存
        self.entity_embeddings = None
        if entity_embedding_path and Path(entity_embedding_path).exists():
            self.load_entity_embeddings(entity_embedding_path)
        
        # 知识图谱
        self.knowledge_graph = None
        
        # 判别器
        self.discriminator = discriminator
        
        # PASE超参数
        self.T_base = T_base  # 基础阈值 
        self.delta = delta    # 衰减因子
        self.alpha = alpha    # 自身信息保留比例
        self.max_hops = max_hops
        
        # 关系模板映射
        self.relation_templates = {
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
        
        # 缓存机制
        self._embedding_cache = {}
        
    def load_entity_embeddings(self, embedding_path: str):
        """加载预计算的实体嵌入"""
        try:
            self.entity_embeddings = torch.load(embedding_path, map_location=self.device)
            print(f"实体嵌入加载完成: {len(self.entity_embeddings)} 个实体")
        except Exception as e:
            print(f"实体嵌入加载失败: {e}")
            
    def load_knowledge_graph(self, graph_path: str):
        """加载知识图谱"""
        try:
            with open(graph_path, 'rb') as f:
                self.knowledge_graph = pickle.load(f)
            print(f"知识图谱加载完成: {self.knowledge_graph.number_of_nodes()} 节点, "
                  f"{self.knowledge_graph.number_of_edges()} 边")
        except Exception as e:
            print(f"知识图谱加载失败: {e}")
    
    def get_entity_embedding(self, entity: str) -> torch.Tensor:
        """获取实体的语义嵌入"""
        if entity in self._embedding_cache:
            return self._embedding_cache[entity]
        
        # 优先使用预计算嵌入
        if self.entity_embeddings and entity in self.entity_embeddings:
            embedding = self.entity_embeddings[entity].to(self.device)
        else:
            # 回退到SBERT实时编码
            with torch.no_grad():
                embedding = self.sbert.encode(entity, convert_to_tensor=True).to(self.device)
        
        # 缓存结果
        self._embedding_cache[entity] = embedding
        return embedding
    
    def get_relation_embedding(self, relation: str) -> torch.Tensor:
        """获取关系的语义嵌入"""
        cache_key = f"rel_{relation}"
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        
        # 使用关系模板或原始关系名
        relation_text = self.relation_templates.get(relation, relation)
        
        with torch.no_grad():
            embedding = self.sbert.encode(relation_text, convert_to_tensor=True).to(self.device)
        
        # 缓存结果
        self._embedding_cache[cache_key] = embedding
        return embedding
    
    def compute_zero_shot_relevance(self, query: str, entities: List[str], 
                                   relations: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        创新点1: 零样本Query关联度注入
        
        使用SBERT计算query与实体/关系的语义相似度作为初始关联度
        """
        # 编码查询
        with torch.no_grad():
            query_embedding = self.sbert.encode(query, convert_to_tensor=True).to(self.device)
        
        # 计算实体关联度
        entity_scores = []
        for entity in entities:
            entity_embedding = self.get_entity_embedding(entity)
            similarity = torch.cosine_similarity(
                query_embedding.unsqueeze(0), 
                entity_embedding.unsqueeze(0)
            ).item()
            entity_scores.append(similarity)
        
        # 计算关系关联度  
        relation_scores = []
        for relation in relations:
            relation_embedding = self.get_relation_embedding(relation)
            similarity = torch.cosine_similarity(
                query_embedding.unsqueeze(0),
                relation_embedding.unsqueeze(0) 
            ).item()
            relation_scores.append(similarity)
        
        return torch.tensor(entity_scores), torch.tensor(relation_scores)
    
    def parameter_free_propagation(self, subgraph: nx.Graph, initial_scores: Dict[str, float],
                                 relation_scores: Dict[str, float], iterations: int = 3) -> Dict[str, float]:
        """
        创新点2: 免参数信号传播GNN
        
        使用类似标签传播的无参数更新规则在子图上传播关联度信号
        """
        # 初始化节点状态
        current_scores = initial_scores.copy()
        
        for iteration in range(iterations):
            next_scores = {}
            
            for node in subgraph.nodes():
                if node not in current_scores:
                    continue
                
                # 获取邻居
                neighbors = list(subgraph.neighbors(node))
                if not neighbors:
                    next_scores[node] = current_scores[node]
                    continue
                
                # 计算邻居信号的加权和
                neighbor_sum = 0.0
                weight_sum = 0.0
                
                for neighbor in neighbors:
                    if neighbor not in current_scores:
                        continue
                    
                    # 获取连接关系及其权重
                    edge_data = subgraph.get_edge_data(node, neighbor)
                    if edge_data:
                        # 处理多重边
                        relations = []
                        for key, data in edge_data.items():
                            if isinstance(data, dict):
                                relation = data.get('relation', key)
                            else:
                                relation = str(data) if data else key
                            relations.append(relation)
                        
                        # 使用关系权重(语义相关度)
                        relation_weight = max([relation_scores.get(rel, 0.1) for rel in relations])
                        
                        neighbor_sum += relation_weight * current_scores[neighbor]
                        weight_sum += relation_weight
                
                # PASE更新公式: h_i^{k+1} = α * h_i^k + (1-α) * 邻居加权平均
                if weight_sum > 0:
                    neighbor_avg = neighbor_sum / weight_sum
                    next_scores[node] = self.alpha * current_scores[node] + (1 - self.alpha) * neighbor_avg
                else:
                    next_scores[node] = current_scores[node]
            
            current_scores = next_scores
        
        return current_scores
    
    def build_h_hop_subgraph(self, query_entity: str, h: int) -> Tuple[nx.Graph, List[str], List[str]]:
        """构建h跳子图并返回实体和关系列表"""
        if not self.knowledge_graph or query_entity not in self.knowledge_graph:
            return nx.Graph(), [], []
        
        # BFS构建h跳子图
        subgraph_nodes = {query_entity}
        current_layer = {query_entity}
        
        for hop in range(h):
            next_layer = set()
            for node in current_layer:
                if node in self.knowledge_graph:
                    neighbors = set(self.knowledge_graph.neighbors(node))
                    next_layer.update(neighbors)
            
            subgraph_nodes.update(next_layer)
            current_layer = next_layer
        
        # 提取子图
        subgraph = self.knowledge_graph.subgraph(subgraph_nodes)
        
        # 收集实体和关系
        entities = list(subgraph_nodes)
        relations = set()
        
        for u, v, data in subgraph.edges(data=True):
            if isinstance(data, dict):
                for key, edge_info in data.items():
                    if isinstance(edge_info, dict):
                        relation = edge_info.get('relation', key)
                    else:
                        relation = str(edge_info) if edge_info else key
                    relations.add(relation)
        
        return subgraph, entities, list(relations)
    
    def extract_best_path(self, query_entity: str, final_scores: Dict[str, float], 
                         subgraph: nx.Graph) -> Tuple[List[str], str, float]:
        """从评分结果中提取最佳路径"""
        if not final_scores:
            return [], "", 0.0
        
        # 找到得分最高的目标实体(排除起始实体)
        target_candidates = [(entity, score) for entity, score in final_scores.items() 
                           if entity != query_entity and entity in subgraph]
        
        if not target_candidates:
            return [], "", 0.0
        
        target_candidates.sort(key=lambda x: x[1], reverse=True)
        target_entity, target_score = target_candidates[0]
        
        # 寻找从query_entity到target_entity的最短路径
        try:
            if nx.has_path(subgraph, query_entity, target_entity):
                node_path = nx.shortest_path(subgraph, query_entity, target_entity)
                
                # 构建完整路径(实体-关系-实体...)
                full_path = [node_path[0]]
                for i in range(len(node_path) - 1):
                    current_node = node_path[i]
                    next_node = node_path[i + 1]
                    
                    # 获取连接关系
                    edge_data = subgraph.get_edge_data(current_node, next_node)
                    if edge_data:
                        # 选择第一个关系
                        for key, data in edge_data.items():
                            if isinstance(data, dict):
                                relation = data.get('relation', key)
                            else:
                                relation = str(data) if data else key
                            break
                        
                        full_path.extend([relation, next_node])
                    else:
                        full_path.append(next_node)
                
                return full_path, target_entity, target_score
            else:
                return [query_entity, "unknown_relation", target_entity], target_entity, target_score
                
        except Exception as e:
            print(f"路径提取失败: {e}")
            return [], "", 0.0
    
    def discriminator_guided_adaptive_decision(self, question: str, path: List[str], 
                                             current_hop: int) -> Tuple[bool, float]:
        """
        创新点3: 判别器指导的自适应决策
        
        使用动态阈值判断是否继续探索更深层路径
        """
        if not self.discriminator or len(path) == 0:
            return False, 0.0
        
        # 计算动态阈值
        dynamic_threshold = max(0.1, self.T_base - (current_hop - 1) * self.delta)
        
        # 构造判别器输入
        final_entity = path[-1]
        path_string = '.'.join(path)
        path_data = [{'paths': {final_entity: [path_string]}}]
        
        try:
            # 获取判别器评分
            with torch.no_grad():
                outputs = self.discriminator([question], path_data, epoch=0)
                raw_score = float(outputs[0]['individual_scores'][0])
                confidence = torch.sigmoid(torch.tensor(raw_score)).item()
            
            # 自适应决策
            should_continue = confidence < dynamic_threshold
            
            return should_continue, confidence
            
        except Exception as e:
            print(f"判别器评估失败: {e}")
            return True, 0.0  # 默认继续探索
    
    def progressive_adaptive_subgraph_expansion(self, question: str, query_entity: str,
                                              verbose: bool = False) -> Tuple[List[str], int, float]:
        """
        PASE核心算法: 渐进式自适应子图扩展
        
        实现proposal中描述的完整流程:
        1. 初始化h=1
        2. 循环: 构建子图 -> 注入信号 -> 传播 -> 构建路径 -> 请求验证 -> 自适应决策
        """
        current_hop = 1
        best_path = []
        best_score = 0.0
        
        if verbose:
            print(f"开始PASE生成: {question}")
            print(f"起始实体: {query_entity}")
        
        while current_hop <= self.max_hops:
            if verbose:
                print(f"\n=== 第{current_hop}跳探索 ===")
            
            # a. 构建子图
            subgraph, entities, relations = self.build_h_hop_subgraph(query_entity, current_hop)
            
            if len(entities) <= 1:
                if verbose:
                    print(f"第{current_hop}跳: 无可用实体，停止探索")
                break
            
            if verbose:
                print(f"子图规模: {len(entities)}个实体, {len(relations)}个关系")
            
            # b. 信号注入: 零样本Query关联度
            entity_scores, relation_scores = self.compute_zero_shot_relevance(
                question, entities, relations
            )
            
            # 转换为字典格式
            initial_scores = {entity: float(score) for entity, score in zip(entities, entity_scores)}
            relation_score_dict = {relation: float(score) for relation, score in zip(relations, relation_scores)}
            
            # c. 信号传播: 免参数GNN
            final_scores = self.parameter_free_propagation(
                subgraph, initial_scores, relation_score_dict
            )
            
            # d. 路径构建: 提取最高分路径
            path, target_entity, path_score = self.extract_best_path(
                query_entity, final_scores, subgraph
            )
            
            if len(path) == 0:
                if verbose:
                    print(f"第{current_hop}跳: 无法构建有效路径")
                current_hop += 1
                continue
            
            if verbose:
                print(f"最佳路径: {' -> '.join(path[-6:])}")  # 显示最后6个元素
                print(f"目标实体: {target_entity}, 路径得分: {path_score:.4f}")
            
            # e. 请求验证: 判别器评估
            should_continue, discriminator_confidence = self.discriminator_guided_adaptive_decision(
                question, path, current_hop
            )
            
            if verbose:
                dynamic_threshold = max(0.1, self.T_base - (current_hop - 1) * self.delta)
                print(f"判别器置信度: {discriminator_confidence:.4f}")
                print(f"动态阈值: {dynamic_threshold:.4f}")
                print(f"决策: {'继续探索' if should_continue else '返回路径'}")
            
            # 更新最佳路径
            if discriminator_confidence > best_score:
                best_path = path
                best_score = discriminator_confidence
            
            # f. 自适应决策
            if not should_continue:
                if verbose:
                    print(f"探索完成! 最终路径长度: {len(best_path)}")
                return best_path, current_hop, discriminator_confidence
            
            # 继续下一跳
            current_hop += 1
        
        # 达到最大跳数，返回最佳结果
        if verbose:
            print(f"达到最大跳数{self.max_hops}，返回最佳路径")
        
        return best_path, self.max_hops, best_score
    
    def generate_paths(self, question: str, start_entity: str, 
                      target_entities: Set[str] = None, max_paths: int = 1,
                      stochastic: bool = False) -> List[Tuple[List[str], float, float]]:
        """
        标准接口: 生成路径
        
        为了与现有训练框架兼容，提供标准的generate_paths接口
        """
        path, final_hop, confidence = self.progressive_adaptive_subgraph_expansion(
            question, start_entity, verbose=False
        )
        
        if len(path) == 0:
            return []
        
        # 返回格式: (path, score, log_prob)
        return [(path, confidence, np.log(confidence + 1e-8))]
    
    def persistent_query_exploration(self, question: str, start_entity: str,
                                   answer_entities: Set[str], max_attempts: int = 1) -> Tuple[List[str], int, bool]:
        """
        持续探索接口 - 兼容现有训练框架
        """
        path, attempts, confidence = self.progressive_adaptive_subgraph_expansion(
            question, start_entity, verbose=False
        )
        
        if len(path) == 0:
            return [], max_attempts, False
        
        # 检查是否命中目标
        final_entity = path[-1] if path else ""
        found_correct = final_entity in answer_entities
        
        return path, 1, found_correct  # PASE只需一次尝试
    
    def enable_stochastic_exploration(self):
        """启用随机探索模式 - 兼容接口"""
        pass  # PASE是确定性的
    
    def disable_stochastic_exploration(self):  
        """禁用随机探索模式 - 兼容接口"""
        pass  # PASE是确定性的
    
    def to(self, device):
        """设备转移"""
        self.device = device
        return self
    
    def get_trainable_parameters(self):
        """获取可训练参数 - 返回空列表因为PASE无参数"""
        return []
    
    def train(self):
        """训练模式 - 无操作"""
        pass
    
    def eval(self):
        """评估模式 - 无操作"""  
        pass
    
    def state_dict(self):
        """状态字典 - 返回配置参数"""
        return {
            'T_base': self.T_base,
            'delta': self.delta,
            'alpha': self.alpha,
            'max_hops': self.max_hops,
            'relation_templates': self.relation_templates
        }
    
    def load_state_dict(self, state_dict):
        """加载状态 - 更新配置参数"""
        self.T_base = state_dict.get('T_base', self.T_base)
        self.delta = state_dict.get('delta', self.delta) 
        self.alpha = state_dict.get('alpha', self.alpha)
        self.max_hops = state_dict.get('max_hops', self.max_hops)
        self.relation_templates.update(state_dict.get('relation_templates', {}))


def create_pase_generator(entity_embedding_path: str = "embeddings/entity_embeddings.pt",
                         knowledge_graph_path: str = "graph/knowledge_graph.pkl",
                         discriminator=None, **kwargs) -> PASEGenerator:
    """
    创建PASE生成器的工厂函数
    """
    generator = PASEGenerator(
        entity_embedding_path=entity_embedding_path,
        discriminator=discriminator,
        **kwargs
    )
    
    # 加载知识图谱
    generator.load_knowledge_graph(knowledge_graph_path)
    
    return generator


if __name__ == "__main__":
    print("PASE生成器测试")
    
    # 创建生成器实例
    generator = PASEGenerator()
    
    # 模拟测试
    test_question = "谁是《泰坦尼克号》的主演？"
    test_entity = "Titanic"
    
    print(f"测试问题: {test_question}")
    print(f"起始实体: {test_entity}")
    
    if generator.knowledge_graph is None:
        print("需要先加载知识图谱才能测试")
    else:
        path, hops, confidence = generator.progressive_adaptive_subgraph_expansion(
            test_question, test_entity, verbose=True
        )
        print(f"\n最终结果:")
        print(f"路径: {' -> '.join(path)}")
        print(f"跳数: {hops}")
        print(f"置信度: {confidence:.4f}")