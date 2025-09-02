import torch
import pickle
import networkx as nx
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict
from tqdm import tqdm

class PrecomputedNegativeSampler:
    """高效的负样本预计算器"""
    
    def __init__(self, kg_file: str, data_file: str, max_negatives_per_sample: int = 10):
        self.kg_file = kg_file
        self.data_file = data_file
        self.max_negatives_per_sample = max_negatives_per_sample
        
        # 加载知识图谱
        print("Loading knowledge graph...")
        with open(kg_file, 'rb') as f:
            self.kg = pickle.load(f)
        print(f"Loaded KG: {len(self.kg.nodes)} nodes, {len(self.kg.edges)} edges")
        
        # 预计算所有实体的多跳邻居（一次性计算）
        self.precompute_multihop_neighbors()
        
        # 预计算负样本
        self.negative_cache = {}
        self.precompute_negative_samples()
    
    def precompute_multihop_neighbors(self):
        """一次性预计算所有实体的1-3跳邻居"""
        print("Precomputing multi-hop neighbors for all entities...")
        
        self.hop_neighbors = defaultdict(lambda: defaultdict(set))  # {entity: {hop: neighbors}}
        
        all_entities = list(self.kg.nodes())
        
        for entity in tqdm(all_entities, desc="Computing neighbors"):
            if entity not in self.kg.nodes():
                continue
            
            # 1跳邻居
            hop1_neighbors = set(self.kg.neighbors(entity))
            self.hop_neighbors[entity][1] = hop1_neighbors
            
            # 2跳邻居
            hop2_neighbors = set()
            for neighbor in hop1_neighbors:
                if neighbor in self.kg.nodes():
                    hop2_neighbors.update(self.kg.neighbors(neighbor))
            hop2_neighbors -= {entity}  # 移除自身
            hop2_neighbors -= hop1_neighbors  # 移除1跳邻居
            self.hop_neighbors[entity][2] = hop2_neighbors
            
            # 3跳邻居
            hop3_neighbors = set()
            for neighbor in hop2_neighbors:
                if neighbor in self.kg.nodes():
                    hop3_neighbors.update(self.kg.neighbors(neighbor))
            hop3_neighbors -= {entity}  # 移除自身
            hop3_neighbors -= hop1_neighbors  # 移除1跳邻居
            hop3_neighbors -= hop2_neighbors  # 移除2跳邻居
            self.hop_neighbors[entity][3] = hop3_neighbors
        
        print(f"Precomputed neighbors for {len(self.hop_neighbors)} entities")
    
    def extract_hop_count(self, sample: Dict) -> int:
        """从样本中提取跳数"""
        query_type = sample.get('type', '')
        if query_type and query_type[0].isdigit():
            return int(query_type.split('_')[0][0])
        return 1
    
    def find_real_negatives(self, question_entity: str, answer_entities: List[str], 
                           hop_count: int) -> List[Tuple[str, str]]:
        """找到真实的负样本实体和路径"""
        negatives = []
        
        if question_entity not in self.hop_neighbors:
            return negatives
        
        # 获取指定跳数的所有可达实体（预计算好的）
        reachable_entities = self.hop_neighbors[question_entity].get(hop_count, set())
        
        # 排除正确答案实体
        negative_entities = reachable_entities - set(answer_entities)
        
        # 为负样本实体生成路径
        for neg_entity in list(negative_entities)[:self.max_negatives_per_sample]:
            try:
                if nx.has_path(self.kg, question_entity, neg_entity):
                    path = nx.shortest_path(self.kg, question_entity, neg_entity)
                    if len(path) - 1 == hop_count:  # 确保跳数匹配
                        # 构造路径字符串
                        path_parts = []
                        for i in range(len(path) - 1):
                            edge_data = self.kg.get_edge_data(path[i], path[i+1])
                            relation = edge_data.get('relation', 'related_to') if edge_data else 'related_to'
                            path_parts.extend([path[i], relation])
                        path_parts.append(path[-1])
                        path_string = '.'.join(path_parts)
                        negatives.append((neg_entity, path_string))
            except:
                continue
        
        return negatives
    
    def precompute_negative_samples(self):
        """预计算所有样本的负样本"""
        print("Precomputing negative samples for all training data...")
        
        # 加载训练数据
        with open(self.data_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        query_blocks = content.strip().split('\n\n')
        samples = []
        
        for block in query_blocks:
            if block.strip():
                try:
                    sample = json.loads(block)
                    if all(key in sample for key in ['question', 'question_entity', 'answer_entities']):
                        samples.append(sample)
                except:
                    continue
        
        print(f"Found {len(samples)} valid samples")
        
        # 为每个样本预计算负样本
        for i, sample in enumerate(tqdm(samples, desc="Computing negatives")):
            sample_id = f"sample_{i}"  # 简单的ID生成
            question_entity = sample['question_entity']
            answer_entities = sample['answer_entities']
            hop_count = self.extract_hop_count(sample)
            
            # 一次性计算所有可能的negative paths
            negatives = self.find_real_negatives(question_entity, answer_entities, hop_count)
            self.negative_cache[sample_id] = negatives
        
        print(f"Precomputed negatives for {len(self.negative_cache)} samples")
        
        # 统计信息
        total_negatives = sum(len(negs) for negs in self.negative_cache.values())
        avg_negatives = total_negatives / len(self.negative_cache) if self.negative_cache else 0
        print(f"Average negatives per sample: {avg_negatives:.2f}")
    
    def get_negatives(self, sample_id: str, num_negatives: int = 2) -> List[Tuple[str, str]]:
        """获取指定样本的负样本"""
        available_negatives = self.negative_cache.get(sample_id, [])
        if len(available_negatives) >= num_negatives:
            return random.sample(available_negatives, num_negatives)
        else:
            return available_negatives
    
    def save_cache(self, cache_file: str):
        """保存预计算的负样本缓存"""
        cache_data = {
            'negative_cache': self.negative_cache,
            'hop_neighbors': dict(self.hop_neighbors)  # 转换为普通dict
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"Saved negative sample cache to {cache_file}")
    
    def load_cache(self, cache_file: str):
        """加载预计算的负样本缓存"""
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        
        self.negative_cache = cache_data['negative_cache']
        self.hop_neighbors = defaultdict(lambda: defaultdict(set), cache_data['hop_neighbors'])
        print(f"Loaded negative sample cache from {cache_file}")

def create_negative_sampler(kg_file: str, data_file: str, cache_file: str = None):
    """创建或加载负样本采样器"""
    
    if cache_file and Path(cache_file).exists():
        print("Loading cached negative sampler...")
        sampler = PrecomputedNegativeSampler.__new__(PrecomputedNegativeSampler)
        sampler.load_cache(cache_file)
        return sampler
    else:
        print("Creating new negative sampler...")
        sampler = PrecomputedNegativeSampler(kg_file, data_file)
        
        if cache_file:
            # 保存缓存
            Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
            sampler.save_cache(cache_file)
        
        return sampler

if __name__ == "__main__":
    # 测试用法
    kg_file = "graph/knowledge_graph.pkl"
    data_file = "query/qa_with_paths_cleaned.json"
    cache_file = "cache/negative_samples.pkl"
    
    sampler = create_negative_sampler(kg_file, data_file, cache_file)
    
    # 测试获取负样本
    negatives = sampler.get_negatives("sample_0", num_negatives=3)
    print(f"\nSample negatives: {negatives}")