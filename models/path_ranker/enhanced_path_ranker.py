import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import pickle
import networkx as nx
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
import random
import math

class PatternMemoryModule(nn.Module):
    """路径模式记忆模块 - 学习常见的关系序列模式"""
    
    def __init__(self, hidden_dim: int = 256, num_patterns: int = 50, pattern_dim: int = 128):
        super().__init__()
        
        self.num_patterns = num_patterns
        self.pattern_dim = pattern_dim
        self.hidden_dim = hidden_dim
        
        # 可学习的模式记忆库
        self.pattern_memory = nn.Parameter(torch.randn(num_patterns, pattern_dim))
        
        # 模式匹配网络
        self.pattern_matcher = nn.Sequential(
            nn.Linear(hidden_dim, pattern_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(pattern_dim)
        )
        
        # 模式权重计算
        self.pattern_attention = nn.MultiheadAttention(
            embed_dim=pattern_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # 模式融合
        self.pattern_fusion = nn.Sequential(
            nn.Linear(pattern_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
    
    def forward(self, path_repr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            path_repr: [hidden_dim] - 路径表示
        Returns:
            enhanced_repr: [hidden_dim] - 增强后的路径表示
        """
        # 将路径映射到模式空间
        query_pattern = self.pattern_matcher(path_repr)  # [pattern_dim]
        
        # 简单的相似度计算代替复杂的attention
        # 计算query与所有pattern的相似度
        similarities = torch.cosine_similarity(
            query_pattern.unsqueeze(0),  # [1, pattern_dim]
            self.pattern_memory,         # [num_patterns, pattern_dim]
            dim=1
        )  # [num_patterns]
        
        # 软注意力权重
        attention_weights = F.softmax(similarities, dim=0)  # [num_patterns]
        
        # 加权平均得到attended pattern
        attended_patterns = torch.sum(
            attention_weights.unsqueeze(1) * self.pattern_memory,  # [num_patterns, 1] * [num_patterns, pattern_dim]
            dim=0
        )  # [pattern_dim]
        
        # 融合原始路径表示和模式信息
        combined = torch.cat([attended_patterns, path_repr])  # [pattern_dim + hidden_dim]
        enhanced_repr = self.pattern_fusion(combined)
        
        return enhanced_repr

class CurriculumLearningScheduler:
    """课程学习调度器 - 根据训练进度调整难度权重"""
    
    def __init__(self, total_epochs: int, warmup_ratio: float = 0.2):
        self.total_epochs = total_epochs
        self.warmup_epochs = int(total_epochs * warmup_ratio)
        
    def get_difficulty_weights(self, epoch: int, hop_counts: List[int]) -> torch.Tensor:
        """
        根据epoch和路径跳数计算难度权重
        早期epoch偏向简单路径，后期处理复杂路径
        """
        progress = epoch / self.total_epochs
        
        weights = []
        for hop_count in hop_counts:
            if epoch < self.warmup_epochs:
                # Warmup阶段：简单路径权重高
                if hop_count == 1:
                    weight = 1.0
                elif hop_count == 2:
                    weight = 0.5
                else:
                    weight = 0.1
            else:
                # 正常训练：逐渐增加复杂路径权重
                curriculum_factor = min(1.0, (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs))
                
                if hop_count == 1:
                    weight = 1.0
                elif hop_count == 2:
                    weight = 0.5 + 0.5 * curriculum_factor
                else:
                    weight = 0.1 + 0.9 * curriculum_factor
            
            weights.append(weight)
        
        return torch.tensor(weights, dtype=torch.float32)

# 移除虚假的知识蒸馏类 - 没有真正的teacher模型
# class KnowledgeDistillationLoss(nn.Module): - 已删除

class EnhancedPathRankerDiscriminator(nn.Module):
    """
    增强版PathRanker: 解决你提到的问题
    
    新增功能:
    1. 路径模式学习 (PatternMemoryModule)
    2. 课程学习 (CurriculumLearningScheduler) 
    3. 知识蒸馏 (KnowledgeDistillationLoss)
    4. 难度自适应损失权重
    5. 统一embedding系统
    """
    
    def __init__(self, 
                 sbert_model_name: str = 'all-MiniLM-L6-v2',
                 hidden_dim: int = 256,
                 num_relations: int = 9,
                 use_pattern_memory: bool = True,
                 use_curriculum_learning: bool = True,
                 use_knowledge_distillation: bool = True,
                 freeze_sbert: bool = True,
                 entity_embedding_path: str = None):
        super().__init__()
        
        # 基础配置
        self.sbert_model_name = sbert_model_name
        self.sbert_dim = 384
        self.hidden_dim = hidden_dim
        self.use_pattern_memory = False  # 移除伪科学的PatternMemoryModule
        self.use_curriculum_learning = False  # 替换为多任务学习
        self.use_knowledge_distillation = False  # 禁用虚假的知识蒸馏
        self.use_multitask_learning = True  # 启用多任务学习
        
        # 初始化SBERT
        self.sbert = SentenceTransformer(sbert_model_name)
        if freeze_sbert:
            self._freeze_sbert()
        
        # 统一embedding系统
        self.entity_embedding_path = entity_embedding_path
        self.entity_embeddings = None
        if entity_embedding_path:
            self.load_entity_embeddings()
        
        # 关系模板
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
        
        # 实体和关系编码器
        self.entity_encoder = nn.Sequential(
            nn.Linear(self.sbert_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_dim)
        )
        
        self.relation_encoder = nn.Sequential(
            nn.Linear(self.sbert_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_dim)
        )
        
        # 路径序列编码器
        self.path_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
            bidirectional=True
        )
        
        # 自适应架构 - 不同跳数的专用transformer
        # 动态计算注意力头数量以适应hidden_dim * 2
        transformer_dim = hidden_dim * 2  # LSTM双向输出维度
        
        # 计算合适的注意力头数量（确保维度能被整除）
        def get_suitable_nhead(dim):
            for nhead in [16, 12, 8, 6, 4, 2, 1]:
                if dim % nhead == 0:
                    return nhead
            return 1
        
        suitable_nhead = get_suitable_nhead(transformer_dim)
        
        self.hop_configs = {
            1: {'nhead': suitable_nhead, 'num_layers': 1, 'dropout': 0.05, 'difficulty': 1.0},
            2: {'nhead': suitable_nhead, 'num_layers': 2, 'dropout': 0.1, 'difficulty': 1.5},
            3: {'nhead': suitable_nhead, 'num_layers': 3, 'dropout': 0.15, 'difficulty': 2.0},
            4: {'nhead': suitable_nhead, 'num_layers': 3, 'dropout': 0.2, 'difficulty': 2.5}
        }
        
        print(f"[INFO] Transformer dim: {transformer_dim}, suitable nhead: {suitable_nhead}")
        
        # 创建hop-specific transformers
        self.hop_transformers = nn.ModuleDict()
        self.hop_scorers = nn.ModuleDict()
        
        for hop_count, config in self.hop_configs.items():
            # Transformer
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=transformer_dim,  # 使用动态计算的维度
                nhead=config['nhead'],
                dim_feedforward=hidden_dim * 4,
                dropout=config['dropout'],
                batch_first=True
            )
            self.hop_transformers[str(hop_count)] = nn.TransformerEncoder(
                encoder_layer, num_layers=config['num_layers']
            )
            
            # 难度自适应评分器
            self.hop_scorers[str(hop_count)] = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(config['dropout']),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()  # 输出0-1分数
            )
        
        # 移除PatternMemoryModule - 没有理论依据的伪科学设计
        # 保留简单的路径表示学习
        
        # 查询编码器
        self.query_encoder = nn.Sequential(
            nn.Linear(self.sbert_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # 最终评分网络
        # enhanced_repr: hidden_dim*2, query_repr: hidden_dim, path_repr: hidden_dim*2
        # 总维度: hidden_dim*2 + hidden_dim + hidden_dim*2 = hidden_dim*5
        self.final_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim * 2),  # path + query + enhanced
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 难度自适应权重
        self.difficulty_weights = nn.Parameter(torch.tensor([1.0, 1.5, 2.0, 2.5]))
        
        # 多任务学习 - hop-specific判别头（相同架构，仅权重不同）
        self.hop_discriminators = nn.ModuleDict({
            '1': nn.Sequential(
                nn.Linear(hidden_dim * 5, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, 1)
            ),
            '2': nn.Sequential(
                nn.Linear(hidden_dim * 5, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, 1)
            ),
            '3': nn.Sequential(
                nn.Linear(hidden_dim * 5, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, 1)
            )
        })
        
        # 移除虚假的知识蒸馏和课程学习 - 使用多任务学习
        # 课程学习已替换为多任务学习
    
    def _freeze_sbert(self):
        """冻结SBERT参数"""
        for param in self.sbert.parameters():
            param.requires_grad = False
        print("[INFO] SBERT parameters frozen")
    
    def load_entity_embeddings(self):
        """加载统一的实体embedding文件"""
        if self.entity_embedding_path and Path(self.entity_embedding_path).exists():
            print(f"Loading entity embeddings from {self.entity_embedding_path}")
            self.entity_embeddings = torch.load(self.entity_embedding_path)
            print(f"Loaded embeddings for {len(self.entity_embeddings)} entities")
        else:
            print("No entity embedding file found, will use SBERT on-the-fly")
    
    def encode_entities(self, entities: List[str]) -> torch.Tensor:
        """编码实体列表，修复梯度泄漏问题"""
        if not entities:
            return torch.empty(0, self.hidden_dim, device=next(self.parameters()).device)
        
        device = next(self.parameters()).device
        
        # 尝试使用预加载的embeddings
        if self.entity_embeddings:
            entity_embeds = []
            for entity in entities:
                if entity in self.entity_embeddings:
                    entity_embeds.append(self.entity_embeddings[entity].clone().detach().to(device))
                else:
                    # Fallback to SBERT - 确保梯度隔离
                    with torch.no_grad():
                        embed = self.sbert.encode(entity, convert_to_tensor=True).detach().to(device)
                    entity_embeds.append(embed)
            
            if entity_embeds:
                raw_embeds = torch.stack(entity_embeds).to(device)
            else:
                raw_embeds = torch.empty(0, self.sbert_dim, device=device)
        else:
            # 使用SBERT实时编码 - 确保梯度隔离
            with torch.no_grad():
                raw_embeds = self.sbert.encode(entities, convert_to_tensor=True)
            raw_embeds = raw_embeds.to(device)
        
        if raw_embeds.requires_grad is False:
            raw_embeds = raw_embeds.clone().requires_grad_(True)
        
        return self.entity_encoder(raw_embeds)
    
    def encode_relations(self, relations: List[str]) -> torch.Tensor:
        """编码关系列表"""
        if not relations:
            return torch.empty(0, self.hidden_dim, device=next(self.parameters()).device)
        
        device = next(self.parameters()).device
        
        # 使用关系模板
        relation_texts = [self.relation_templates.get(r, r) for r in relations]
        
        with torch.no_grad():
            raw_embeds = self.sbert.encode(relation_texts, convert_to_tensor=True)
        raw_embeds = raw_embeds.to(device)
        
        if raw_embeds.requires_grad is False:
            raw_embeds = raw_embeds.clone().requires_grad_(True)
        
        return self.relation_encoder(raw_embeds)
    
    def parse_path_string(self, path_string: str) -> Tuple[List[str], List[str]]:
        """解析路径字符串为实体和关系列表"""
        if not path_string:
            return [], []
        
        parts = path_string.split('.')
        entities = parts[::2]  # 偶数索引是实体
        relations = parts[1::2]  # 奇数索引是关系
        
        return entities, relations
    
    def get_hop_count(self, relations: List[str]) -> int:
        """获取跳数"""
        return len(relations)
    
    def get_adaptive_components(self, hop_count: int) -> Tuple[nn.Module, nn.Module]:
        """获取自适应组件"""
        if hop_count <= 1:
            key = '1'
        elif hop_count == 2:
            key = '2'
        elif hop_count == 3:
            key = '3'
        else:
            key = '4'
        
        # ModuleDict使用[]访问，不是get方法
        transformer = self.hop_transformers[key] if key in self.hop_transformers else self.hop_transformers['2']
        scorer = self.hop_scorers[key] if key in self.hop_scorers else self.hop_scorers['2']
        
        return transformer, scorer
    
    def compute_difficulty_weight(self, hop_count: int) -> float:
        """计算路径难度权重"""
        difficulty_config = self.hop_configs.get(hop_count, self.hop_configs[4])
        return difficulty_config['difficulty']
    
    def forward_single_path(self, question: str, path_string: str, 
                           epoch: int = 0, use_teacher_forcing: bool = False) -> Dict:
        """处理单个路径"""
        device = next(self.parameters()).device
        
        # 解析路径
        entities, relations = self.parse_path_string(path_string)
        
        if len(entities) < 2:
            return {
                'path_score': torch.tensor(0.0, device=device),
                'hop_count': 0,
                'difficulty_weight': 1.0
            }
        
        # 编码查询
        with torch.no_grad():
            query_embed = self.sbert.encode(question, convert_to_tensor=True)
        query_embed = query_embed.to(device)
        if query_embed.requires_grad is False:
            query_embed = query_embed.clone().requires_grad_(True)
        
        query_repr = self.query_encoder(query_embed)
        
        # 编码路径元素
        entity_embeds = self.encode_entities(entities)
        relation_embeds = self.encode_relations(relations)
        
        # 创建交替序列: entity -> relation -> entity -> ...
        sequence = []
        for i in range(len(entities)):
            sequence.append(entity_embeds[i])
            if i < len(relations):
                sequence.append(relation_embeds[i])
        
        if not sequence:
            return {
                'path_score': torch.tensor(0.0, device=device),
                'hop_count': 0,
                'difficulty_weight': 1.0
            }
        
        path_sequence = torch.stack(sequence).unsqueeze(0)  # [1, seq_len, hidden_dim]
        
        # LSTM编码
        lstm_out, _ = self.path_lstm(path_sequence)  # [1, seq_len, hidden_dim*2]
        
        # 获取路径表示
        path_repr = lstm_out.mean(dim=1).squeeze(0)  # [hidden_dim*2]
        
        # 跳数和自适应组件
        hop_count = self.get_hop_count(relations)
        transformer, scorer = self.get_adaptive_components(hop_count)
        
        # 应用hop-specific transformer
        transformed_repr = transformer(lstm_out).mean(dim=1).squeeze(0)  # [hidden_dim*2]
        
        # 直接使用transformer输出，无需伪科学的pattern memory
        enhanced_repr = transformed_repr
        
        # 多任务学习: 使用hop-specific判别头
        combined_repr = torch.cat([enhanced_repr, query_repr, path_repr])  # [hidden_dim*5]
        
        if self.use_multitask_learning and str(hop_count) in self.hop_discriminators:
            # 使用hop-specific判别头
            hop_discriminator = self.hop_discriminators[str(hop_count)]
            final_score = hop_discriminator(combined_repr.unsqueeze(0)).squeeze()
        else:
            # fallback到通用判别头
            final_score = self.final_scorer(combined_repr).squeeze()
        
        # 雾度自适应(保留)
        difficulty_weight = self.compute_difficulty_weight(hop_count)
        
        return {
            'path_score': final_score,
            'raw_score': final_score,
            'hop_count': hop_count,
            'difficulty_weight': difficulty_weight,
            'path_representation': enhanced_repr
        }
    
    def forward(self, questions: List[str], path_data: List[Dict], 
                epoch: int = 0) -> List[Dict]:
        """批量前向传播"""
        results = []
        
        for question, data in zip(questions, path_data):
            # 处理每个问题的所有候选路径
            candidate_scores = []
            candidate_details = []
            
            # 从路径字典中获取所有路径
            paths = data.get('paths', {})
            
            for answer_entity, path_strings in paths.items():
                for path_string in path_strings:
                    result = self.forward_single_path(question, path_string, epoch)
                    candidate_scores.append(result['path_score'])
                    candidate_details.append({
                        **result,
                        'answer_entity': answer_entity,
                        'path_string': path_string
                    })
            
            if candidate_scores:
                individual_scores = torch.stack(candidate_scores)
                
                # 聚合分数 (可以使用最大值或平均值)
                aggregated_score = individual_scores.max()
            else:
                device = next(self.parameters()).device
                individual_scores = torch.empty(0, device=device)
                aggregated_score = torch.tensor(0.0, device=device)
            
            results.append({
                'aggregated_score': aggregated_score,
                'individual_scores': individual_scores,
                'path_details': candidate_details,
                'num_paths': len(candidate_scores)
            })
        
        return results
    
    def get_hop_discriminator(self, hop_count: int):
        """获取hop-specific判别头"""
        hop_key = str(min(hop_count, 3))  # 最大支持3hop
        return self.hop_discriminators.get(hop_key, self.hop_discriminators['3'])
    
    def get_trainable_parameters(self):
        """获取可训练参数"""
        trainable_params = []
        total_params = 0
        trainable_count = 0
        
        for name, param in self.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params.append(param)
                trainable_count += param.numel()
        
        frozen_ratio = (total_params - trainable_count) / total_params
        print(f"[INFO] Enhanced PathRanker parameters: {trainable_count:,} trainable, {total_params:,} total")
        print(f"[INFO] Frozen parameters: {frozen_ratio:.1%}")
        
        return trainable_params

# 创建统一实体embedding的工具函数
def create_entity_embeddings(entity_names_file: str, output_path: str, 
                           sbert_model_name: str = 'all-MiniLM-L6-v2'):
    """创建统一的实体embedding文件"""
    print(f"Creating entity embeddings from {entity_names_file}")
    
    # 加载实体名称
    with open(entity_names_file, 'r', encoding='utf-8') as f:
        entity_names = json.load(f)
    
    print(f"Found {len(entity_names)} entities")
    
    # 初始化SBERT
    sbert = SentenceTransformer(sbert_model_name)
    
    # 批量编码
    batch_size = 256
    entity_embeddings = {}
    
    print("Encoding entities...")
    for i in range(0, len(entity_names), batch_size):
        batch_entities = entity_names[i:i+batch_size]
        batch_embeds = sbert.encode(batch_entities, convert_to_tensor=True)
        
        for entity, embed in zip(batch_entities, batch_embeds):
            entity_embeddings[entity] = embed
        
        if (i + batch_size) % 1000 == 0:
            print(f"  Processed {min(i + batch_size, len(entity_names))}/{len(entity_names)} entities")
    
    # 保存
    torch.save(entity_embeddings, output_path)
    print(f"Saved entity embeddings to {output_path}")
    
    return entity_embeddings

if __name__ == "__main__":
    print("Testing Enhanced PathRanker...")
    
    # 创建实体embeddings
    entity_names_file = "graph/entity_names.json"
    embedding_output = "embeddings/entity_embeddings.pt"
    
    if not Path(embedding_output).exists():
        Path("embeddings").mkdir(exist_ok=True)
        create_entity_embeddings(entity_names_file, embedding_output)
    
    # 测试模型
    model = EnhancedPathRankerDiscriminator(
        entity_embedding_path=embedding_output,
        use_pattern_memory=True,
        use_curriculum_learning=True,
        use_knowledge_distillation=True
    )
    
    print("✅ Enhanced PathRanker initialized successfully!")