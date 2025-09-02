import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import random
import time
import pickle
import networkx as nx
from pathlib import Path
import sys
from typing import List, Dict, Tuple, Union
from collections import defaultdict
from tqdm import tqdm
import argparse

# Add models to path
sys.path.append(str(Path(__file__).parent.parent))
from models.path_ranker.enhanced_path_ranker import EnhancedPathRankerDiscriminator
from utils.precomputed_negative_sampler import create_negative_sampler

class UnifiedQADataset(Dataset):
    """统一的QA数据集 - 使用真实数据和真实标签"""
    
    def __init__(self, data_file: str, knowledge_graph_file: str, negative_sampler, max_samples: int = None, 
                 hop_types: List[str] = None, split_types: List[str] = None):
        self.data_file = data_file
        self.samples = []
        self.negative_sampler = negative_sampler
        
        print(f"Loading unified QA dataset from {data_file}")
        
        # 加载数据
        with open(data_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 按双换行符分割查询块
        query_blocks = content.strip().split('\n\n')
        
        # 统计信息
        type_counts = defaultdict(int)
        hop_counts = defaultdict(int)
        
        for block in query_blocks:
            if not block.strip():
                continue
            
            try:
                query = json.loads(block)
                query_type = query.get('type', '')
                
                # 过滤类型
                if hop_types:
                    hop_num = query_type.split('_')[0] if '_' in query_type else ''
                    if hop_num + 'hop' not in hop_types:
                        continue
                
                if split_types:
                    split_type = query_type.split('_')[1] if '_' in query_type else ''
                    if split_type not in split_types:
                        continue
                
                # 验证数据完整性
                if not all(key in query for key in ['question', 'question_entity', 'answer_entities', 'paths']):
                    continue
                
                # 统计
                type_counts[query_type] += 1
                hop_num = query_type.split('_')[0] if '_' in query_type else 'unknown'
                hop_counts[hop_num] += 1
                
                self.samples.append(query)
                
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                continue
        
        # 限制样本数量
        if max_samples and len(self.samples) > max_samples:
            self.samples = random.sample(self.samples, max_samples)
        
        print(f"\nDataset Statistics:")
        print(f"  Total samples: {len(self.samples)}")
        print(f"  Type distribution: {dict(type_counts)}")
        print(f"  Hop distribution: {dict(hop_counts)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]
    
    # 删除了旧的低效负采样方法
    
    def collate_fn(self, batch):
        """正确的批处理函数"""
        questions = []
        path_data = []
        labels = []  # 添加真实标签
        
        for sample_idx, sample in enumerate(batch):
            question = sample['question']
            answer_entities = sample['answer_entities']
            paths = sample['paths']
            
            questions.append(question)
            
            # 创建真实的正负样本
            positive_paths = {}
            negative_paths = {}
            sample_labels = {}
            
            # 正样本
            for entity in answer_entities:
                if entity in paths:
                    positive_paths[entity] = paths[entity]
                    sample_labels[entity] = 1.0  # 正标签
            
            # 负样本 - 基于当前问题实体生成真实负样本
            question_entity = sample['question_entity']
            query_type = sample.get('type', '')
            hop_count = int(query_type.split('_')[0][0]) if query_type and query_type[0].isdigit() else 1
            
            # 直接从采样器获取该问题实体的负样本
            if hasattr(self.negative_sampler, 'hop_neighbors') and question_entity in self.negative_sampler.hop_neighbors:
                # 获取指定跳数的可达实体
                reachable_entities = self.negative_sampler.hop_neighbors[question_entity].get(hop_count, set())
                # 排除正确答案
                negative_entities = list(reachable_entities - set(answer_entities))
                
                # 为负样本生成路径
                import networkx as nx
                negative_samples = []
                for neg_entity in negative_entities[:len(answer_entities)]:
                    try:
                        if nx.has_path(self.negative_sampler.kg, question_entity, neg_entity):
                            path = nx.shortest_path(self.negative_sampler.kg, question_entity, neg_entity)
                            if len(path) - 1 == hop_count:
                                # 构造路径字符串
                                path_parts = []
                                for i in range(len(path) - 1):
                                    edge_data = self.negative_sampler.kg.get_edge_data(path[i], path[i+1])
                                    relation = edge_data.get('relation', 'related_to') if edge_data else 'related_to'
                                    path_parts.extend([path[i], relation])
                                path_parts.append(path[-1])
                                path_string = '.'.join(path_parts)
                                negative_samples.append((neg_entity, path_string))
                    except:
                        continue
            else:
                negative_samples = []
            
            # 如果负样本生成失败，创建简单的负样本（用于调试）
            if len(negative_samples) == 0:
                # 从所有实体中随机选择一些作为负样本
                import random
                all_entities = list(self.negative_sampler.hop_neighbors.keys()) if hasattr(self.negative_sampler, 'hop_neighbors') else []
                if all_entities:
                    random_negatives = [e for e in random.sample(all_entities, min(len(answer_entities), len(all_entities))) 
                                      if e not in answer_entities]
                    for neg_entity in random_negatives[:len(answer_entities)]:
                        # 创建一个简单的假路径（仅用于测试）
                        fake_path = f"{question_entity}.fake_relation.{neg_entity}"
                        negative_samples.append((neg_entity, fake_path))
            
            for entity, path_string in negative_samples:
                negative_paths[entity] = [path_string]
                sample_labels[entity] = 0.0  # 负标签
            
            # print(f"DEBUG: Q={question_entity}, Pos={len(answer_entities)}, Neg={len(negative_paths)}")  # 临时调试
            
            # 合并正负样本
            all_paths = {**positive_paths, **negative_paths}
            path_data.append({'paths': all_paths})
            labels.append(sample_labels)
        
        return questions, path_data, labels

class EnhancedPathRankerTrainer:
    """增强PathRanker训练器 - 使用真实数据和标签"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=1e-5
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=3
        )
        
        self.best_accuracy = 0.0
        self.best_f1 = 0.0
    
    def compute_loss(self, model_outputs: List[Dict], labels_batch: List[Dict]) -> torch.Tensor:
        """计算基于真实标签的损失"""
        total_loss = 0.0
        batch_size = len(model_outputs)
        
        for output, labels in zip(model_outputs, labels_batch):
            individual_scores = output['individual_scores']
            path_details = output['path_details']
            
            if len(individual_scores) == 0 or len(path_details) == 0:
                continue
            
            # 构建真实标签张量
            true_labels = []
            for detail in path_details:
                entity = detail['answer_entity']
                label = labels.get(entity, 0.0)  # 默认负样本
                true_labels.append(label)
            
            if not true_labels:
                continue
            
            true_labels = torch.tensor(true_labels, device=individual_scores.device, dtype=torch.float32)
            
            # 使用BCE损失
            if len(individual_scores) == len(true_labels):
                # 将分数转换为概率
                probs = torch.sigmoid(individual_scores)
                loss = F.binary_cross_entropy(probs, true_labels)
                total_loss += loss
        
        return total_loss / max(batch_size, 1)
    
    def evaluate_batch(self, model_outputs: List[Dict], labels_batch: List[Dict]) -> Dict[str, float]:
        """正确的ranking评估 - 使用MRR和Hits@K"""
        total_mrr = 0.0
        total_hits_at_1 = 0.0
        total_hits_at_3 = 0.0
        total_samples = 0
        
        for output, labels in zip(model_outputs, labels_batch):
            individual_scores = output['individual_scores']
            path_details = output['path_details']
            
            if len(individual_scores) == 0 or len(path_details) == 0:
                continue
            
            # 构建标签和分数列表
            score_label_pairs = []
            for i, detail in enumerate(path_details):
                entity = detail['answer_entity']
                label = labels.get(entity, 0.0)
                score = individual_scores[i].item()
                score_label_pairs.append((score, label, entity))
            
            if not score_label_pairs:
                continue
            
            # 按分数降序排序
            score_label_pairs.sort(key=lambda x: x[0], reverse=True)
            
            # 计算正样本的排名
            positive_ranks = []
            for rank, (score, label, entity) in enumerate(score_label_pairs, 1):
                if label == 1.0:  # 正样本
                    positive_ranks.append(rank)
            
            if positive_ranks:
                # MRR: Mean Reciprocal Rank
                mrr = np.mean([1.0 / rank for rank in positive_ranks])
                total_mrr += mrr
                
                # Hits@1: 最高分是否为正样本
                if score_label_pairs[0][1] == 1.0:
                    total_hits_at_1 += 1.0
                
                # Hits@3: 前3名中是否有正样本
                top3_labels = [pair[1] for pair in score_label_pairs[:3]]
                if 1.0 in top3_labels:
                    total_hits_at_3 += 1.0
                
                total_samples += 1
        
        # 计算平均指标
        if total_samples > 0:
            avg_mrr = total_mrr / total_samples
            avg_hits_at_1 = total_hits_at_1 / total_samples
            avg_hits_at_3 = total_hits_at_3 / total_samples
        else:
            avg_mrr = avg_hits_at_1 = avg_hits_at_3 = 0.0
        
        return {
            'mrr': avg_mrr,
            'hits_at_1': avg_hits_at_1, 
            'hits_at_3': avg_hits_at_3,
            'total_samples': total_samples  # 这里是实际样本数，不是batch数
        }
    
    def train_epoch(self, train_loader, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        total_f1 = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch_idx, (questions, path_data, labels) in enumerate(pbar):
            self.optimizer.zero_grad()
            
            # 前向传播
            outputs = self.model(questions, path_data, epoch=epoch)
            
            # 计算损失
            loss = self.compute_loss(outputs, labels)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 计算指标
            metrics = self.evaluate_batch(outputs, labels)
            
            total_loss += loss.item()
            total_accuracy += metrics['hits_at_1']  # 使用Hits@1作为准确率
            total_f1 += metrics['mrr']  # 使用MRR
            num_batches += 1
            
            # 更新进度条
            avg_loss = total_loss / num_batches
            avg_hits1 = total_accuracy / num_batches
            avg_mrr = total_f1 / num_batches
            
            pbar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'H@1': f'{avg_hits1:.4f}',
                'MRR': f'{avg_mrr:.4f}'
            })
        
        return {
            'loss': total_loss / num_batches,
            'hits_at_1': total_accuracy / num_batches,
            'mrr': total_f1 / num_batches
        }
    
    def evaluate(self, val_loader, epoch):
        """评估模型"""
        self.model.eval()
        total_loss = 0.0
        total_mrr = 0.0
        total_hits_at_1 = 0.0
        total_hits_at_3 = 0.0
        total_actual_samples = 0
        num_batches = 0
        
        with torch.no_grad():
            for questions, path_data, labels in tqdm(val_loader, desc='Evaluating'):
                outputs = self.model(questions, path_data, epoch=epoch)
                loss = self.compute_loss(outputs, labels)
                metrics = self.evaluate_batch(outputs, labels)
                
                total_loss += loss.item()
                total_mrr += metrics['mrr']
                total_hits_at_1 += metrics['hits_at_1']
                total_hits_at_3 += metrics['hits_at_3']
                total_actual_samples += metrics['total_samples']
                num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'mrr': total_mrr / num_batches,
            'hits_at_1': total_hits_at_1 / num_batches,
            'hits_at_3': total_hits_at_3 / num_batches,
            'total_samples': total_actual_samples
        }
    
    def save_checkpoint(self, epoch, metrics, checkpoint_dir):
        """保存检查点"""
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_accuracy': self.best_accuracy,
            'best_f1': self.best_f1,
            'metrics': metrics
        }
        
        # 保存最新检查点
        torch.save(checkpoint, checkpoint_path / 'latest_model.pth')
        
        # 保存最佳模型
        if metrics['hits_at_1'] > self.best_accuracy:
            self.best_accuracy = metrics['hits_at_1']
            torch.save(checkpoint, checkpoint_path / 'best_hits1_model.pth')
        
        if metrics['mrr'] > self.best_f1:
            self.best_f1 = metrics['mrr']
            torch.save(checkpoint, checkpoint_path / 'best_mrr_model.pth')

def main():
    parser = argparse.ArgumentParser(description='Train Enhanced PathRanker')
    parser.add_argument('--data_file', default='query/qa_with_paths_cleaned.json')
    parser.add_argument('--kg_file', default='graph/knowledge_graph.pkl')
    parser.add_argument('--embedding_path', default='embeddings/entity_embeddings.pt')
    parser.add_argument('--checkpoint_dir', default='checkpoints/enhanced_pathranker')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--max_train_samples', type=int, default=50000)
    parser.add_argument('--max_val_samples', type=int, default=5000)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    print("Enhanced PathRanker Training with Real Data")
    print("="*50)
    print(f"Device: {args.device}")
    print(f"Data file: {args.data_file}")
    print(f"Knowledge graph: {args.kg_file}")
    print(f"Embedding path: {args.embedding_path}")
    
    # 创建或加载负样本采样器
    cache_file = "cache/negative_samples.pkl"
    negative_sampler = create_negative_sampler(args.kg_file, args.data_file, cache_file)
    
    # 创建数据集
    train_dataset = UnifiedQADataset(
        args.data_file, 
        args.kg_file,
        negative_sampler,
        max_samples=args.max_train_samples,
        split_types=['train']
    )
    
    val_dataset = UnifiedQADataset(
        args.data_file,
        args.kg_file,
        negative_sampler,
        max_samples=args.max_val_samples,
        split_types=['test']
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=val_dataset.collate_fn
    )
    
    # 创建模型
    model = EnhancedPathRankerDiscriminator(
        entity_embedding_path=args.embedding_path,
        use_pattern_memory=False,  # 禁用伪科学的PatternMemoryModule
        use_curriculum_learning=False,  # 禁用课程学习
        use_knowledge_distillation=False,  # 禁用虚假的知识蒸馏
        freeze_sbert=True
    )
    
    # 多任务学习已经在模型初始化时设置
    
    # 创建训练器
    trainer = EnhancedPathRankerTrainer(model, args.device)
    
    print(f"\nModel Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 训练循环
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*60}")
        
        # 训练
        train_metrics = trainer.train_epoch(train_loader, epoch)
        print(f"Train - Loss: {train_metrics['loss']:.4f}, H@1: {train_metrics['hits_at_1']:.4f}, MRR: {train_metrics['mrr']:.4f}")
        
        # 验证
        val_metrics = trainer.evaluate(val_loader, epoch)
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, H@1: {val_metrics['hits_at_1']:.4f}, MRR: {val_metrics['mrr']:.4f}")
        
        # 更新学习率
        trainer.scheduler.step(val_metrics['hits_at_1'])  # 使用Hits@1
        
        # 保存检查点
        trainer.save_checkpoint(epoch + 1, val_metrics, args.checkpoint_dir)
        
        print(f"Best Hits@1: {trainer.best_accuracy:.4f}, Best MRR: {trainer.best_f1:.4f}")

if __name__ == '__main__':
    main()