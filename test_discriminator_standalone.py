#!/usr/bin/env python3
"""
判别器独立测试脚本

测试设计：
- 1hop, 2hop, 3hop 各200个样本
- 正样本：使用真实的正确路径
- 负样本：同跳数但不通往答案实体的路径（从真实图结构生成）
- 数据来源：qa_with_paths_cleaned.json 中 type 包含 'test' 的样本
"""

import sys
from pathlib import Path
import torch
import json
import pickle
import random
import numpy as np
from collections import defaultdict
from typing import List, Dict, Set, Tuple

# 添加模型路径
sys.path.append(str(Path(__file__).parent / "models" / "path_ranker"))
sys.path.append(str(Path(__file__).parent / "models" / "path_discover"))

from enhanced_path_ranker import EnhancedPathRankerDiscriminator

class DiscriminatorTester:
    """判别器独立测试器"""
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        self.discriminator = None
        self.knowledge_graph = None
        self.test_data = {'1hop': [], '2hop': [], '3hop': []}
        
    def load_models_and_data(self):
        """加载判别器和数据"""
        print("Loading discriminator and data...")
        
        # 1. 加载判别器
        try:
            self.discriminator = EnhancedPathRankerDiscriminator(
                freeze_sbert=True,
                use_pattern_memory=False
            )
            
            # 加载预训练权重
            checkpoint_path = "checkpoints/enhanced_pathranker/best_hits1_model.pth"
            print(f"Loading discriminator weights from: {checkpoint_path}")
            
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                
                # Extract model_state_dict if it's nested
                if 'model_state_dict' in checkpoint:
                    model_weights = checkpoint['model_state_dict']
                    print("Found nested model_state_dict in checkpoint")
                else:
                    model_weights = checkpoint
                    print("Using checkpoint directly as state_dict")
                
                self.discriminator.load_state_dict(model_weights, strict=False)
                print("Pre-trained weights loaded successfully")
            except Exception as e:
                print(f"Failed to load weights: {e}")
                print("Using randomly initialized weights instead")
            
            self.discriminator.to(self.device)
            self.discriminator.eval()
            print("Discriminator loaded successfully")
        except Exception as e:
            print(f"Failed to load discriminator: {e}")
            return False
        
        # 2. 加载知识图谱
        try:
            with open("graph/knowledge_graph.pkl", 'rb') as f:
                self.knowledge_graph = pickle.load(f)
            print(f"OK Knowledge graph loaded: {self.knowledge_graph.number_of_nodes()} nodes, {self.knowledge_graph.number_of_edges()} edges")
        except Exception as e:
            print(f"ERROR Failed to load knowledge graph: {e}")
            return False
        
        # 3. 加载测试数据
        return self.load_test_data()
    
    def load_test_data(self):
        """加载测试数据"""
        print("Loading test data from qa_with_paths_cleaned.json...")
        
        try:
            all_data = self._load_multiline_json("query/qa_with_paths_cleaned.json")
            if not all_data:
                print("ERROR No data loaded")
                return False
            
            # 筛选训练数据 (避免过拟合问题影响分析)
            for item in all_data:
                type_field = item.get('type', '')
                if 'test' in type_field:
                    if '1hop' in type_field:
                        self.test_data['1hop'].append(item)
                    elif '2hop' in type_field:
                        self.test_data['2hop'].append(item)
                    elif '3hop' in type_field:
                        self.test_data['3hop'].append(item)
            
            # 统计
            total_test = sum(len(data) for data in self.test_data.values())
            print("Test data loaded:")
            for hop_type, data in self.test_data.items():
                print(f"  {hop_type}: {len(data)} samples")
            print(f"  Total: {total_test} samples")
            
            return total_test > 0
            
        except Exception as e:
            print(f"ERROR Failed to load test data: {e}")
            return False
    
    def _load_multiline_json(self, file_path: str) -> List[Dict]:
        """加载多行JSON格式的文件"""
        try:
            # 尝试直接加载为JSON数组
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # 如果文件是JSON数组格式
                if content.strip().startswith('['):
                    return json.loads(content)
                
                # 如果是每个对象独立的格式，按行读取
                data = []
                lines = content.strip().split('\n')
                current_obj = ""
                brace_count = 0
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    current_obj += line + " "
                    brace_count += line.count('{') - line.count('}')
                    
                    if brace_count == 0 and current_obj.strip():
                        try:
                            obj = json.loads(current_obj.strip())
                            data.append(obj)
                            current_obj = ""
                        except json.JSONDecodeError as e:
                            print(f"JSON decode error: {e}")
                            current_obj = ""
                
                return data
                
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return []
    
    def generate_negative_samples(self, positive_sample: Dict, target_count: int = 5) -> List[List[str]]:
        """
        为给定的正样本生成负样本路径
        负样本：同跳数但不通往答案实体的路径
        """
        question_entity = positive_sample['question_entity']
        answer_entities = set(positive_sample['answer_entities'])
        
        # 确定跳数
        hop_count = 1
        if '2hop' in positive_sample.get('type', ''):
            hop_count = 2
        elif '3hop' in positive_sample.get('type', ''):
            hop_count = 3
        
        negative_paths = []
        attempts = 0
        max_attempts = target_count * 10
        
        while len(negative_paths) < target_count and attempts < max_attempts:
            attempts += 1
            
            try:
                path = self._generate_random_path(question_entity, hop_count)
                if path and len(path) >= 3:  # 至少要有起始实体 -> 关系 -> 终止实体
                    final_entity = path[-1]
                    # 确保不是正确答案
                    if final_entity not in answer_entities:
                        negative_paths.append(path)
            except:
                continue
        
        return negative_paths
    
    def _generate_random_path(self, start_entity: str, hop_count: int) -> List[str]:
        """生成指定跳数的随机路径"""
        if start_entity not in self.knowledge_graph.nodes:
            return []
        
        path = [start_entity]
        current_entity = start_entity
        
        for _ in range(hop_count):
            # 获取邻居
            neighbors = list(self.knowledge_graph.neighbors(current_entity))
            if not neighbors:
                break
            
            # 随机选择一个邻居
            next_entity = random.choice(neighbors)
            
            # 获取关系
            edge_data = self.knowledge_graph.get_edge_data(current_entity, next_entity)
            if edge_data:
                # 随机选择一个关系
                relations = []
                for key, data in edge_data.items():
                    if isinstance(data, dict):
                        relation = data.get('relation', key)
                    else:
                        relation = str(data) if data else key
                    relations.append(relation)
                
                if relations:
                    relation = random.choice(relations)
                    path.extend([relation, next_entity])
                    current_entity = next_entity
                else:
                    break
            else:
                break
        
        return path
    
    def create_cross_hop_test_dataset(self, samples_per_hop: int = 200):
        """创建跨跳数负样本测试数据集
        
        测试设计：
        - 1hop问题 -> 负样本使用2hop和3hop路径
        - 2hop问题 -> 负样本使用1hop和3hop路径  
        - 3hop问题 -> 负样本使用1hop和2hop路径
        """
        print(f"Creating CROSS-HOP negative samples test dataset ({samples_per_hop} samples per hop)...")
        print("Negative sample strategy: Different hop counts (e.g., 1hop question -> 2hop/3hop negative paths)")
        
        test_cases = []
        
        # 为每种跳数类型收集所有可用路径
        all_paths_by_hop = {'1hop': [], '2hop': [], '3hop': []}
        
        for hop_type in ['1hop', '2hop', '3hop']:
            available_data = self.test_data[hop_type]
            for sample in available_data:
                if 'paths' in sample and sample['paths']:
                    for entity, path_list in sample['paths'].items():
                        if entity in sample['answer_entities']:
                            for path_str in path_list[:2]:  # 每个实体取前2条路径
                                path = path_str.split('.')
                                if len(path) >= 3:
                                    all_paths_by_hop[hop_type].append({
                                        'path': path,
                                        'question': sample['question'],
                                        'final_entity': path[-1]
                                    })
        
        print(f"Available paths collected:")
        for hop, paths in all_paths_by_hop.items():
            print(f"  {hop}: {len(paths)} paths")
        
        # 为每种跳数创建测试用例
        for target_hop in ['1hop', '2hop', '3hop']:
            available_data = self.test_data[target_hop]
            if not available_data:
                continue
                
            selected_samples = random.sample(
                available_data, 
                min(samples_per_hop, len(available_data))
            )
            
            positive_count = 0
            negative_count = 0
            
            for sample in selected_samples:
                question = sample['question']
                question_entity = sample['question_entity']
                answer_entities = sample['answer_entities']
                
                # 添加正样本
                if 'paths' in sample and sample['paths']:
                    for entity, path_list in sample['paths'].items():
                        if entity in answer_entities:
                            for path_str in path_list[:1]:  # 每个问题只取1条正样本
                                path = path_str.split('.')
                                if len(path) >= 3:
                                    test_cases.append({
                                        'question': question,
                                        'path': path,
                                        'label': 1,  # 正样本
                                        'hop_type': target_hop,
                                        'sample_type': 'positive',
                                        'final_entity': path[-1],
                                        'is_correct': True
                                    })
                                    positive_count += 1
                                    break
                
                # 添加跨跳数负样本
                other_hops = [h for h in ['1hop', '2hop', '3hop'] if h != target_hop]
                
                for wrong_hop in other_hops:
                    if all_paths_by_hop[wrong_hop]:
                        # 随机选择2条不同跳数的路径作为负样本
                        neg_paths = random.sample(
                            all_paths_by_hop[wrong_hop], 
                            min(2, len(all_paths_by_hop[wrong_hop]))
                        )
                        
                        for neg_path_data in neg_paths:
                            test_cases.append({
                                'question': question,
                                'path': neg_path_data['path'],
                                'label': 0,  # 负样本
                                'hop_type': target_hop,
                                'sample_type': f'negative_{wrong_hop}',
                                'final_entity': neg_path_data['final_entity'],
                                'is_correct': False,
                                'mismatch_type': f"{target_hop}_question_with_{wrong_hop}_path"
                            })
                            negative_count += 1
            
            print(f"  {target_hop}: {positive_count} positive, {negative_count} negative samples")
        
        print(f"\nTotal test cases created: {len(test_cases)}")
        return test_cases

    def create_test_dataset(self, samples_per_hop: int = 200):
        """创建平衡的测试数据集"""
        print(f"Creating balanced test dataset ({samples_per_hop} samples per hop)...")
        
        test_cases = []
        
        for hop_type in ['1hop', '2hop', '3hop']:
            available_data = self.test_data[hop_type]
            if not available_data:
                print(f"WARNING  No {hop_type} test data available")
                continue
            
            # 随机选择样本
            selected_samples = random.sample(
                available_data, 
                min(samples_per_hop, len(available_data))
            )
            
            positive_count = 0
            negative_count = 0
            
            for sample in selected_samples:
                question = sample['question']
                question_entity = sample['question_entity']
                answer_entities = sample['answer_entities']
                
                # 从真实路径中提取正样本
                if 'paths' in sample and sample['paths']:
                    for entity, path_list in sample['paths'].items():
                        if entity in answer_entities:
                            for path_str in path_list:
                                # 解析路径字符串
                                path = path_str.split('.')
                                if len(path) >= 3:  # 有效路径
                                    test_cases.append({
                                        'question': question,
                                        'path': path,
                                        'label': 1,  # 正样本
                                        'hop_type': hop_type,
                                        'sample_type': 'positive',
                                        'final_entity': path[-1],
                                        'is_correct': path[-1] in answer_entities
                                    })
                                    positive_count += 1
                                    break  # 每个实体只取一条路径
                
                # 生成负样本
                negative_paths = self.generate_negative_samples(sample, target_count=3)
                for neg_path in negative_paths:
                    test_cases.append({
                        'question': question,
                        'path': neg_path,
                        'label': 0,  # 负样本
                        'hop_type': hop_type,
                        'sample_type': 'negative',
                        'final_entity': neg_path[-1],
                        'is_correct': neg_path[-1] in answer_entities
                    })
                    negative_count += 1
            
            print(f"  {hop_type}: {positive_count} positive, {negative_count} negative samples")
        
        print(f"\\nTotal test cases created: {len(test_cases)}")
        return test_cases
    
    def evaluate_discriminator(self, test_cases: List[Dict]):
        """评估判别器性能"""
        print("\\nEvaluating discriminator performance...")
        print("="*80)
        
        # 按跳数和样本类型分组统计
        stats = defaultdict(lambda: defaultdict(list))
        overall_stats = {'predictions': [], 'labels': [], 'confidences': []}
        
        batch_size = 32
        correct_predictions = 0
        total_predictions = 0
        
        # 分批处理
        for i in range(0, len(test_cases), batch_size):
            batch = test_cases[i:i+batch_size]
            
            # 准备批次数据
            questions = []
            path_data = []
            true_labels = []
            
            for case in batch:
                questions.append(case['question'])
                
                # 构造路径数据格式
                final_entity = case['final_entity']
                path_string = '.'.join(case['path'])
                path_data.append({
                    'paths': {final_entity: [path_string]}
                })
                true_labels.append(case['label'])
            
            # 判别器预测
            try:
                with torch.no_grad():
                    predictions = self.discriminator(questions, path_data, epoch=0)
                
                # 收集结果
                for j, pred in enumerate(predictions):
                    case = batch[j]
                    # 判别器返回字典格式，提取 aggregated_score
                    if isinstance(pred, dict):
                        score_tensor = pred.get('aggregated_score', 0.0)
                        if isinstance(score_tensor, torch.Tensor):
                            confidence = float(score_tensor.item())
                        else:
                            confidence = float(score_tensor)
                    else:
                        confidence = float(pred)
                    
                    # 将原始分数转换为概率 (使用sigmoid)
                    confidence = 1.0 / (1.0 + np.exp(-confidence))  # sigmoid
                    predicted_label = 1 if confidence > 0.5 else 0
                    true_label = true_labels[j]
                    
                    # 统计
                    hop_type = case['hop_type']
                    sample_type = case['sample_type']
                    
                    stats[hop_type][sample_type].append({
                        'confidence': confidence,
                        'predicted': predicted_label,
                        'true': true_label,
                        'correct': predicted_label == true_label
                    })
                    
                    # 全局统计
                    overall_stats['predictions'].append(predicted_label)
                    overall_stats['labels'].append(true_label)
                    overall_stats['confidences'].append(confidence)
                    
                    if predicted_label == true_label:
                        correct_predictions += 1
                    total_predictions += 1
                    
            except Exception as e:
                print(f"ERROR Batch {i//batch_size + 1} failed: {e}")
                continue
        
        # 计算和显示结果
        self._display_results(stats, overall_stats, correct_predictions, total_predictions)
    
    def _display_results(self, stats, overall_stats, correct_predictions, total_predictions):
        """显示评估结果"""
        print("\\nDISCRIMINATOR EVALUATION RESULTS")
        print("="*80)
        
        # 整体准确率
        overall_accuracy = correct_predictions / max(1, total_predictions)
        print(f"Overall Accuracy: {overall_accuracy:.3f} ({correct_predictions}/{total_predictions})")
        
        # 计算整体指标
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
        
        try:
            precision = precision_score(overall_stats['labels'], overall_stats['predictions'])
            recall = recall_score(overall_stats['labels'], overall_stats['predictions'])
            f1 = f1_score(overall_stats['labels'], overall_stats['predictions'])
            auc = roc_auc_score(overall_stats['labels'], overall_stats['confidences'])
            
            print(f"Precision: {precision:.3f}")
            print(f"Recall:    {recall:.3f}")
            print(f"F1-Score:  {f1:.3f}")
            print(f"AUC-ROC:   {auc:.3f}")
        except Exception as e:
            print(f"WARNING  Could not compute advanced metrics: {e}")
        
        print("\\nDETAILED BREAKDOWN BY HOP COUNT")
        print("-"*80)
        
        # 按跳数详细统计
        for hop_type in ['1hop', '2hop', '3hop']:
            if hop_type not in stats:
                continue
            
            print(f"\\n{hop_type.upper()} RESULTS:")
            
            for sample_type in ['positive', 'negative']:
                if sample_type not in stats[hop_type]:
                    continue
                
                samples = stats[hop_type][sample_type]
                if not samples:
                    continue
                
                # 计算指标
                total = len(samples)
                correct = sum(1 for s in samples if s['correct'])
                accuracy = correct / total
                
                avg_confidence = np.mean([s['confidence'] for s in samples])
                conf_std = np.std([s['confidence'] for s in samples])
                
                print(f"  {sample_type.capitalize():>8}: {accuracy:.3f} ({correct:>3}/{total:>3}) | "
                      f"Avg Conf: {avg_confidence:.3f}±{conf_std:.3f}")
        
        # 置信度分布分析
        print("\\nCONFIDENCE DISTRIBUTION ANALYSIS")
        print("-"*80)
        
        pos_confs = [s['confidence'] for hop_stats in stats.values() 
                    for s in hop_stats.get('positive', [])]
        neg_confs = [s['confidence'] for hop_stats in stats.values() 
                    for s in hop_stats.get('negative', [])]
        
        if pos_confs and neg_confs:
            print(f"Positive samples - Mean: {np.mean(pos_confs):.3f}, Std: {np.std(pos_confs):.3f}")
            print(f"Negative samples - Mean: {np.mean(neg_confs):.3f}, Std: {np.std(neg_confs):.3f}")
            print(f"Separation (pos_mean - neg_mean): {np.mean(pos_confs) - np.mean(neg_confs):.3f}")
        
        # 错误案例分析
        print("\\nERROR ANALYSIS")
        print("-"*80)
        
        false_positives = []
        false_negatives = []
        
        for hop_stats in stats.values():
            for sample_type, samples in hop_stats.items():
                for s in samples:
                    if not s['correct']:
                        if s['true'] == 0 and s['predicted'] == 1:
                            false_positives.append(s)
                        elif s['true'] == 1 and s['predicted'] == 0:
                            false_negatives.append(s)
        
        print(f"False Positives: {len(false_positives)} (negative samples predicted as positive)")
        if false_positives:
            fp_confs = [s['confidence'] for s in false_positives]
            print(f"  Avg confidence: {np.mean(fp_confs):.3f} (range: {min(fp_confs):.3f}-{max(fp_confs):.3f})")
        
        print(f"False Negatives: {len(false_negatives)} (positive samples predicted as negative)")
        if false_negatives:
            fn_confs = [s['confidence'] for s in false_negatives]
            print(f"  Avg confidence: {np.mean(fn_confs):.3f} (range: {min(fn_confs):.3f}-{max(fn_confs):.3f})")


def main():
    """主测试程序 - 跨跳数负样本测试"""
    print("DISCRIMINATOR CROSS-HOP EVALUATION (PRE-TRAINED WEIGHTS)")
    print("="*80)
    print("Testing discriminator on cross-hop negative samples:")
    print("- Model: Enhanced Path Ranker with pre-trained weights")
    print("- Weights: checkpoints/enhanced_pathranker/best_hits1_model.pth")
    print("- 1hop, 2hop, 3hop samples (200 each)")
    print("- Positive samples: Real correct paths")  
    print("- Negative samples: DIFFERENT hop counts (e.g., 1hop question -> 2hop/3hop paths)")
    print("- Data source: qa_with_paths_cleaned.json (TEST samples)")
    print("="*80)
    
    # 创建测试器
    tester = DiscriminatorTester()
    
    # 加载模型和数据
    if not tester.load_models_and_data():
        print("ERROR Failed to load models and data")
        return
    
    # 创建跨跳数测试数据集
    test_cases = tester.create_cross_hop_test_dataset(samples_per_hop=100)
    if not test_cases:
        print("ERROR Failed to create cross-hop test dataset")
        return
    
    # 评估判别器
    tester.evaluate_discriminator(test_cases)
    
    print("\\nDiscriminator evaluation completed!")


if __name__ == "__main__":
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    main()