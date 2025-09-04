"""
PASE对抗训练系统
Progressive Adaptive Subgraph Expansion with Discriminator Training

实现PASE生成器与判别器的完整对抗训练循环：
1. PASE生成路径候选
2. 检测判别器被骗情况  
3. Ground Truth纠错判别器
4. 形成对抗循环
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import json
import random
import time
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Set, Tuple
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 添加模型路径
sys.path.append(str(Path(__file__).parent / "models"))
sys.path.append(str(Path(__file__).parent / "models" / "path_ranker"))

from models.pase_generator import PASEGenerator, create_pase_generator
from enhanced_path_ranker import EnhancedPathRankerDiscriminator


class PASEAdversarialTrainer:
    """
    PASE对抗训练器
    
    核心机制：
    1. PASE生成多样化路径
    2. 检测判别器被"骗"的情况（高分但错误）
    3. 用Ground Truth纠正判别器
    4. 形成"探索-验证-纠错"循环
    """
    
    def __init__(self, pase_generator, discriminator, device='cuda'):
        self.pase_generator = pase_generator
        self.discriminator = discriminator
        self.device = device
        
        # 判别器优化器
        self.discriminator_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=1e-4,
            weight_decay=1e-5
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.discriminator_optimizer, mode='max', factor=0.5, patience=3
        )
        
        # 对抗统计
        self.adversarial_stats = {
            'pase_fooled_discriminator': 0,
            'discriminator_corrections': 0,
            'total_explorations': 0,
            'correct_generations': 0,
            'correction_history': []
        }
        
        # 全局样本收集（用于最终统一训练）
        self.global_positive_samples = []
        self.global_fooled_cases = []
        
        # 并行处理锁
        self.lock = threading.Lock()
        
        # 训练日志
        self.training_log = {
            'training_config': {},
            'epoch_metrics': [],
            'final_summary': {}
        }
        
    def collect_pase_exploration_samples(self, qa_dataset: List[Dict]) -> Tuple[List, List]:
        """
        收集PASE探索样本
        
        关键：使用PASE生成路径，检测判别器被骗情况
        """
        print(f"PASE探索样本收集中... ({len(qa_dataset)} 个查询)")
        
        positive_samples = []      # 正确路径样本
        fooled_cases = []         # 判别器被骗样本
        
        exploration_stats = {
            'total_paths': 0,
            'correct_paths': 0,
            'wrong_paths': 0,
            'discriminator_fooled': 0,
            'discriminator_correct': 0
        }
        
        for i, qa_item in enumerate(qa_dataset):
            if (i + 1) % 50 == 0:
                print(f"  进度: {i+1}/{len(qa_dataset)}")
            
            question = qa_item['question']
            start_entity = qa_item['question_entity']
            answer_entities = set(qa_item['answer_entities'])
            
            try:
                # 使用PASE生成路径
                path, final_hop, confidence = self.pase_generator.progressive_adaptive_subgraph_expansion(
                    question, start_entity, verbose=False
                )
                
                if len(path) == 0:
                    continue
                
                exploration_stats['total_paths'] += 1
                final_entity = path[-1]
                is_gt_correct = final_entity in answer_entities
                
                # 获取判别器对此路径的评分
                path_string = '.'.join(path)
                path_data = [{'paths': {final_entity: [path_string]}}]
                
                with torch.no_grad():
                    disc_output = self.discriminator([question], path_data, epoch=0)
                    disc_raw_score = float(disc_output[0]['individual_scores'][0])
                    disc_confidence = torch.sigmoid(torch.tensor(disc_raw_score)).item()
                
                # 分析结果
                if is_gt_correct:
                    exploration_stats['correct_paths'] += 1
                    # 正样本用于判别器训练
                    positive_samples.append((question, path, 'positive'))
                else:
                    exploration_stats['wrong_paths'] += 1
                
                # 关键：检测判别器被骗情况
                discriminator_threshold = 0.8  # 提高阈值到0.8
                if disc_confidence > discriminator_threshold and not is_gt_correct:
                    # PASE成功"骗"了判别器！
                    exploration_stats['discriminator_fooled'] += 1
                    self.adversarial_stats['pase_fooled_discriminator'] += 1
                    
                    fooled_cases.append({
                        'question': question,
                        'path': path,
                        'pase_confidence': confidence,
                        'discriminator_confidence': disc_confidence,
                        'ground_truth': False,
                        'label': 'pase_fooled_discriminator'
                    })
                    
                    print(f"    [PASE-FOOLED] 第{i+1}个查询: PASE骗过了判别器!")
                    print(f"      路径: {' -> '.join(path[-4:])}")
                    print(f"      PASE置信度: {confidence:.3f}")
                    print(f"      判别器置信度: {disc_confidence:.3f}")
                    print(f"      实际正确性: {is_gt_correct}")
                    
                else:
                    exploration_stats['discriminator_correct'] += 1
                
            except Exception as e:
                print(f"    处理第{i+1}个查询失败: {e}")
                continue
        
        # 统计报告
        total = exploration_stats['total_paths']
        print(f"PASE探索完成:")
        print(f"  生成路径总数: {total}")
        print(f"  正确路径: {exploration_stats['correct_paths']} ({exploration_stats['correct_paths']/max(1,total)*100:.1f}%)")
        print(f"  错误路径: {exploration_stats['wrong_paths']} ({exploration_stats['wrong_paths']/max(1,total)*100:.1f}%)")
        print(f"  判别器被骗: {exploration_stats['discriminator_fooled']} 次")
        print(f"  判别器正确: {exploration_stats['discriminator_correct']} 次")
        
        # 将样本添加到全局收集中（不在每个epoch训练，最后统一训练）
        self.global_positive_samples.extend(positive_samples)
        self.global_fooled_cases.extend(fooled_cases)
        
        return positive_samples, fooled_cases
    
    def train_discriminator_with_pase_correction(self, positive_samples: List, 
                                               fooled_cases: List, batch_size: int = 16,
                                               epochs: int = 3) -> float:
        """
        使用PASE发现的错误案例训练判别器
        """
        print(f"训练判别器 (Ground Truth纠错)...")
        print(f"  正样本: {len(positive_samples)}")
        print(f"  PASE欺骗案例: {len(fooled_cases)}")
        
        if len(fooled_cases) == 0:
            print(f"  没有发现判别器被骗案例，跳过训练")
            return 0.0
        
        self.discriminator.train()
        
        # 准备训练样本
        all_samples = positive_samples.copy()
        
        # 添加被PASE骗的样本，用GT标签纠正
        for fooled_case in fooled_cases:
            all_samples.append((
                fooled_case['question'],
                fooled_case['path'],
                'ground_truth_negative'  # GT强制纠错
            ))
        
        random.shuffle(all_samples)
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        corrections_made = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for i in range(0, len(all_samples), batch_size):
                batch = all_samples[i:i+batch_size]
                
                questions = []
                path_data = []  
                labels = []
                
                for question, path, label in batch:
                    if len(path) < 3:
                        continue
                    
                    questions.append(question)
                    
                    final_entity = path[-1]
                    path_string = '.'.join(path)
                    path_data.append({'paths': {final_entity: [path_string]}})
                    
                    # 标签处理
                    if label == 'positive':
                        label_value = 1.0
                    elif label in ['negative', 'ground_truth_negative']:
                        label_value = 0.0
                        if label == 'ground_truth_negative':
                            corrections_made += 1
                    else:
                        label_value = 0.0
                    
                    labels.append({final_entity: label_value})
                
                if not questions:
                    continue
                
                # 前向传播
                self.discriminator_optimizer.zero_grad()
                outputs = self.discriminator(questions, path_data, epoch=0)
                
                # 计算损失
                batch_loss = 0.0
                batch_correct = 0
                
                for output, label_dict in zip(outputs, labels):
                    individual_scores = output['individual_scores']
                    path_details = output['path_details']
                    
                    for j, detail in enumerate(path_details):
                        entity = detail['answer_entity']
                        if entity in label_dict:
                            pred_score = torch.sigmoid(individual_scores[j])
                            true_label = torch.tensor(label_dict[entity], device=self.device)
                            
                            # 对纠错样本使用更大权重
                            loss_weight = 2.0 if corrections_made > 0 else 1.0
                            loss = nn.BCELoss()(pred_score, true_label) * loss_weight
                            batch_loss += loss
                            
                            # 统计准确率  
                            pred_label = (pred_score > 0.8).float()  # 保持一致的高阈值
                            if pred_label == true_label:
                                batch_correct += 1
                            
                            total_samples += 1
                
                if batch_loss > 0:
                    batch_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
                    self.discriminator_optimizer.step()
                    
                    epoch_loss += batch_loss.item()
                    total_correct += batch_correct
            
            total_loss += epoch_loss
        
        # 统计更新
        self.adversarial_stats['discriminator_corrections'] += corrections_made
        self.adversarial_stats['correction_history'].append(corrections_made)
        
        avg_loss = total_loss / max(1, len(all_samples))
        accuracy = total_correct / max(1, total_samples)
        
        print(f"  判别器训练完成:")
        print(f"    平均损失: {avg_loss:.4f}")
        print(f"    准确率: {accuracy:.3f}")
        print(f"    纠错次数: {corrections_made}")
        
        return avg_loss
    
    def evaluate_adversarial_performance(self, test_data: List[Dict]) -> Dict:
        """评估对抗性能"""
        print("评估PASE vs 判别器对抗性能...")
        
        self.discriminator.eval()
        
        metrics = {
            'total_queries': 0,
            'pase_correct': 0,
            'discriminator_correct': 0,
            'both_correct': 0,
            'both_wrong': 0,
            'pase_right_disc_wrong': 0,
            'pase_wrong_disc_right': 0,
            'agreement_rate': 0.0,
            'pase_accuracy': 0.0,
            'discriminator_accuracy': 0.0
        }
        
        for qa_item in test_data:
            question = qa_item['question']
            start_entity = qa_item['question_entity']  
            answer_entities = set(qa_item['answer_entities'])
            
            try:
                # PASE生成
                path, _, pase_confidence = self.pase_generator.progressive_adaptive_subgraph_expansion(
                    question, start_entity, verbose=False
                )
                
                if len(path) == 0:
                    continue
                
                metrics['total_queries'] += 1
                final_entity = path[-1]
                pase_is_correct = final_entity in answer_entities
                
                # 判别器评估
                path_string = '.'.join(path)
                path_data = [{'paths': {final_entity: [path_string]}}]
                
                with torch.no_grad():
                    disc_output = self.discriminator([question], path_data, epoch=0)
                    disc_raw_score = float(disc_output[0]['individual_scores'][0])
                    disc_confidence = torch.sigmoid(torch.tensor(disc_raw_score)).item()
                
                disc_thinks_correct = disc_confidence > 0.8  # 保持一致的高阈值
                
                # 统计各种情况
                if pase_is_correct:
                    metrics['pase_correct'] += 1
                if disc_thinks_correct == pase_is_correct:  # 判别器判断正确
                    metrics['discriminator_correct'] += 1
                
                if pase_is_correct and disc_thinks_correct:
                    metrics['both_correct'] += 1
                elif not pase_is_correct and not disc_thinks_correct:
                    metrics['both_wrong'] += 1
                elif pase_is_correct and not disc_thinks_correct:
                    metrics['pase_right_disc_wrong'] += 1
                elif not pase_is_correct and disc_thinks_correct:
                    metrics['pase_wrong_disc_right'] += 1
                    
            except Exception as e:
                continue
        
        # 计算最终指标
        total = metrics['total_queries']
        if total > 0:
            metrics['pase_accuracy'] = metrics['pase_correct'] / total
            metrics['discriminator_accuracy'] = metrics['discriminator_correct'] / total
            metrics['agreement_rate'] = (metrics['both_correct'] + metrics['both_wrong']) / total
        
        return metrics
    
    def process_single_query(self, qa_item: Dict) -> Dict:
        """处理单个查询 - 用于并行处理"""
        try:
            query_start = time.time()
            
            question = qa_item['question']
            start_entity = qa_item['question_entity']
            answer_entities = set(qa_item['answer_entities'])
            question_type = qa_item.get('type', 'unknown')
            
            # 确定问题类型
            if '1hop' in question_type:
                hop_category = '1hop'
            elif '2hop' in question_type:
                hop_category = '2hop'
            elif '3hop' in question_type:
                hop_category = '3hop'
            else:
                return {'status': 'skipped'}  # 跳过未知类型
            
            # 生成Top-K路径候选
            top_k_paths = self.generate_top_k_paths_with_discriminator(
                question, start_entity, k=5, confidence_threshold=0.8
            )
            
            query_time = time.time() - query_start
            
            if not top_k_paths:
                return {'status': 'no_paths', 'query_time': query_time, 'hop_category': hop_category}
            
            # 计算指标
            ranked_results = []
            path_confidences = []
            ground_truth_labels = []
            average_hops = []
            subgraph_sizes = []
            
            for path, confidence, hop_count, subgraph_size in top_k_paths:
                final_entity = path[-1] if path else ""
                is_correct = final_entity in answer_entities
                ranked_results.append((confidence, is_correct))
                
                # 记录路径信息
                path_confidences.append(confidence)
                ground_truth_labels.append(1.0 if is_correct else 0.0)
                average_hops.append(hop_count)
                subgraph_sizes.append(subgraph_size)
            
            # 按置信度降序排列
            ranked_results.sort(key=lambda x: x[0], reverse=True)
            
            # 计算各项指标
            hits_at_1 = 1 if ranked_results and ranked_results[0][1] else 0
            hits_at_3 = 1 if any(result[1] for result in ranked_results[:3]) else 0
            
            # 计算MRR
            mrr_score = 0.0
            for i, (conf, is_correct) in enumerate(ranked_results):
                if is_correct:
                    mrr_score = 1.0 / (i + 1)
                    break
            
            return {
                'status': 'success',
                'hop_category': hop_category,
                'query_time': query_time,
                'hits_at_1': hits_at_1,
                'hits_at_3': hits_at_3,
                'mrr_score': mrr_score,
                'path_confidences': path_confidences,
                'ground_truth_labels': ground_truth_labels,
                'average_hops': average_hops,
                'subgraph_sizes': subgraph_sizes
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}

    def run_pase_inference_evaluation(self, test_data: List[Dict], sample_size: int = 3000, 
                                    num_workers: int = 4):
        """运行PASE推理评估 - 并行处理加速"""
        print("开始PASE推理评估")
        print("="*50)
        print(f"并行工作线程数: {num_workers}")
        
        start_time = time.time()
        
        # 限制样本数量
        test_data = test_data[:sample_size]
        
        # 评估指标
        metrics = {
            'total_queries': 0,
            'hits_at_1': 0,
            'hits_at_3': 0,
            'mrr_scores': [],
            'path_confidences': [],
            'ground_truth_labels': [],
            'average_hops': [],
            'subgraph_sizes': [],
            'query_times': [],
            'hop_distributions': {'1hop': 0, '2hop': 0, '3hop': 0},
            'hop_accuracies': {'1hop': {'total': 0, 'correct': 0}, 
                              '2hop': {'total': 0, 'correct': 0},
                              '3hop': {'total': 0, 'correct': 0}}
        }
        
        # 并行处理
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # 提交所有任务
            future_to_idx = {executor.submit(self.process_single_query, qa_item): i 
                           for i, qa_item in enumerate(test_data)}
            
            # 创建进度条
            with tqdm(total=len(test_data), desc="PASE并行推理", 
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
                
                # 处理完成的任务
                for future in as_completed(future_to_idx):
                    try:
                        result = future.result(timeout=30)  # 30秒超时
                        
                        if result['status'] == 'success':
                            hop_category = result['hop_category']
                            
                            # 线程安全地更新指标
                            with self.lock:
                                metrics['total_queries'] += 1
                                metrics['hits_at_1'] += result['hits_at_1']
                                metrics['hits_at_3'] += result['hits_at_3']
                                metrics['mrr_scores'].append(result['mrr_score'])
                                metrics['query_times'].append(result['query_time'])
                                
                                metrics['hop_distributions'][hop_category] += 1
                                metrics['hop_accuracies'][hop_category]['total'] += 1
                                if result['hits_at_1']:
                                    metrics['hop_accuracies'][hop_category]['correct'] += 1
                                
                                # 扩展列表
                                metrics['path_confidences'].extend(result['path_confidences'])
                                metrics['ground_truth_labels'].extend(result['ground_truth_labels'])
                                metrics['average_hops'].extend(result['average_hops'])
                                metrics['subgraph_sizes'].extend(result['subgraph_sizes'])
                        
                        elif result['status'] == 'no_paths':
                            hop_category = result['hop_category']
                            with self.lock:
                                metrics['hop_distributions'][hop_category] += 1
                                metrics['hop_accuracies'][hop_category]['total'] += 1
                                metrics['query_times'].append(result['query_time'])
                    
                    except Exception as e:
                        pass  # 静默跳过错误
                    
                    pbar.update(1)
        
        # 计算最终指标
        total = metrics['total_queries']
        results = {
            'evaluation_config': {
                'model_type': 'PASE Generator Inference (Parallel)',
                'total_samples': len(test_data),
                'valid_queries': total,
                'evaluation_time': time.time() - start_time,
                'parallel_workers': num_workers,
                'timestamp': datetime.now().isoformat()
            },
            'performance_metrics': {
                'hits_at_1': metrics['hits_at_1'] / max(1, total),
                'hits_at_3': metrics['hits_at_3'] / max(1, total),
                'mrr': np.mean(metrics['mrr_scores']) if metrics['mrr_scores'] else 0.0,
                'path_confidence_auc': self.calculate_auc(metrics['path_confidences'], metrics['ground_truth_labels']),
                'average_hop_count': np.mean(metrics['average_hops']) if metrics['average_hops'] else 0.0,
                'average_subgraph_size': np.mean(metrics['subgraph_sizes']) if metrics['subgraph_sizes'] else 0.0,
                'average_query_time': np.mean(metrics['query_times']) if metrics['query_times'] else 0.0
            },
            'hop_specific_metrics': {
                hop: {
                    'accuracy': stats['correct'] / max(1, stats['total']),
                    'total_queries': stats['total'],
                    'correct_queries': stats['correct']
                } for hop, stats in metrics['hop_accuracies'].items()
            },
            'detailed_statistics': {
                'hop_distribution': metrics['hop_distributions'],
                'total_paths_generated': len(metrics['path_confidences']),
                'confidence_statistics': {
                    'mean': np.mean(metrics['path_confidences']) if metrics['path_confidences'] else 0.0,
                    'std': np.std(metrics['path_confidences']) if metrics['path_confidences'] else 0.0,
                    'min': np.min(metrics['path_confidences']) if metrics['path_confidences'] else 0.0,
                    'max': np.max(metrics['path_confidences']) if metrics['path_confidences'] else 0.0
                }
            }
        }
        
        # 保存JSON报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"pase_inference_report_{timestamp}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n评估完成!")
        print(f"报告已保存: {report_file}")
        print(f"处理样本: {processed}")
        print(f"有效查询: {total}")
        print(f"Hits@1: {results['performance_metrics']['hits_at_1']:.3f}")
        print(f"Hits@3: {results['performance_metrics']['hits_at_3']:.3f}")
        print(f"MRR: {results['performance_metrics']['mrr']:.3f}")
        print(f"路径置信度AUC: {results['performance_metrics']['path_confidence_auc']:.3f}")
        
        return results
    
    def generate_top_k_paths_with_discriminator(self, question: str, start_entity: str, 
                                              k: int = 5, confidence_threshold: float = 0.8):
        """生成Top-K路径并用判别器评分"""
        all_candidates = []
        max_attempts = 20  # 最大尝试次数
        
        for attempt in range(max_attempts):
            try:
                # PASE生成路径
                path, hop_count, pase_confidence = self.pase_generator.progressive_adaptive_subgraph_expansion(
                    question, start_entity, verbose=False
                )
                
                if len(path) == 0:
                    continue
                
                # 计算子图大小（估算）
                subgraph_size = len(set(path[::2]))  # 实体数量作为子图规模
                
                # 判别器评分
                final_entity = path[-1]
                path_string = '.'.join(path)
                path_data = [{'paths': {final_entity: [path_string]}}]
                
                with torch.no_grad():
                    disc_output = self.discriminator([question], path_data, epoch=0)
                    disc_raw_score = float(disc_output[0]['individual_scores'][0])
                    disc_confidence = torch.sigmoid(torch.tensor(disc_raw_score)).item()
                
                all_candidates.append((path, disc_confidence, hop_count, subgraph_size))
                
                # 如果已有足够高置信度的路径，或收集到足够的候选
                high_conf_paths = [c for c in all_candidates if c[1] >= confidence_threshold]
                if len(high_conf_paths) >= k or len(all_candidates) >= k * 2:
                    break
                    
            except Exception:
                continue
        
        # 按置信度排序并返回Top-K
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        return all_candidates[:k]
    
    def calculate_auc(self, scores, labels):
        """计算AUC"""
        if not scores or not labels or len(scores) != len(labels):
            return 0.0
        
        try:
            from sklearn.metrics import roc_auc_score
            return roc_auc_score(labels, scores)
        except:
            return 0.0
    
    def _finalize_training(self):
        """完成训练并保存结果"""
        print("\n对抗训练完成!")
        
        # 最终统计
        final_stats = {
            'completion_time': datetime.now().isoformat(),
            'total_epochs': len(self.training_log['epoch_metrics']),
            'total_pase_fooled': self.adversarial_stats['pase_fooled_discriminator'],
            'total_corrections': self.adversarial_stats['discriminator_corrections'],
            'adversarial_effectiveness': (
                self.adversarial_stats['discriminator_corrections'] / 
                max(1, self.adversarial_stats['pase_fooled_discriminator'])
            )
        }
        
        self.training_log['final_summary'] = final_stats
        
        # 保存训练日志
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"pase_adversarial_log_{timestamp}.json"
        
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(self.training_log, f, indent=2, ensure_ascii=False)
            print(f"训练日志已保存: {log_file}")
        except Exception as e:
            print(f"保存日志失败: {e}")
        
        # 保存模型检查点
        try:
            checkpoint_dir = "checkpoints/pase_adversarial"
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            
            checkpoint = {
                'discriminator_state_dict': self.discriminator.state_dict(),
                'optimizer_state_dict': self.discriminator_optimizer.state_dict(),
                'pase_config': {
                    'T_base': self.pase_generator.T_base,
                    'delta': self.pase_generator.delta,
                    'alpha': self.pase_generator.alpha,
                    'max_hops': self.pase_generator.max_hops
                },
                'adversarial_stats': self.adversarial_stats,
                'final_stats': final_stats
            }
            
            torch.save(checkpoint, f"{checkpoint_dir}/pase_adversarial_checkpoint.pth")
            print(f"模型检查点已保存: {checkpoint_dir}/pase_adversarial_checkpoint.pth")
            
        except Exception as e:
            print(f"保存检查点失败: {e}")
        
        # 显示最终报告
        print(f"\n最终对抗统计:")
        print(f"  PASE欺骗判别器总次数: {self.adversarial_stats['pase_fooled_discriminator']}")
        print(f"  判别器纠错总次数: {self.adversarial_stats['discriminator_corrections']}")  
        print(f"  对抗有效性: {final_stats['adversarial_effectiveness']:.3f}")
        print(f"  训练轮次: {final_stats['total_epochs']}")
        print(f"  最终收集的被骗样本: {len(self.global_fooled_cases)}")
        print(f"  判别器置信度阈值: 0.8 (高要求)")
        print(f"  训练策略: 统一收集 -> 最后训练")


def main():
    """主程序 - PASE推理评估"""
    print("PASE推理评估系统")
    print("Progressive Adaptive Subgraph Expansion Inference")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")
    
    try:
        # 1. 数据加载 - 从test数据中采样3000个
        print("\n1. 加载测试数据...")
        qa_file = "query/qa_with_paths_cleaned.json"
        
        all_data = []
        with open(qa_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        query_blocks = content.strip().split('\n\n')
        for block in query_blocks:
            if not block.strip():
                continue
            try:
                query = json.loads(block)
                all_data.append(query)
            except json.JSONDecodeError:
                continue
        
        # 按类型分组test数据
        test_hop_data = {'1hop': [], '2hop': [], '3hop': []}
        for item in all_data:
            type_field = item.get('type', '')
            if 'test' in type_field:  # 只要test数据
                if '1hop' in type_field:
                    test_hop_data['1hop'].append(item)
                elif '2hop' in type_field:
                    test_hop_data['2hop'].append(item)
                elif '3hop' in type_field:
                    test_hop_data['3hop'].append(item)
        
        # 从各类型随机采样200个，总共600个
        test_data = []
        for hop_type, data in test_hop_data.items():
            if data:
                sample_size = min(200, len(data))
                samples = random.sample(data, sample_size)
                test_data.extend(samples)
                print(f"  {hop_type} test数据: {sample_size}个")
        
        print(f"  测试数据总计: {len(test_data)}个")
        
        # 2. 模型初始化
        print("\n2. 初始化模型...")
        
        # 判别器
        discriminator = EnhancedPathRankerDiscriminator(
            entity_embedding_path="embeddings/entity_embeddings.pt",
            freeze_sbert=True,
            use_pattern_memory=False
        )
        
        # 加载预训练权重
        try:
            checkpoint_path = "checkpoints/enhanced_pathranker/best_hits1_model.pth"
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                model_weights = checkpoint['model_state_dict']
            else:
                model_weights = checkpoint
            discriminator.load_state_dict(model_weights, strict=False)
            print("  预训练判别器权重加载成功")
        except Exception as e:
            print(f"  判别器权重加载失败: {e}")
        
        discriminator.to(device)
        discriminator.eval()  # 评估模式
        
        # PASE生成器
        pase_generator = create_pase_generator(
            entity_embedding_path="embeddings/entity_embeddings.pt",
            knowledge_graph_path="graph/knowledge_graph.pkl",
            discriminator=discriminator,
            T_base=0.8,
            delta=0.1, 
            alpha=0.7,
            max_hops=4
        )
        
        print("  PASE生成器创建成功")
        
        # 3. 推理评估器
        print("\n3. 创建推理评估器...")
        trainer = PASEAdversarialTrainer(pase_generator, discriminator, device)
        
        # 4. 开始推理评估
        print("\n4. 开始PASE推理评估...")
        
        # 根据CPU核心数决定并行线程数
        import os
        cpu_count = os.cpu_count() or 4
        num_workers = min(8, max(2, cpu_count // 2))  # 2-8线程之间
        print(f"  检测到{cpu_count}个CPU核心，使用{num_workers}个工作线程")
        
        results = trainer.run_pase_inference_evaluation(test_data, sample_size=600, 
                                                       num_workers=num_workers)
        
    except Exception as e:
        print(f"评估失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()