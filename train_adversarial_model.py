"""
对抗学习知识图谱路径推理 - 正式训练脚本

使用修复版GAN-RL训练器，包含完整指标监控和JSON日志记录

运行命令：
python train_adversarial_model.py

训练指标：
一、判别器 (Ranker) 指标：Loss_D, Acc_Real, Acc_Fake, F1-Score
二、生成器 (Discoverer) 指标：Loss_G, Avg_Reward, Path_Length_Avg, Path_Diversity 
三、端到端任务指标：Hits@1, MRR, Success_Rate
"""

import sys
import torch
import pickle
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Set

# 添加models路径
sys.path.append(str(Path(__file__).parent / "models" / "path_discover"))
sys.path.append(str(Path(__file__).parent / "models" / "path_ranker"))

from models.path_discover.beam_search_generator import BeamSearchPathGenerator
from enhanced_path_ranker import EnhancedPathRankerDiscriminator
from gan_rl_trainer_fixed import GANRLTrainerFixed

class AdversarialTrainingPipeline:
    """对抗学习训练流水线"""
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        self.generator = None
        self.discriminator = None
        self.trainer = None
        
        # 数据
        self.knowledge_graph = None
        self.train_data = {}  # 按hop类型组织的训练数据
        
        # 训练日志
        self.training_log = {
            'training_config': {},
            'epoch_metrics': [],
            'final_summary': {}
        }
        
        # 最佳模型跟踪
        self.best_metrics = {
            'best_hits_at_1': 0.0,
            'best_mrr': 0.0,
            'best_success_rate': 0.0,
            'best_epoch': 0
        }
    
    def load_dataset(self):
        """加载数据集"""
        print("Loading dataset...")
        
        # 1. 加载QA数据
        qa_file = "query/qa_with_paths_cleaned.json"
        if not Path(qa_file).exists():
            print(f"ERROR: QA dataset not found: {qa_file}")
            return False
        
        all_data = self._load_multiline_json(qa_file)
        if not all_data:
            return False
        
        # 2. 按hop分组训练数据
        self.train_data = {'1hop': [], '2hop': [], '3hop': []}
        
        for item in all_data:
            type_field = item.get('type', '')
            if 'train' in type_field:
                if '1hop' in type_field:
                    self.train_data['1hop'].append(item)
                elif '2hop' in type_field:
                    self.train_data['2hop'].append(item)
                elif '3hop' in type_field:
                    self.train_data['3hop'].append(item)
        
        total_train = sum(len(data) for data in self.train_data.values())
        print(f"Training data loaded:")
        for hop_type, data in self.train_data.items():
            print(f"  {hop_type}: {len(data)} samples")
        print(f"  Total: {total_train} samples")
        
        # 3. 加载知识图谱
        kg_file = "graph/knowledge_graph.pkl"
        if not Path(kg_file).exists():
            print(f"ERROR: Knowledge graph not found: {kg_file}")
            return False
        
        with open(kg_file, 'rb') as f:
            self.knowledge_graph = pickle.load(f)
        
        print(f"Knowledge graph: {self.knowledge_graph.number_of_nodes()} nodes, {self.knowledge_graph.number_of_edges()} edges")
        
        return True
    
    def _load_multiline_json(self, file_path: str) -> List[Dict]:
        """加载多行JSON格式的文件"""
        data = []
        current_obj = ""
        brace_count = 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    current_obj += line + "\n"
                    brace_count += line.count('{') - line.count('}')
                    
                    if brace_count == 0 and current_obj.strip():
                        try:
                            obj = json.loads(current_obj.strip())
                            data.append(obj)
                            current_obj = ""
                        except json.JSONDecodeError:
                            current_obj = ""
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return []
        
        return data
    
    def initialize_models(self):
        """初始化模型"""
        print("Initializing models...")
        
        if not Path("embeddings/entity_embeddings.pt").exists():
            print("ERROR: Entity embeddings not found. Run: python create_embeddings.py")
            return False
        
        try:
            # 生成器 - 新的Beam Search生成器
            self.generator = BeamSearchPathGenerator(
                entity_embedding_path="embeddings/entity_embeddings.pt",
                max_path_length=6,
                beam_width=5
            )
            self.generator.load_knowledge_graph("graph/knowledge_graph.pkl")
            
            # 判别器
            self.discriminator = EnhancedPathRankerDiscriminator(
                freeze_sbert=True,
                use_pattern_memory=False
            )
            
            # 加载预训练权重
            try:
                checkpoint_path = "checkpoints/enhanced_pathranker/best_hits1_model.pth"
                print(f"Loading discriminator weights from: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                
                # 提取model_state_dict
                if 'model_state_dict' in checkpoint:
                    model_weights = checkpoint['model_state_dict']
                else:
                    model_weights = checkpoint
                
                self.discriminator.load_state_dict(model_weights, strict=False)
                print("Pre-trained discriminator weights loaded successfully")
            except Exception as e:
                print(f"Warning: Failed to load discriminator weights: {e}")
                print("Using randomly initialized discriminator")
            
            # 训练器 - 设置判别器阈值为0.8
            self.trainer = GANRLTrainerFixed(
                self.generator, 
                self.discriminator, 
                device=self.device,
                discriminator_threshold=0.8  # 设置阈值为0.8
            )
            
            self.generator.to(self.device)
            self.discriminator.to(self.device)
            
            print("Models initialized successfully")
            return True
            
        except Exception as e:
            print(f"Model initialization failed: {e}")
            return False
    
    def train_with_metrics(self, epochs: int = 15):
        """执行带完整指标监控的对抗训练"""
        print(f"Starting adversarial training ({epochs} epochs)")
        print("="*80)
        
        # 训练配置
        self.training_log['training_config'] = {
            'epochs': epochs,
            'device': str(self.device),
            'start_time': datetime.now().isoformat(),
            'data_distribution': {hop: len(data) for hop, data in self.train_data.items()},
            'model_types': {
                'generator': type(self.generator).__name__,
                'discriminator': type(self.discriminator).__name__,
                'trainer': type(self.trainer).__name__
            }
        }
        
        for epoch in range(epochs):
            epoch_start = time.time()
            print(f"\nEPOCH {epoch + 1}/{epochs}")
            print("-" * 60)
            
            try:
                # 随机选择hop类型训练数据
                epoch_plan = self._create_epoch_plan(epoch)
                if not epoch_plan:
                    continue
                
                # 合并本epoch的训练数据
                epoch_data = []
                for hop_type, samples in epoch_plan.items():
                    epoch_data.extend(samples)
                
                # 执行对抗训练（带完整指标计算）
                epoch_metrics = self.trainer.train_epoch_with_adversarial_correction(
                    epoch_data, current_epoch=epoch
                )
                
                # 添加时间和epoch信息
                epoch_metrics.update({
                    'epoch': epoch + 1,
                    'training_time': time.time() - epoch_start,
                    'timestamp': datetime.now().isoformat(),
                    'hop_distribution': {hop: len(samples) for hop, samples in epoch_plan.items()},
                    'total_samples': len(epoch_data)
                })
                
                # 保存epoch指标到日志
                self.training_log['epoch_metrics'].append(epoch_metrics)
                
                # 每个epoch保存日志
                self._save_training_log()
                
                # 检查是否需要保存最佳模型
                self._check_and_save_best_model(epoch_metrics, epoch)
                
            except Exception as e:
                print(f"Epoch {epoch + 1} failed: {e}")
                import traceback
                traceback.print_exc()
        
        # 完成训练
        self._finalize_training()
    
    def _create_epoch_plan(self, epoch: int) -> Dict[str, List[Dict]]:
        """创建epoch训练计划"""
        hop_types = ['1hop', '2hop', '3hop']
        random.shuffle(hop_types)
        
        epoch_plan = {}
        for hop_type in hop_types:
            available_data = self.train_data[hop_type]
            if not available_data:
                continue
            
            # 动态采样 - 按hop类型分配不同数量
            if hop_type == '1hop' and len(available_data) > 1000:
                sample_count = 1000  # 1hop: 1000个
                selected_samples = random.sample(available_data, sample_count)
            elif hop_type == '2hop' and len(available_data) > 2000:
                sample_count = 2000  # 2hop: 2000个  
                selected_samples = random.sample(available_data, sample_count)
            elif hop_type == '3hop' and len(available_data) > 3000:
                sample_count = 3000  # 3hop: 3000个
                selected_samples = random.sample(available_data, sample_count)
            else:
                selected_samples = available_data.copy()
                random.shuffle(selected_samples)
            
            epoch_plan[hop_type] = selected_samples
        
        return epoch_plan
    
    def _check_and_save_best_model(self, epoch_metrics, epoch):
        """检查并保存最佳模型"""
        current_hits_at_1 = epoch_metrics.get('Hits_at_1', 0.0)
        current_mrr = epoch_metrics.get('MRR', 0.0)
        current_success_rate = epoch_metrics.get('Success_Rate', 0.0)
        
        # 综合指标：加权平均
        current_score = (current_hits_at_1 * 0.4 + current_mrr * 0.4 + current_success_rate * 0.2)
        best_score = (self.best_metrics['best_hits_at_1'] * 0.4 + 
                     self.best_metrics['best_mrr'] * 0.4 + 
                     self.best_metrics['best_success_rate'] * 0.2)
        
        if current_score > best_score:
            print(f"\n🏆 NEW BEST MODEL! Score: {current_score:.4f} (prev: {best_score:.4f})")
            
            # 更新最佳指标
            self.best_metrics.update({
                'best_hits_at_1': current_hits_at_1,
                'best_mrr': current_mrr,
                'best_success_rate': current_success_rate,
                'best_epoch': epoch + 1,
                'best_score': current_score
            })
            
            # 保存最佳模型到adversarial目录
            try:
                from advanced_inference_system import save_production_models
                best_model_dir = save_production_models(
                    self.trainer, 
                    f"checkpoints/adversarial/best_epoch_{epoch+1}"
                )
                print(f"💎 Best model saved: {best_model_dir}")
                
                # 也更新主adversarial目录
                production_dir = save_production_models(
                    self.trainer, 
                    "checkpoints/adversarial"
                )
                print(f"🔄 Adversarial models updated: {production_dir}")
                
            except Exception as e:
                print(f"❌ Failed to save best model: {e}")
    
    def _save_training_log(self):
        """保存训练日志"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"training_log_{timestamp}.json"
        
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(self.training_log, f, indent=2, ensure_ascii=False)
            print(f"Training log saved: {log_file}")
        except Exception as e:
            print(f"Failed to save training log: {e}")
    
    def _finalize_training(self):
        """完成训练并保存最终模型"""
        print("\nFinalizing training...")
        
        # 最终总结
        if self.training_log['epoch_metrics']:
            final_metrics = self.training_log['epoch_metrics'][-1]
            
            # 计算趋势
            first_metrics = self.training_log['epoch_metrics'][0]
            improvements = {
                'Loss_D_change': final_metrics.get('Loss_D', 0) - first_metrics.get('Loss_D', 0),
                'Avg_Reward_change': final_metrics.get('Avg_Reward', 0) - first_metrics.get('Avg_Reward', 0),
                'Hits_at_1_change': final_metrics.get('Hits_at_1', 0) - first_metrics.get('Hits_at_1', 0),
                'MRR_change': final_metrics.get('MRR', 0) - first_metrics.get('MRR', 0)
            }
            
            self.training_log['final_summary'] = {
                'completion_time': datetime.now().isoformat(),
                'total_epochs': len(self.training_log['epoch_metrics']),
                'final_metrics': final_metrics,
                'improvements': improvements
            }
        
        # 保存最终日志
        self._save_training_log()
        
        # 保存最终对抗模型
        try:
            from advanced_inference_system import save_production_models
            production_dir = save_production_models(
                self.trainer, 
                "checkpoints/adversarial"
            )
            print(f"Final adversarial models saved: {production_dir}")
        except Exception as e:
            print(f"Failed to save adversarial models: {e}")
        
        # 显示最终报告
        self._display_final_report()
    
    def _display_final_report(self):
        """显示最终训练报告"""
        print("\n" + "="*80)
        print("ADVERSARIAL TRAINING COMPLETED!")
        print("="*80)
        
        if not self.training_log['epoch_metrics']:
            print("No training metrics available.")
            return
        
        final_metrics = self.training_log['epoch_metrics'][-1]
        improvements = self.training_log['final_summary'].get('improvements', {})
        
        print("\nFINAL PERFORMANCE METRICS:")
        print(f"  Loss_D:           {final_metrics.get('Loss_D', 0):.4f}")
        print(f"  Acc_Real:         {final_metrics.get('Acc_Real', 0):.3f}")
        print(f"  Acc_Fake:         {final_metrics.get('Acc_Fake', 0):.3f}")
        print(f"  F1-Score:         {final_metrics.get('F1_Score', 0):.3f}")
        print(f"  Avg_Reward:       {final_metrics.get('Avg_Reward', 0):.4f}")
        print(f"  Path_Length_Avg:  {final_metrics.get('Path_Length_Avg', 0):.2f}")
        print(f"  Path_Diversity:   {final_metrics.get('Path_Diversity', 0):.3f}")
        print(f"  Hits@1:           {final_metrics.get('Hits_at_1', 0):.3f}")
        print(f"  MRR:              {final_metrics.get('MRR', 0):.3f}")
        print(f"  Success_Rate:     {final_metrics.get('Success_Rate', 0):.3f}")
        
        print("\nTRAINING IMPROVEMENTS:")
        for metric, change in improvements.items():
            direction = "↑" if change > 0 else "↓" if change < 0 else "→"
            print(f"  {metric}: {change:+.4f} {direction}")
        
        print("\nNEXT STEPS:")
        print("  1. Test inference: python demo_advanced_inference.py")
        print("  2. View training logs in current directory")
        print("  3. Deploy models from checkpoints/production_models/")
        print("="*80)

def main():
    """主训练程序"""
    print("ADVERSARIAL KNOWLEDGE GRAPH PATH REASONING")
    print("Training Pipeline with Comprehensive Metrics")
    print("="*80)
    
    # 创建训练流水线
    pipeline = AdversarialTrainingPipeline()
    
    # 加载数据
    if not pipeline.load_dataset():
        print("ERROR: Dataset loading failed")
        return
    
    # 初始化模型
    if not pipeline.initialize_models():
        print("ERROR: Model initialization failed")
        return
    
    # 开始训练
    print("\nStarting adversarial training with:")
    print("* Fixed GAN-RL trainer (handles discriminator fooling)")
    print("* Ground Truth correction mechanism")
    print("* Intelligent reward shaping")
    print("* Comprehensive metrics monitoring")
    print("* JSON logging for all epochs")
    
    pipeline.train_with_metrics(epochs=5)

if __name__ == "__main__":
    main()