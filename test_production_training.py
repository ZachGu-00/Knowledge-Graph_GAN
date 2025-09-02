"""
正式对抗训练脚本

数据集：
- query/qa_with_paths_cleaned.json (完整数据集)
- graph/knowledge_graph.pkl (知识图谱)
- graph/entity_names.json (实体名称)

训练策略：
- 只使用 "xhop_train" 数据
- 每个epoch随机选择不同hop进行训练
- 使用修复版GANRLTrainerFixed处理"双方都犯错"情况
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

from differentiable_path_generator_truly_fixed import DifferentiablePathGeneratorTrulyFixed
from enhanced_path_ranker import EnhancedPathRankerDiscriminator
from gan_rl_trainer_fixed import GANRLTrainerFixed

class ProductionTrainingPipeline:
    """生产环境训练流水线"""
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🖥️  Using device: {self.device}")
        
        self.generator = None
        self.discriminator = None
        self.trainer = None
        
        # 数据
        self.knowledge_graph = None
        self.entity_names = None
        self.train_data = {}  # 按hop类型组织的训练数据
        
        # 训练统计
        self.training_history = {
            'epochs': [],
            'hop_distribution': [],
            'adversarial_stats': []
        }
    
    def load_complete_dataset(self):
        """加载完整数据集"""
        print("📊 Loading complete dataset...")
        
        # 1. 加载QA数据
        qa_file = "query/qa_with_paths_cleaned.json"
        if not Path(qa_file).exists():
            print(f"❌ QA dataset not found: {qa_file}")
            return False
        
        print(f"📖 Loading QA dataset from {qa_file}")
        all_data = self._load_multiline_json(qa_file)
        
        if not all_data:
            print(f"❌ Failed to load QA dataset")
            return False
        
        print(f"✅ Loaded {len(all_data)} total QA samples")
        
        # 2. 筛选训练数据并按hop分组
        self.train_data = {
            '1hop': [],
            '2hop': [], 
            '3hop': []
        }
        
        for item in all_data:
            type_field = item.get('type', '')
            if 'train' in type_field:
                if '1hop' in type_field:
                    self.train_data['1hop'].append(item)
                elif '2hop' in type_field:
                    self.train_data['2hop'].append(item)
                elif '3hop' in type_field:
                    self.train_data['3hop'].append(item)
        
        print(f"📈 Training data distribution:")
        for hop_type, data in self.train_data.items():
            print(f"   {hop_type}: {len(data)} samples")
        
        total_train = sum(len(data) for data in self.train_data.values())
        print(f"   Total training samples: {total_train}")
        
        if total_train == 0:
            print(f"❌ No training data found!")
            return False
        
        # 3. 加载知识图谱
        kg_file = "graph/knowledge_graph.pkl"
        if not Path(kg_file).exists():
            print(f"❌ Knowledge graph not found: {kg_file}")
            return False
        
        print(f"🕸️  Loading knowledge graph from {kg_file}")
        with open(kg_file, 'rb') as f:
            self.knowledge_graph = pickle.load(f)
        
        print(f"✅ Knowledge graph loaded: {self.knowledge_graph.number_of_nodes()} nodes, {self.knowledge_graph.number_of_edges()} edges")
        
        # 4. 加载实体名称（可选）
        entity_file = "graph/entity_names.json"
        if Path(entity_file).exists():
            print(f"📝 Loading entity names from {entity_file}")
            with open(entity_file, 'r', encoding='utf-8') as f:
                self.entity_names = json.load(f)
            print(f"✅ Loaded {len(self.entity_names)} entity names")
        
        return True
    
    def _load_multiline_json(self, file_path: str) -> List[Dict]:
        """加载多行JSON格式的文件"""
        data = []
        current_obj = ""
        brace_count = 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    current_obj += line + "\\n"
                    
                    # 计算大括号
                    brace_count += line.count('{') - line.count('}')
                    
                    # 当大括号平衡时，说明一个JSON对象结束
                    if brace_count == 0 and current_obj.strip():
                        try:
                            obj = json.loads(current_obj.strip())
                            data.append(obj)
                            current_obj = ""
                        except json.JSONDecodeError as e:
                            print(f"JSON parse error at line {line_num}: {e}")
                            print(f"Object: {current_obj[:100]}...")
                            current_obj = ""
                            
        except Exception as e:
            print(f"File read error: {e}")
            return []
        
        return data
    
    def initialize_models(self):
        """初始化模型"""
        print("\\n🤖 Initializing models...")
        
        # 检查必要文件
        if not Path("embeddings/entity_embeddings.pt").exists():
            print("❌ Entity embeddings not found. Run: python create_embeddings.py")
            return False
        
        try:
            # 1. 初始化生成器 (Discoverer)
            print("⚡ Initializing Discoverer (Generator)...")
            self.generator = DifferentiablePathGeneratorTrulyFixed(
                entity_embedding_path="embeddings/entity_embeddings.pt",
                max_path_length=6,
                beam_width=5
            )
            self.generator.knowledge_graph = self.knowledge_graph  # 设置KG
            print("✅ Discoverer initialized")
            
            # 2. 初始化判别器 (Ranker)
            print("🧠 Initializing Ranker (Discriminator)...")
            self.discriminator = EnhancedPathRankerDiscriminator(
                freeze_sbert=True,
                disable_pattern_memory=True
            )
            print("✅ Ranker initialized")
            
            # 3. 初始化修复版训练器
            print("🔧 Initializing Fixed GAN-RL Trainer...")
            self.trainer = GANRLTrainerFixed(
                self.generator, 
                self.discriminator, 
                device=self.device
            )
            print("✅ Fixed GAN-RL Trainer initialized")
            
            # 移动到设备
            self.generator.to(self.device)
            self.discriminator.to(self.device)
            
            print(f"🎯 All models ready on {self.device}")
            return True
            
        except Exception as e:
            print(f"❌ Model initialization failed: {e}")
            return False
    
    def create_epoch_training_plan(self, epoch: int) -> Dict[str, List[Dict]]:
        """
        创建本轮训练计划：随机选择不同hop类型进行训练
        
        策略：
        - 每个epoch随机选择hop类型和样本
        - 确保各hop类型都有训练机会
        - 动态平衡不同复杂度的训练
        """
        print(f"\\n📋 Planning training for Epoch {epoch + 1}...")
        
        # 随机选择hop类型顺序
        hop_types = ['1hop', '2hop', '3hop']
        random.shuffle(hop_types)
        
        epoch_plan = {}
        
        for hop_type in hop_types:
            available_data = self.train_data[hop_type]
            if not available_data:
                continue
            
            # 根据数据量动态选择样本数
            if len(available_data) > 100:
                # 大数据量：随机采样
                sample_count = min(50, len(available_data) // 2)
                selected_samples = random.sample(available_data, sample_count)
            else:
                # 小数据量：使用全部
                selected_samples = available_data.copy()
                random.shuffle(selected_samples)
            
            epoch_plan[hop_type] = selected_samples
            print(f"   {hop_type}: {len(selected_samples)} samples")
        
        total_samples = sum(len(samples) for samples in epoch_plan.values())
        print(f"   📊 Total epoch samples: {total_samples}")
        
        # 记录hop分布
        hop_distribution = {hop: len(samples) for hop, samples in epoch_plan.items()}
        self.training_history['hop_distribution'].append(hop_distribution)
        
        return epoch_plan
    
    def train_with_hop_curriculum(self, epochs: int = 10):
        """
        按hop课程进行对抗训练
        
        每个epoch：
        1. 随机选择hop类型和样本
        2. 使用修复版训练器处理"双方都犯错"
        3. 记录对抗学习统计信息
        """
        print(f"\\n🚀 Starting Production Adversarial Training ({epochs} epochs)")
        print("="*80)
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            print(f"\\n🎯 EPOCH {epoch + 1}/{epochs}")
            print("-" * 60)
            
            try:
                # 1. 创建本轮训练计划
                epoch_plan = self.create_epoch_training_plan(epoch)
                
                if not epoch_plan:
                    print("⚠️  No training data for this epoch, skipping...")
                    continue
                
                # 2. 按hop类型依次训练
                epoch_stats = {
                    'epoch': epoch + 1,
                    'hop_results': {},
                    'overall_discriminator_loss': 0.0,
                    'overall_generator_reward': 0.0,
                    'timestamp': datetime.now().isoformat()
                }
                
                total_disc_loss = 0.0
                total_gen_reward = 0.0
                hop_count = 0
                
                for hop_type, samples in epoch_plan.items():
                    if not samples:
                        continue
                        
                    print(f"\\n🔍 Training on {hop_type} data ({len(samples)} samples)...")
                    
                    # 使用修复版训练器
                    hop_result = self.trainer.train_epoch_with_adversarial_correction(
                        samples, current_epoch=epoch
                    )
                    
                    epoch_stats['hop_results'][hop_type] = hop_result
                    total_disc_loss += hop_result.get('discriminator_loss', 0.0)
                    total_gen_reward += hop_result.get('generator_reward', 0.0)
                    hop_count += 1
                    
                    print(f"✅ {hop_type} training completed:")
                    print(f"   Discriminator Loss: {hop_result.get('discriminator_loss', 0):.4f}")
                    print(f"   Generator Reward: {hop_result.get('generator_reward', 0):.4f}")
                    print(f"   Discriminator Fooled: {hop_result.get('discriminator_fooled', 0)} cases")
                    print(f"   GT Corrections: {hop_result.get('ground_truth_corrections', 0)} cases")
                
                # 3. 汇总epoch统计
                epoch_stats['overall_discriminator_loss'] = total_disc_loss / max(1, hop_count)
                epoch_stats['overall_generator_reward'] = total_gen_reward / max(1, hop_count)
                
                # 4. 获取对抗学习统计
                adversarial_stats = self.trainer.get_adversarial_stats()
                epoch_stats['adversarial_stats'] = adversarial_stats
                self.training_history['adversarial_stats'].append(adversarial_stats)
                
                epoch_time = time.time() - epoch_start_time
                epoch_stats['training_time'] = epoch_time
                
                print(f"\\n📊 EPOCH {epoch + 1} SUMMARY:")
                print(f"   Overall Discriminator Loss: {epoch_stats['overall_discriminator_loss']:.4f}")
                print(f"   Overall Generator Reward: {epoch_stats['overall_generator_reward']:.4f}")
                print(f"   🎭 Discriminator Fooled Total: {adversarial_stats['discriminator_fooled_total']}")
                print(f"   🎯 GT Corrections Applied: {adversarial_stats['discriminator_corrected_total']}")
                print(f"   📈 Correction Effectiveness: {adversarial_stats['correction_effectiveness']:.1%}")
                print(f"   ⏱️  Training Time: {epoch_time:.2f}s")
                
                self.training_history['epochs'].append(epoch_stats)
                
                # 5. 定期保存检查点
                if (epoch + 1) % 5 == 0:
                    self.save_training_checkpoint(epoch + 1)
                
            except Exception as e:
                print(f"❌ Epoch {epoch + 1} failed: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\\n🎊 Production Training Completed!")
        self.save_final_models()
        self.generate_training_report()
    
    def save_training_checkpoint(self, epoch: int):
        """保存训练检查点"""
        try:
            checkpoint_dir = Path("checkpoints")
            checkpoint_dir.mkdir(exist_ok=True)
            
            checkpoint_path = checkpoint_dir / f"production_training_epoch_{epoch}.pt"
            self.trainer.save_checkpoint(str(checkpoint_path), epoch)
            
            # 保存训练历史
            history_path = checkpoint_dir / f"training_history_epoch_{epoch}.json"
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(self.training_history, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Checkpoint saved: {checkpoint_path}")
            
        except Exception as e:
            print(f"❌ Checkpoint save failed: {e}")
    
    def save_final_models(self):
        """保存最终生产模型"""
        try:
            from advanced_inference_system import save_production_models
            
            # 保存生产环境模型
            production_dir = save_production_models(
                self.trainer, 
                "checkpoints/production_models_final"
            )
            
            # 保存完整训练历史
            history_file = Path(production_dir) / "complete_training_history.json"
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(self.training_history, f, indent=2, ensure_ascii=False)
            
            print(f"🎉 Final models saved to: {production_dir}")
            
        except Exception as e:
            print(f"❌ Final model save failed: {e}")
    
    def generate_training_report(self):
        """生成训练报告"""
        print("\\n📊 FINAL TRAINING REPORT")
        print("="*80)
        
        if not self.training_history['epochs']:
            print("❌ No training history available")
            return
        
        # 整体统计
        total_epochs = len(self.training_history['epochs'])
        final_stats = self.training_history['adversarial_stats'][-1] if self.training_history['adversarial_stats'] else {}
        
        print(f"🎯 Training Overview:")
        print(f"   Total Epochs: {total_epochs}")
        print(f"   Total Training Data:")
        for hop_type, data in self.train_data.items():
            print(f"     {hop_type}: {len(data)} samples")
        
        # 对抗学习效果
        print(f"\\n🎭 Adversarial Learning Results:")
        print(f"   Discriminator Fooled Cases: {final_stats.get('discriminator_fooled_total', 0)}")
        print(f"   Ground Truth Corrections: {final_stats.get('discriminator_corrected_total', 0)}")
        print(f"   Generator Misled Cases: {final_stats.get('generator_misled_total', 0)}")
        print(f"   Correction Effectiveness: {final_stats.get('correction_effectiveness', 0):.1%}")
        
        # hop分布统计
        if self.training_history['hop_distribution']:
            print(f"\\n📈 Hop Distribution Across Epochs:")
            hop_totals = {'1hop': 0, '2hop': 0, '3hop': 0}
            for epoch_dist in self.training_history['hop_distribution']:
                for hop_type, count in epoch_dist.items():
                    hop_totals[hop_type] += count
            
            for hop_type, total in hop_totals.items():
                print(f"   {hop_type}: {total} total samples trained")
        
        # 性能趋势
        if len(self.training_history['epochs']) > 1:
            first_epoch = self.training_history['epochs'][0]
            last_epoch = self.training_history['epochs'][-1]
            
            disc_improvement = last_epoch['overall_discriminator_loss'] - first_epoch['overall_discriminator_loss']
            gen_improvement = last_epoch['overall_generator_reward'] - first_epoch['overall_generator_reward']
            
            print(f"\\n📊 Performance Trends:")
            print(f"   Discriminator Loss Change: {disc_improvement:+.4f}")
            print(f"   Generator Reward Change: {gen_improvement:+.4f}")
        
        print(f"\\n🚀 Models ready for production use!")
        print(f"   Use: python demo_advanced_inference.py")

def main():
    """主训练程序"""
    print("🎯 Production Adversarial Training Pipeline")
    print("="*80)
    
    # 初始化训练流水线
    pipeline = ProductionTrainingPipeline()
    
    # 1. 加载完整数据集
    if not pipeline.load_complete_dataset():
        print("❌ Dataset loading failed. Exiting.")
        return
    
    # 2. 初始化模型
    if not pipeline.initialize_models():
        print("❌ Model initialization failed. Exiting.")
        return
    
    # 3. 开始对抗训练
    print("\\n🚀 Starting adversarial training...")
    print("💡 This will use the FIXED GAN-RL trainer that properly handles:")
    print("   ✅ Discriminator fooling detection")
    print("   ✅ Ground Truth correction mechanism") 
    print("   ✅ Intelligent reward shaping")
    print("   ✅ True adversarial learning spiral")
    
    # 训练参数
    epochs = 15  # 更多epoch以充分对抗训练
    
    pipeline.train_with_hop_curriculum(epochs=epochs)
    
    print("\\n🎊 Production training completed successfully!")
    print("\\nNext steps:")
    print("1. 🧪 Test the trained models: python demo_advanced_inference.py")
    print("2. 🔍 Analyze training history in checkpoints/")
    print("3. 🚀 Deploy the production models")

if __name__ == "__main__":
    main()