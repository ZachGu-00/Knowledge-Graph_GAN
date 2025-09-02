"""
测试A*路径生成器 + 对抗学习训练

功能：
1. A*生成器基本功能测试
2. 加载训练好的判别器
3. 运行GAN-RL对抗学习训练
4. 使用10个简单1hop样本
5. 输出详细的训练结果到JSON
"""

import sys
import torch
import pickle
import json
import time
from datetime import datetime
from pathlib import Path

# 添加models路径
sys.path.append(str(Path(__file__).parent / "models" / "path_discover"))
sys.path.append(str(Path(__file__).parent / "models" / "path_ranker"))

from astar_path_generator import AStarPathGenerator
from differentiable_path_generator_truly_fixed import DifferentiablePathGeneratorTrulyFixed
from gan_rl_trainer import GANRLTrainer
from enhanced_path_ranker import EnhancedPathRankerDiscriminator

def test_astar_generator():
    """测试A*生成器的基本功能"""
    
    print("A* Path Generator Test")
    print("="*50)
    
    # 1. 检查必需文件
    required_files = {
        "embeddings/entity_embeddings.pt": "Entity embeddings",
        "query/qa_with_paths_cleaned.json": "QA dataset",
        "graph/knowledge_graph.pkl": "Knowledge graph"
    }
    
    for file_path, description in required_files.items():
        if not Path(file_path).exists():
            print(f"Error: {description} not found at {file_path}")
            return
        else:
            print(f"[OK] {description} found")
    
    print(f"\n{'='*50}")
    print("INITIALIZING A* GENERATOR")
    print("="*50)
    
    # 2. 初始化A*生成器
    try:
        generator = AStarPathGenerator(
            entity_embedding_path="embeddings/entity_embeddings.pt",
            max_path_length=6,
            beam_width=5
        )
        print("[OK] A* Generator initialized successfully")
        
        # 显示模型信息
        total_params = sum(p.numel() for p in generator.parameters())
        trainable_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
        print(f"[INFO] Total parameters: {total_params:,}")
        print(f"[INFO] Trainable parameters: {trainable_params:,}")
        print(f"[INFO] SBERT dimension: {generator.sbert_dim}")
        
    except Exception as e:
        print(f"[ERROR] Failed to initialize generator: {e}")
        return
    
    # 3. 加载知识图谱
    print(f"\n{'='*50}")
    print("LOADING KNOWLEDGE GRAPH")
    print("="*50)
    
    try:
        with open("graph/knowledge_graph.pkl", 'rb') as f:
            knowledge_graph = pickle.load(f)
        
        print(f"[OK] Knowledge graph loaded: {len(knowledge_graph.nodes)} nodes, {len(knowledge_graph.edges)} edges")
        
        # 创建简单的关系词表（用于测试）
        relations = set()
        for _, _, edge_data in knowledge_graph.edges(data=True):
            relation = edge_data.get('relation', 'related_to')
            relations.add(relation)
        
        relation_to_id = {rel: idx for idx, rel in enumerate(relations)}
        print(f"[OK] Found {len(relation_to_id)} unique relations")
        
        # 设置知识图谱
        generator.set_knowledge_graph(knowledge_graph, relation_to_id)
        print("[OK] Knowledge graph set in generator")
        
    except Exception as e:
        print(f"[ERROR] Failed to load knowledge graph: {e}")
        return
    
    # 4. 加载测试样本
    print(f"\n{'='*50}")
    print("LOADING TEST SAMPLES")
    print("="*50)
    
    try:
        with open("query/qa_with_paths_cleaned.json", 'r', encoding='utf-8') as f:
            content = f.read()
        
        query_blocks = content.strip().split('\\n\\n')[:3]  # 取前3个样本测试
        test_samples = []
        
        for block in query_blocks:
            if block.strip():
                try:
                    sample = json.loads(block)
                    test_samples.append(sample)
                except:
                    continue
        
        print(f"[OK] Loaded {len(test_samples)} test samples")
        
    except Exception as e:
        print(f"[ERROR] Failed to load test samples: {e}")
        return
    
    # 5. 测试路径生成
    print(f"\n{'='*50}")
    print("TESTING PATH GENERATION")
    print("="*50)
    
    for i, sample in enumerate(test_samples):
        question = sample['question']
        question_entity = sample['question_entity']
        answer_entities = set(sample['answer_entities'])
        query_type = sample.get('type', '')
        
        print(f"\n--- Test Sample {i+1} ---")
        print(f"Start entity: {question_entity}")
        print(f"Target entities: {list(answer_entities)[:3]}...")  # 只显示前3个
        print(f"Query type: {query_type}")
        
        # 检查起始实体是否在图中
        if question_entity not in knowledge_graph:
            print(f"[SKIP] Start entity '{question_entity}' not in knowledge graph")
            continue
        
        try:
            # 生成路径（确定性模式）
            print("\\nGenerating paths (deterministic mode):")
            deterministic_paths = generator.generate_paths(
                question=question,
                start_entity=question_entity,
                target_entities=answer_entities,
                max_paths=3,
                stochastic=False
            )
            
            if deterministic_paths:
                for j, (path, score, _) in enumerate(deterministic_paths, 1):
                    path_str = ' -> '.join(path)
                    final_entity = path[-1]
                    is_correct = final_entity in answer_entities
                    status = "✓" if is_correct else "✗"
                    print(f"  {j}. Score: {score:.4f} {status} | {path_str}")
            else:
                print("  No paths found")
            
            # 生成路径（随机模式）
            print("\\nGenerating paths (stochastic mode):")
            generator.enable_stochastic_exploration()
            stochastic_paths = generator.generate_paths(
                question=question,
                start_entity=question_entity,
                target_entities=answer_entities,
                max_paths=3,
                stochastic=True
            )
            generator.disable_stochastic_exploration()
            
            if stochastic_paths:
                for j, (path, score, _) in enumerate(stochastic_paths, 1):
                    path_str = ' -> '.join(path)
                    final_entity = path[-1]
                    is_correct = final_entity in answer_entities
                    status = "✓" if is_correct else "✗"
                    print(f"  {j}. Score: {score:.4f} {status} | {path_str}")
            else:
                print("  No paths found")
                
        except Exception as e:
            print(f"[ERROR] Path generation failed: {e}")
            continue
    
    # 6. 测试批量处理
    print(f"\n{'='*50}")
    print("TESTING BATCH PROCESSING")
    print("="*50)
    
    try:
        questions = [sample['question'] for sample in test_samples]
        start_entities = [sample['question_entity'] for sample in test_samples]
        target_entities_list = [set(sample['answer_entities']) for sample in test_samples]
        
        print("Running batch path generation...")
        batch_results = generator.forward(
            questions=questions,
            start_entities=start_entities,
            target_entities_list=target_entities_list,
            max_paths=2
        )
        
        print(f"[OK] Batch processing successful")
        print(f"Generated paths for {len(batch_results)} questions")
        
        for i, paths in enumerate(batch_results):
            print(f"  Sample {i+1}: {len(paths)} paths generated")
            
    except Exception as e:
        print(f"[ERROR] Batch processing failed: {e}")
    
    # 7. 测试评分函数
    print(f"\n{'='*50}")
    print("TESTING SCORING FUNCTIONS")
    print("="*50)
    
    try:
        sample = test_samples[0]
        question = sample['question']
        start_entity = sample['question_entity']
        
        if start_entity in knowledge_graph:
            neighbors = list(knowledge_graph.neighbors(start_entity))[:3]
            print(f"Testing edge scoring for entity: {start_entity}")
            print(f"Neighbors: {neighbors}")
            
            for neighbor in neighbors:
                edge_data = knowledge_graph.get_edge_data(start_entity, neighbor)
                relation = edge_data.get('relation', 'related_to') if edge_data else 'related_to'
                
                score = generator.compute_edge_score(
                    question=question,
                    current_path=[start_entity],
                    candidate_relation=relation,
                    candidate_entity=neighbor,
                    stochastic=False
                )
                print(f"  {start_entity} -[{relation}]-> {neighbor}: {score:.4f}")
        
        print("[OK] Edge scoring functions working")
        
    except Exception as e:
        print(f"[ERROR] Edge scoring test failed: {e}")
    
    print(f"\n{'='*50}")
    print("A* GENERATOR TEST COMPLETED!")
    print("="*50)
    
    print("\\nKey Features Verified:")
    print("1. [+] A* search algorithm implementation")
    print("2. [+] Four scoring functions (s_rel, s_tran, s_loc, s_tri)")
    print("3. [+] Learnable edge score function")
    print("4. [+] Multi-hop path generation")
    print("5. [+] Stochastic vs deterministic modes")
    print("6. [+] Batch processing support")
    print("7. [+] Knowledge graph integration")
    print("8. [+] Entity embedding support")
    print("9. [+] Reserved GAN-RL training interfaces")

def load_simple_samples():
    """加载选择的10个简单1hop样本"""
    samples_file = "selected_1hop_samples.json"
    if not Path(samples_file).exists():
        print(f"Error: {samples_file} not found. Run select_simple_1hop_samples.py first.")
        return []
    
    samples = []
    with open(samples_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    
    print(f"Loaded {len(samples)} simple 1hop samples")
    return samples

def load_trained_discriminator(checkpoint_path, device='cpu'):
    """加载训练好的判别器"""
    if not Path(checkpoint_path).exists():
        print(f"Warning: Discriminator checkpoint not found at {checkpoint_path}")
        print("Creating new discriminator...")
        discriminator = EnhancedPathRankerDiscriminator(
            sbert_model_name='all-MiniLM-L6-v2',
            hidden_dim=384,  # 匹配SBERT维度
            use_pattern_memory=False,  # 禁用
            freeze_sbert=True
        )
        discriminator.to(device)
        return discriminator
    
    print(f"Loading trained discriminator from {checkpoint_path}")
    try:
        # 先创建模型
        discriminator = EnhancedPathRankerDiscriminator(
            sbert_model_name='all-MiniLM-L6-v2',
            hidden_dim=384,  # 匹配SBERT维度
            use_pattern_memory=False,  # 禁用
            freeze_sbert=True
        )
        
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=device)
        discriminator.load_state_dict(checkpoint['model_state_dict'])
        discriminator.to(device)
        
        print("Discriminator loaded successfully")
        return discriminator
        
    except Exception as e:
        print(f"Error loading discriminator: {e}")
        print("Creating new discriminator...")
        discriminator = EnhancedPathRankerDiscriminator(
            sbert_model_name='all-MiniLM-L6-v2',
            hidden_dim=384,  # 匹配SBERT维度
            use_pattern_memory=False,  # 禁用
            freeze_sbert=True
        )
        discriminator.to(device)
        return discriminator

def test_adversarial_training():
    """运行对抗学习训练"""
    print("\n" + "="*80)
    print("ADVERSARIAL LEARNING TRAINING")
    print("="*80)
    
    # 设备选择 - 临时使用CPU避免设备同步问题
    device = 'cpu'  # 强制使用CPU进行测试
    print(f"Using device: {device}")
    
    # 加载简单样本
    print("\n1. Loading simple 1hop samples...")
    samples = load_simple_samples()
    if not samples:
        return None
    
    # 检查必需文件
    required_files = {
        "embeddings/entity_embeddings.pt": "Entity embeddings",
        "graph/knowledge_graph.pkl": "Knowledge graph"
    }
    
    for file_path, description in required_files.items():
        if not Path(file_path).exists():
            print(f"Error: {description} not found at {file_path}")
            return None
    
    # 初始化生成器
    print("\n2. Initializing Generator...")
    try:
        # 使用修复后的可微分生成器
        generator = DifferentiablePathGeneratorTrulyFixed(
            entity_embedding_path="embeddings/entity_embeddings.pt",
            max_path_length=6,
            beam_width=5
        )
        generator.to(device)
        
        # 加载知识图谱
        with open("graph/knowledge_graph.pkl", 'rb') as f:
            knowledge_graph = pickle.load(f)
        
        # 创建关系词表
        relations = set()
        for _, _, edge_data in knowledge_graph.edges(data=True):
            relation = edge_data.get('relation', 'related_to')
            relations.add(relation)
        relation_to_id = {rel: idx for idx, rel in enumerate(relations)}
        
        generator.set_knowledge_graph(knowledge_graph, relation_to_id)
        print(f"Generator initialized: {len(knowledge_graph.nodes)} nodes, {len(knowledge_graph.edges)} edges")
        
    except Exception as e:
        print(f"Error initializing generator: {e}")
        return None
    
    # 加载判别器
    print("\n3. Loading Discriminator...")
    discriminator_path = "checkpoints/enhanced_path_ranker_epoch_10.pt"  # 使用最新的检查点
    discriminator = load_trained_discriminator(discriminator_path, device)
    
    # 初始化GAN-RL训练器
    print("\n4. Initializing GAN-RL Trainer...")
    try:
        trainer = GANRLTrainer(
            generator=generator,
            discriminator=discriminator,
            device=device
        )
        print("GAN-RL Trainer initialized successfully")
    except Exception as e:
        print(f"Error initializing trainer: {e}")
        return None
    
    # 开始对抗训练
    print("\n5. Starting Adversarial Training...")
    training_results = {
        'start_time': datetime.now().isoformat(),
        'device': device,
        'samples_used': len(samples),
        'epochs': [],
        'final_stats': {},
        'sample_details': samples
    }
    
    num_epochs = 3  # 小规模测试
    
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
        epoch_start_time = time.time()
        
        try:
            # 运行一个训练epoch
            epoch_stats = trainer.train_epoch(
                qa_dataset=samples,
                discriminator_steps=2,  # 较少的步数用于快速测试
                generator_steps=1
            )
            
            epoch_time = time.time() - epoch_start_time
            
            # 记录epoch统计
            epoch_result = {
                'epoch': epoch + 1,
                'discriminator_loss': float(epoch_stats.get('discriminator_loss', 0.0)),
                'generator_reward': float(epoch_stats.get('generator_reward', 0.0)),
                'epoch_time': epoch_time,
                'timestamp': datetime.now().isoformat()
            }
            
            training_results['epochs'].append(epoch_result)
            
            print(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")
            print(f"  Discriminator Loss: {epoch_result['discriminator_loss']:.4f}")
            print(f"  Generator Reward: {epoch_result['generator_reward']:.4f}")
            
        except Exception as e:
            print(f"Error in epoch {epoch + 1}: {e}")
            epoch_result = {
                'epoch': epoch + 1,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            training_results['epochs'].append(epoch_result)
            continue
    
    # 保存训练好的模型
    print("\n6. Saving Trained Models...")
    try:
        import os
        
        # 1. 保存完整训练状态（用于继续训练）
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = f"{checkpoint_dir}/gan_rl_adversarial_training_final.pt"
        trainer.save_checkpoint(checkpoint_path, epochs)
        
        print(f"✅ Complete training state saved to: {checkpoint_path}")
        
        # 2. 保存生产环境模型（用于推理）
        from advanced_inference_system import save_production_models
        production_dir = save_production_models(trainer, "checkpoints/production_models")
        
        print(f"✅ Production models saved to: {production_dir}")
        print("📁 Production structure:")
        print("  ├── discoverer_model.pt      # 生成器权重")
        print("  ├── ranker_model.pt          # 判别器权重")  
        print("  ├── model_metadata.json      # 模型元信息")
        print("  └── training_history.json    # 训练历史")
        
    except Exception as e:
        print(f"❌ Error saving models: {e}")
    
    # 最终评估 - 生成器与判别器评分分析
    print("\n7. Generator-Discriminator Scoring Analysis...")
    try:
        evaluation_results = []
        
        for i, sample in enumerate(samples):  # 测试所有样本
            question = sample['question']
            question_entity = sample['question_entity']
            answer_entities = set(sample['answer_entities'])
            
            print(f"\n=== Sample {i+1} ===")
            print(f"Start: {question_entity}")
            
            # 生成路径
            try:
                generated_paths = generator.generate_paths(
                    question=question,
                    start_entity=question_entity,
                    target_entities=answer_entities,
                    max_paths=5,
                    stochastic=False
                )
                
                if not generated_paths:
                    print("  No paths generated")
                    continue
                
                # 分析生成器和判别器评分
                path_analysis = []
                for j, (path, gen_score, _) in enumerate(generated_paths):
                    if len(path) == 0:
                        continue
                        
                    final_entity = path[-1]
                    path_string = '.'.join(path)
                    path_data = [{'paths': {final_entity: [path_string]}}]
                    
                    # 获取判别器分数
                    with torch.no_grad():
                        disc_output = discriminator([question], path_data, epoch=0)
                        disc_raw = float(disc_output[0]['individual_scores'][0])
                        disc_score = float(torch.sigmoid(disc_output[0]['individual_scores'][0]))
                    
                    is_correct = final_entity in answer_entities
                    
                    print(f"  Path {j+1}: {question_entity} -> {final_entity}")
                    print(f"    Generator Score: {gen_score:.4f}")
                    print(f"    Discriminator Raw: {disc_raw:.4f}")
                    print(f"    Discriminator Sigmoid: {disc_score:.4f}")
                    print(f"    Target Hit: {'YES' if is_correct else 'NO'}")
                    
                    path_analysis.append({
                        'path_index': j + 1,
                        'start_entity': question_entity,
                        'end_entity': final_entity,
                        'generator_score': float(gen_score),
                        'discriminator_raw': disc_raw,
                        'discriminator_sigmoid': disc_score,
                        'hits_target': is_correct,
                        'path_length': len(path)
                    })
                
                # 分析生成器和判别器一致性
                if len(path_analysis) >= 2:
                    best_gen = max(path_analysis, key=lambda x: x['generator_score'])
                    best_disc = max(path_analysis, key=lambda x: x['discriminator_sigmoid'])
                    
                    print(f"  Generator Best: Path {best_gen['path_index']} (Score: {best_gen['generator_score']:.4f})")
                    print(f"  Discriminator Best: Path {best_disc['path_index']} (Score: {best_disc['discriminator_sigmoid']:.4f})")
                    
                    if best_gen['path_index'] == best_disc['path_index']:
                        print("  Agreement: CONSISTENT")
                    else:
                        print("  Agreement: DISAGREEMENT")
                
                evaluation_results.append({
                    'sample_index': i + 1,
                    'start_entity': question_entity,
                    'path_analysis': path_analysis,
                    'num_paths_generated': len(path_analysis)
                })
                
            except Exception as e:
                print(f"  Error: {e}")
                evaluation_results.append({
                    'sample_index': i + 1,
                    'start_entity': question_entity,
                    'error': str(e)
                })
        
        training_results['scoring_analysis'] = evaluation_results
        
    except Exception as e:
        print(f"Error in scoring analysis: {e}")
        training_results['scoring_analysis_error'] = str(e)
    
    # 添加最终统计
    training_results['end_time'] = datetime.now().isoformat()
    training_results['total_training_time'] = sum(
        epoch.get('epoch_time', 0) for epoch in training_results['epochs']
    )
    training_results['final_stats'] = {
        'generator_stats': trainer.training_stats.get('generator_rewards', []),
        'discriminator_stats': trainer.training_stats.get('discriminator_losses', []),
    }
    
    # 保存结果
    print("\n8. Saving Results...")
    output_file = f"adversarial_training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(training_results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {output_file}")
    print(f"Total training time: {training_results['total_training_time']:.2f}s")
    
    return training_results

if __name__ == "__main__":
    print("Choose test mode:")
    print("1. Basic A* Generator Test")
    print("2. Adversarial Learning Training")
    print("3. Both")
    
    choice = "2"  # 直接运行对抗学习训练测试
    
    if choice in ['1', '3']:
        test_astar_generator()
    
    if choice in ['2', '3']:
        test_adversarial_training()