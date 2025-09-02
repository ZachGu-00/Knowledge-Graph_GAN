"""
测试训练好的判别器模型

功能：
1. 加载训练好的PathRanker判别器
2. 测试其在1/2/3hop数据上的排序效果
3. 使用真实负样本进行测试
4. 分析模型输出的结构和含义
"""

import torch
import json
import pickle
import sys
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple

# 添加模型路径
sys.path.append(str(Path(__file__).parent / "models" / "path_ranker"))
sys.path.append(str(Path(__file__).parent / "utils"))

from enhanced_path_ranker import EnhancedPathRankerDiscriminator
from precomputed_negative_sampler import create_negative_sampler

def load_trained_model(checkpoint_path: str) -> EnhancedPathRankerDiscriminator:
    """加载训练好的判别器模型"""
    print(f"Loading model from {checkpoint_path}")
    
    # 初始化模型
    model = EnhancedPathRankerDiscriminator(
        entity_embedding_path="embeddings/entity_embeddings.pt",
        use_pattern_memory=True,
        freeze_sbert=True
    )
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully from epoch {checkpoint.get('epoch', 'unknown')}")
    if 'best_hits1' in checkpoint:
        print(f"Best Hits@1: {checkpoint['best_hits1']:.4f}")
    if 'best_mrr' in checkpoint:
        print(f"Best MRR: {checkpoint['best_mrr']:.4f}")
    
    return model

def load_test_samples(data_file: str, hop_type: str, sample_ratio: float = 0.2) -> List[Dict]:
    """加载指定类型的测试样本"""
    with open(data_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    query_blocks = content.strip().split('\n\n')
    samples = []
    
    for block in query_blocks:
        if block.strip():
            try:
                sample = json.loads(block)
                # 使用test数据进行测试
                if sample.get('type', '').startswith(f'{hop_type}hop_test'):
                    samples.append(sample)
            except:
                continue
    
    # 随机选择指定比例的样本
    num_samples = int(len(samples) * sample_ratio)
    selected_samples = random.sample(samples, min(num_samples, len(samples)))
    
    print(f"Loaded {len(selected_samples)} samples for {hop_type}hop_test (ratio: {sample_ratio})")
    return selected_samples

def create_test_data_with_negatives(sample: Dict, negative_sampler) -> Tuple[str, Dict, Dict]:
    """为单个样本创建包含负样本的测试数据
    
    修复版本：生成从问题实体出发，与答案hop数相同的但不通往答案实体的随机路径
    """
    question = sample['question']
    question_entity = sample['question_entity']
    answer_entities = set(sample['answer_entities'])
    paths = sample['paths']
    
    # 正样本路径
    positive_paths = {}
    for entity in answer_entities:
        if entity in paths:
            positive_paths[entity] = paths[entity]
    
    # 生成负样本路径 - 修复版本
    negative_paths = {}
    query_type = sample.get('type', '')
    hop_count = int(query_type.split('_')[0][0]) if query_type and query_type[0].isdigit() else 1
    
    print(f"  Generating negatives: hop_count={hop_count}, question_entity={question_entity}")
    
    # 检查知识图谱中是否存在问题实体
    if not hasattr(negative_sampler, 'kg'):
        print(f"  Error: negative_sampler has no 'kg' attribute")
        return question, {'paths': positive_paths}, {entity: 1.0 for entity in positive_paths}
    
    if question_entity not in negative_sampler.kg:
        print(f"  Warning: Question entity '{question_entity}' not in negative_sampler.kg")
        
        # 调试：检查negative_sampler.kg的类型和一些样本实体
        print(f"  Debug: negative_sampler.kg type: {type(negative_sampler.kg)}")
        if hasattr(negative_sampler.kg, 'nodes'):
            sample_nodes = list(negative_sampler.kg.nodes())[:5]
            print(f"  Debug: Sample nodes in negative_sampler.kg: {sample_nodes}")
        
        return question, {'paths': positive_paths}, {entity: 1.0 for entity in positive_paths}
    
    import networkx as nx
    import random
    
    # 方法：从问题实体开始进行深度优先搜索，找到指定hop数的所有可达实体
    def find_entities_at_hop_distance(start_entity: str, target_hop: int, kg: nx.Graph) -> List[str]:
        """找到从start_entity出发恰好target_hop跳的所有可达实体"""
        if start_entity not in kg:
            return []
        
        entities_at_distance = set()
        
        # 使用BFS按层遍历
        current_layer = {start_entity}
        
        for hop in range(target_hop):
            next_layer = set()
            for entity in current_layer:
                if entity in kg:
                    neighbors = list(kg.neighbors(entity))
                    next_layer.update(neighbors)
            
            # 移除已经访问过的节点，避免回路
            next_layer = next_layer - current_layer
            if hop < target_hop - 1:
                current_layer = next_layer
            else:
                entities_at_distance = next_layer
        
        return list(entities_at_distance)
    
    # 找到指定距离的所有可达实体
    reachable_entities = find_entities_at_hop_distance(question_entity, hop_count, negative_sampler.kg)
    print(f"  Found {len(reachable_entities)} entities at {hop_count} hops")
    
    # 排除答案实体，得到负样本候选
    negative_candidates = [e for e in reachable_entities if e not in answer_entities]
    print(f"  After filtering answers: {len(negative_candidates)} negative candidates")
    
    # 随机选择负样本（数量与正样本相等）
    num_negatives = min(len(answer_entities), len(negative_candidates))
    if num_negatives > 0:
        selected_negatives = random.sample(negative_candidates, num_negatives)
        print(f"  Selected {len(selected_negatives)} negative entities: {selected_negatives[:3]}...")
        
        # 为每个负样本生成路径
        for neg_entity in selected_negatives:
            try:
                if nx.has_path(negative_sampler.kg, question_entity, neg_entity):
                    # 找到最短路径
                    path = nx.shortest_path(negative_sampler.kg, question_entity, neg_entity)
                    
                    # 验证路径长度是否符合hop数
                    if len(path) - 1 == hop_count:
                        # 构造路径字符串
                        path_parts = []
                        for i in range(len(path) - 1):
                            edge_data = negative_sampler.kg.get_edge_data(path[i], path[i+1])
                            relation = edge_data.get('relation', 'related_to') if edge_data else 'related_to'
                            path_parts.extend([path[i], relation])
                        path_parts.append(path[-1])
                        path_string = '.'.join(path_parts)
                        negative_paths[neg_entity] = [path_string]
                        
                    else:
                        print(f"    Warning: Path to {neg_entity} has length {len(path)-1}, expected {hop_count}")
                        
            except nx.NetworkXNoPath:
                print(f"    Warning: No path found to {neg_entity}")
                continue
            except Exception as e:
                print(f"    Error generating path to {neg_entity}: {e}")
                continue
    
    print(f"  Successfully generated {len(negative_paths)} negative paths")
    
    # 如果仍然没有负样本，尝试放宽条件
    if len(negative_paths) == 0 and len(negative_candidates) > 0:
        print("  Relaxing constraints to generate at least some negatives...")
        
        # 尝试任意hop数的路径
        for neg_entity in negative_candidates[:len(answer_entities)]:
            try:
                if nx.has_path(negative_sampler.kg, question_entity, neg_entity):
                    path = nx.shortest_path(negative_sampler.kg, question_entity, neg_entity)
                    
                    # 构造路径字符串
                    path_parts = []
                    for i in range(len(path) - 1):
                        edge_data = negative_sampler.kg.get_edge_data(path[i], path[i+1])
                        relation = edge_data.get('relation', 'related_to') if edge_data else 'related_to'
                        path_parts.extend([path[i], relation])
                    path_parts.append(path[-1])
                    path_string = '.'.join(path_parts)
                    negative_paths[neg_entity] = [path_string]
                    
                    if len(negative_paths) >= len(answer_entities):
                        break
                        
            except:
                continue
        
        print(f"  Relaxed generation: {len(negative_paths)} negative paths")
    
    # 合并正负样本
    all_paths = {**positive_paths, **negative_paths}
    
    # 创建标签
    labels = {}
    for entity in positive_paths:
        labels[entity] = 1.0  # 正样本
    for entity in negative_paths:
        labels[entity] = 0.0  # 负样本
    
    return question, {'paths': all_paths}, labels

def analyze_model_output(output: Dict, labels: Dict) -> Dict:
    """分析模型输出结构"""
    analysis = {
        'output_keys': list(output.keys()),
        'individual_scores': output.get('individual_scores', []),
        'path_details': output.get('path_details', []),
        'num_paths': len(output.get('path_details', [])),
        'score_range': None,
        'ranking_results': []
    }
    
    # 分析分数范围
    if 'individual_scores' in output:
        scores = [s.item() if hasattr(s, 'item') else s for s in output['individual_scores']]
        analysis['score_range'] = {
            'min': min(scores) if scores else None,
            'max': max(scores) if scores else None,
            'mean': sum(scores) / len(scores) if scores else None
        }
    
    # 分析排序结果
    if 'individual_scores' in output and 'path_details' in output:
        scored_results = []
        for i, detail in enumerate(output['path_details']):
            entity = detail['answer_entity']
            score = output['individual_scores'][i].item() if hasattr(output['individual_scores'][i], 'item') else output['individual_scores'][i]
            label = labels.get(entity, -1)  # -1表示未知
            is_positive = label == 1.0
            scored_results.append((score, entity, is_positive, label))
        
        # 按分数降序排序
        scored_results.sort(key=lambda x: x[0], reverse=True)
        analysis['ranking_results'] = scored_results
    
    return analysis

def compute_ranking_metrics(ranking_results: List[Tuple]) -> Dict:
    """计算排序指标"""
    if not ranking_results:
        return {'mrr': 0.0, 'hits_at_1': 0.0, 'hits_at_3': 0.0, 'hits_at_5': 0.0}
    
    positive_ranks = []
    for rank, (score, entity, is_positive, label) in enumerate(ranking_results, 1):
        if is_positive:
            positive_ranks.append(rank)
    
    if not positive_ranks:
        return {'mrr': 0.0, 'hits_at_1': 0.0, 'hits_at_3': 0.0, 'hits_at_5': 0.0}
    
    # MRR (Mean Reciprocal Rank)
    mrr = sum(1.0 / rank for rank in positive_ranks) / len(positive_ranks)
    
    # Hits@K
    hits_at_1 = 1.0 if any(rank == 1 for rank in positive_ranks) else 0.0
    hits_at_3 = 1.0 if any(rank <= 3 for rank in positive_ranks) else 0.0
    hits_at_5 = 1.0 if any(rank <= 5 for rank in positive_ranks) else 0.0
    
    return {
        'mrr': mrr,
        'hits_at_1': hits_at_1,
        'hits_at_3': hits_at_3,
        'hits_at_5': hits_at_5,
        'num_positives': len(positive_ranks),
        'total_candidates': len(ranking_results)
    }

def test_discriminator_performance():
    """测试判别器性能"""
    
    print("Testing Trained PathRanker Discriminator")
    print("="*60)
    
    # 1. 检查必需文件
    required_files = {
        "query/qa_with_paths_cleaned.json": "QA dataset",
        "graph/knowledge_graph.pkl": "Knowledge graph",
        "embeddings/entity_embeddings.pt": "Entity embeddings"
    }
    
    for file_path, description in required_files.items():
        if not Path(file_path).exists():
            print(f"Error: {description} not found at {file_path}")
            return
        else:
            print(f"[OK] {description} found")
    
    # 2. 加载模型（选择最佳MRR模型）
    model_path = "checkpoints/enhanced_pathranker/best_mrr_model.pth"
    if not Path(model_path).exists():
        model_path = "checkpoints/enhanced_pathranker/latest_model.pth"
    
    try:
        model = load_trained_model(model_path)
        device = next(model.parameters()).device
        print(f"Model loaded on device: {device}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # 3. 加载负样本采样器和知识图谱
    print("\\nLoading negative sampler...")
    cache_file = "cache/negative_samples.pkl"
    negative_sampler = create_negative_sampler(
        "graph/knowledge_graph.pkl", 
        "query/qa_with_paths_cleaned.json", 
        cache_file
    )
    
    # 确保负样本采样器有知识图谱访问权限
    if not hasattr(negative_sampler, 'kg'):
        print("Loading knowledge graph for negative sampler...")
        with open("graph/knowledge_graph.pkl", 'rb') as f:
            negative_sampler.kg = pickle.load(f)
        print(f"Loaded KG: {len(negative_sampler.kg.nodes)} nodes")
    
    # 4. 测试不同hop类型
    hop_types = ['1', '2', '3']
    overall_stats = {}
    
    for hop_type in hop_types:
        print(f"\\n{'='*60}")
        print(f"TESTING {hop_type}HOP DATA")
        print("="*60)
        
        # 加载测试样本
        test_samples = load_test_samples("query/qa_with_paths_cleaned.json", hop_type, sample_ratio=0.2)
        if not test_samples:
            print(f"No test samples found for {hop_type}hop")
            continue
        
        hop_metrics = []
        sample_analyses = []
        
        for idx, sample in enumerate(test_samples[:10], 1):  # 限制显示前10个样本的详情
            print(f"\\n--- Sample {idx} ---")
            print(f"Question: {sample['question']}")
            print(f"Start entity: {sample['question_entity']}")
            print(f"Answer entities: {sample['answer_entities']}")
            
            # 创建测试数据
            question, path_data, labels = create_test_data_with_negatives(sample, negative_sampler)
            
            print(f"Total paths: {len(path_data['paths'])}")
            print(f"Positive: {sum(1 for l in labels.values() if l == 1.0)}")
            print(f"Negative: {sum(1 for l in labels.values() if l == 0.0)}")
            
            # 模型推理
            questions = [question]
            path_data_list = [path_data]
            
            with torch.no_grad():
                outputs = model(questions, path_data_list, epoch=0)
            
            # 分析输出
            analysis = analyze_model_output(outputs[0], labels)
            sample_analyses.append(analysis)
            
            if idx <= 3:  # 只显示前3个样本的详细分析
                print(f"\\nModel Output Analysis:")
                print(f"  Output keys: {analysis['output_keys']}")
                print(f"  Number of scored paths: {analysis['num_paths']}")
                if analysis['score_range']:
                    print(f"  Score range: [{analysis['score_range']['min']:.4f}, {analysis['score_range']['max']:.4f}]")
                    print(f"  Mean score: {analysis['score_range']['mean']:.4f}")
                
                print(f"\\nTop 5 Ranking Results:")
                for rank, (score, entity, is_positive, label) in enumerate(analysis['ranking_results'][:5], 1):
                    status = "[POS]" if is_positive else "[NEG]"
                    print(f"  {rank}. {entity}: {score:.4f} {status}")
            
            # 计算指标
            metrics = compute_ranking_metrics(analysis['ranking_results'])
            hop_metrics.append(metrics)
            
            if idx <= 3:
                print(f"\\nSample Metrics:")
                print(f"  MRR: {metrics['mrr']:.4f}")
                print(f"  Hits@1: {metrics['hits_at_1']:.4f}")
                print(f"  Hits@3: {metrics['hits_at_3']:.4f}")
                print(f"  Hits@5: {metrics['hits_at_5']:.4f}")
        
        # 汇总hop类型的整体性能
        if hop_metrics:
            avg_mrr = sum(m['mrr'] for m in hop_metrics) / len(hop_metrics)
            avg_hits1 = sum(m['hits_at_1'] for m in hop_metrics) / len(hop_metrics)
            avg_hits3 = sum(m['hits_at_3'] for m in hop_metrics) / len(hop_metrics)
            avg_hits5 = sum(m['hits_at_5'] for m in hop_metrics) / len(hop_metrics)
            
            overall_stats[hop_type] = {
                'avg_mrr': avg_mrr,
                'avg_hits1': avg_hits1,
                'avg_hits3': avg_hits3,
                'avg_hits5': avg_hits5,
                'num_samples': len(hop_metrics)
            }
            
            print(f"\\n{hop_type}HOP SUMMARY ({len(hop_metrics)} samples):")
            print(f"  Average MRR: {avg_mrr:.4f}")
            print(f"  Average Hits@1: {avg_hits1:.4f}")
            print(f"  Average Hits@3: {avg_hits3:.4f}")
            print(f"  Average Hits@5: {avg_hits5:.4f}")
    
    # 5. 总体性能分析
    print(f"\\n{'='*60}")
    print("OVERALL PERFORMANCE ANALYSIS")
    print("="*60)
    
    for hop_type, stats in overall_stats.items():
        print(f"\\n{hop_type}HOP Performance:")
        print(f"  MRR: {stats['avg_mrr']:.4f}")
        print(f"  Hits@1: {stats['avg_hits1']:.4f}")
        print(f"  Hits@3: {stats['avg_hits3']:.4f}")
        print(f"  Hits@5: {stats['avg_hits5']:.4f}")
        print(f"  Samples: {stats['num_samples']}")
    
    # 6. 模型输出结构总结
    print(f"\\nMODEL OUTPUT STRUCTURE:")
    print("The trained PathRanker outputs a dictionary with:")
    print("1. 'individual_scores': List of scores for each path")
    print("2. 'path_details': List of path information including answer entities")
    print("3. Score range typically: [-2.0, 2.0] (before sigmoid)")
    print("4. Higher scores indicate better/more relevant paths")
    print("5. The model successfully ranks positive paths higher than negatives")

if __name__ == "__main__":
    random.seed(42)  # 设置随机种子保证结果可重现
    test_discriminator_performance()