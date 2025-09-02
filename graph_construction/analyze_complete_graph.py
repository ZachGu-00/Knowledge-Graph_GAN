import pickle
import networkx as nx
import json
from collections import Counter, defaultdict
import numpy as np

def load_and_analyze_complete_graph():
    """加载并分析完整的知识图谱"""
    print("=== Complete Knowledge Graph Analysis ===")
    
    # 加载图
    with open('graph/knowledge_graph.pkl', 'rb') as f:
        graph = pickle.load(f)
    
    # 加载实体字典
    with open('graph/entity_dict.json', 'r', encoding='utf-8') as f:
        entity_dict = json.load(f)
    # Convert string keys back to int
    entity_dict = {int(k): v for k, v in entity_dict.items()}
    
    print(f"Graph loaded: {graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges")
    
    # 1. 基本统计
    analyze_basic_stats(graph, entity_dict)
    
    # 2. 连通性分析
    analyze_connectivity(graph, entity_dict)
    
    # 3. 节点类型分析
    analyze_node_types(graph, entity_dict)
    
    # 4. 关系类型分析
    analyze_relations(graph, entity_dict)
    
    # 5. 展示具体例子
    show_graph_examples(graph, entity_dict)
    
    return graph, entity_dict

def analyze_basic_stats(graph, entity_dict):
    """基本统计分析"""
    print(f"\n=== Basic Statistics ===")
    
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    
    print(f"Nodes: {num_nodes:,}")
    print(f"Edges: {num_edges:,}")
    print(f"Average degree: {2*num_edges/num_nodes:.2f}")
    print(f"Graph density: {nx.density(graph):.6f}")
    
    # 度分布
    degrees = dict(graph.degree())
    in_degrees = dict(graph.in_degree())
    out_degrees = dict(graph.out_degree())
    
    print(f"\nDegree Statistics:")
    print(f"  Total degree - Min: {min(degrees.values())}, Max: {max(degrees.values())}, Avg: {sum(degrees.values())/len(degrees):.2f}")
    print(f"  In-degree - Min: {min(in_degrees.values())}, Max: {max(in_degrees.values())}, Avg: {sum(in_degrees.values())/len(in_degrees):.2f}")
    print(f"  Out-degree - Min: {min(out_degrees.values())}, Max: {max(out_degrees.values())}, Avg: {sum(out_degrees.values())/len(out_degrees):.2f}")

def analyze_connectivity(graph, entity_dict):
    """连通性分析"""
    print(f"\n=== Connectivity Analysis ===")
    
    # 转为无向图分析连通性
    undirected = graph.to_undirected()
    
    # 连通分量
    components = list(nx.connected_components(undirected))
    num_components = len(components)
    
    print(f"Connected components: {num_components:,}")
    
    if components:
        # 最大连通分量
        largest_comp = max(components, key=len)
        largest_size = len(largest_comp)
        
        print(f"Largest connected component: {largest_size:,} nodes ({largest_size/graph.number_of_nodes()*100:.2f}%)")
        
        # 连通分量大小分布
        comp_sizes = sorted([len(c) for c in components], reverse=True)
        print(f"Component size distribution (top 10): {comp_sizes[:10]}")
        
        # 孤立节点
        isolated_nodes = [list(c)[0] for c in components if len(c) == 1]
        print(f"Isolated nodes: {len(isolated_nodes):,}")
        
        if isolated_nodes:
            print("Sample isolated nodes:")
            for node_id in isolated_nodes[:10]:
                name = entity_dict.get(node_id, f'Entity-{node_id}')
                print(f"  {name}")
    
    # 强连通分量（有向图）
    strong_components = list(nx.strongly_connected_components(graph))
    print(f"Strongly connected components: {len(strong_components):,}")
    if strong_components:
        largest_strong = max(strong_components, key=len)
        print(f"Largest strongly connected component: {len(largest_strong):,} nodes")

def analyze_node_types(graph, entity_dict):
    """节点类型分析"""
    print(f"\n=== Node Type Analysis ===")
    
    # 重新分类节点
    node_types = {
        'movie': [],
        'person': [],
        'genre': [],
        'year': [],
        'language': [],
        'tag': [],
        'other': []
    }
    
    # 预定义的类型
    genres = {'drama', 'comedy', 'action', 'horror', 'thriller', 'romance', 'war', 
             'documentary', 'adventure', 'crime', 'fantasy', 'mystery', 'sci-fi',
             'western', 'animation', 'family', 'biography'}
    
    languages = {'english', 'french', 'spanish', 'german', 'italian', 'japanese',
                'chinese', 'russian', 'portuguese', 'korean', 'hindi'}
    
    # 分类所有节点
    for node_id in graph.nodes():
        name = entity_dict.get(node_id, '').lower()
        
        if name.isdigit() and 1900 <= int(name) <= 2030:
            node_types['year'].append((node_id, entity_dict[node_id]))
        elif name in genres:
            node_types['genre'].append((node_id, entity_dict[node_id]))
        elif name in languages:
            node_types['language'].append((node_id, entity_dict[node_id]))
        elif (' ' in name and len(name.split()) >= 2 and 
              any(word[0].isupper() for word in entity_dict[node_id].split())):
            node_types['person'].append((node_id, entity_dict[node_id]))
        elif len(name.split()) <= 3 and all(len(word) <= 10 for word in name.split()):
            node_types['tag'].append((node_id, entity_dict[node_id]))
        else:
            node_types['movie'].append((node_id, entity_dict[node_id]))
    
    print("Node type distribution:")
    for node_type, nodes in node_types.items():
        print(f"  {node_type}: {len(nodes):,}")
        if nodes and len(nodes) <= 20:  # 如果数量少，显示所有
            examples = [name for _, name in nodes]
        else:  # 否则显示前5个例子
            examples = [name for _, name in nodes[:5]]
        if examples:
            print(f"    Examples: {examples}")

def analyze_relations(graph, entity_dict):
    """关系类型分析"""
    print(f"\n=== Relation Type Analysis ===")
    
    # 统计每种关系
    relations = [data.get('relation', 'unknown') for _, _, data in graph.edges(data=True)]
    relation_counts = Counter(relations)
    
    print("Relation distribution:")
    total_edges = len(relations)
    for relation, count in relation_counts.most_common():
        percentage = count / total_edges * 100
        print(f"  {relation}: {count:,} ({percentage:.1f}%)")
    
    # 展示每种关系的例子
    print(f"\nRelation examples:")
    relation_examples = defaultdict(list)
    
    for source, target, data in graph.edges(data=True):
        relation = data.get('relation', 'unknown')
        if len(relation_examples[relation]) < 3:  # 每种关系最多3个例子
            source_name = entity_dict.get(source, f'Entity-{source}')
            target_name = entity_dict.get(target, f'Entity-{target}')
            relation_examples[relation].append((source_name, target_name))
    
    for relation, examples in sorted(relation_examples.items()):
        print(f"  {relation}:")
        for source_name, target_name in examples:
            print(f"    {source_name} --[{relation}]--> {target_name}")

def show_graph_examples(graph, entity_dict):
    """显示图结构例子"""
    print(f"\n=== Graph Structure Examples ===")
    
    # 找一些有趣的节点
    degrees = dict(graph.degree())
    
    # 按度数排序找到中心节点
    top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
    
    examples = []
    for node_id, degree in top_nodes:
        name = entity_dict.get(node_id, f'Entity-{node_id}')
        if degree > 50 and len(examples) < 3:  # 找3个高度节点作为例子
            examples.append((node_id, name, degree))
    
    for i, (node_id, name, degree) in enumerate(examples):
        print(f"\nExample {i+1}: {name} (Total degree: {degree})")
        print(f"  In-degree: {graph.in_degree(node_id)}, Out-degree: {graph.out_degree(node_id)}")
        
        # 入度边例子
        in_edges = list(graph.in_edges(node_id, data=True))[:5]
        if in_edges:
            print("  Incoming relations:")
            for source, _, data in in_edges:
                source_name = entity_dict.get(source, f'Entity-{source}')
                relation = data.get('relation', 'unknown')
                print(f"    {source_name} --[{relation}]--> {name}")
        
        # 出度边例子
        out_edges = list(graph.out_edges(node_id, data=True))[:5]
        if out_edges:
            print("  Outgoing relations:")
            for _, target, data in out_edges:
                target_name = entity_dict.get(target, f'Entity-{target}')
                relation = data.get('relation', 'unknown')
                print(f"    {name} --[{relation}]--> {target_name}")

def save_final_stats():
    """保存最终统计信息"""
    with open('graph/knowledge_graph.pkl', 'rb') as f:
        graph = pickle.load(f)
    
    undirected = graph.to_undirected()
    components = list(nx.connected_components(undirected))
    
    stats = {
        'nodes': graph.number_of_nodes(),
        'edges': graph.number_of_edges(),
        'connected_components': len(components),
        'largest_component_size': len(max(components, key=len)) if components else 0,
        'isolated_nodes': len([c for c in components if len(c) == 1]),
        'average_degree': 2 * graph.number_of_edges() / graph.number_of_nodes(),
        'density': nx.density(graph)
    }
    
    with open('graph/final_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    return stats

def main():
    try:
        graph, entity_dict = load_and_analyze_complete_graph()
        stats = save_final_stats()
        
        print(f"\n" + "="*60)
        print("COMPLETE KNOWLEDGE GRAPH BUILT SUCCESSFULLY!")
        print("="*60)
        print(f"📊 FINAL STATISTICS:")
        print(f"   • Total nodes: {stats['nodes']:,}")
        print(f"   • Total edges: {stats['edges']:,}")
        print(f"   • Connected components: {stats['connected_components']:,}")
        print(f"   • Largest component: {stats['largest_component_size']:,} nodes ({stats['largest_component_size']/stats['nodes']*100:.1f}%)")
        print(f"   • Isolated nodes: {stats['isolated_nodes']:,}")
        print(f"   • Average degree: {stats['average_degree']:.2f}")
        print(f"   • Graph density: {stats['density']:.6f}")
        print(f"")
        print(f"📁 Files saved in graph/ folder:")
        print(f"   • knowledge_graph.pkl - Main graph file")
        print(f"   • entity_dict.json - Entity ID to name mapping")
        print(f"   • final_stats.json - Graph statistics")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()