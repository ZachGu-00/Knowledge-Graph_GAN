import numpy as np
import networkx as nx
import pickle
import json
from pathlib import Path
from collections import defaultdict, Counter
import os

class CompleteKnowledgeGraphBuilder:
    def __init__(self, data_path="data", output_path="graph"):
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
        
        # Data structures
        self.entity_names = set()       # 所有实体名称
        self.entity_case_map = {}       # lowercase -> preferred case mapping
        self.embeddings = None
        self.graph = nx.Graph()         # 无向图
        self.relation_types = set()
        
    def load_entity_names(self):
        """Load entity names from kb_entity_dict.txt and normalize case"""
        print("Loading entity names...")
        
        entity_file = self.data_path / "kb_entity_dict.txt"
        with open(entity_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if '\t' in line:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        name = parts[1]
                        name_lower = name.lower()
                        
                        # 保留第一次出现的大小写格式
                        if name_lower not in self.entity_case_map:
                            self.entity_case_map[name_lower] = name
                            self.entity_names.add(name)
        
        original_count = sum(1 for _ in open(entity_file, 'r', encoding='utf-8'))
        print(f"Loaded {len(self.entity_names)} unique entity names (original: {original_count})")
        print(f"Eliminated {original_count - len(self.entity_names)} case duplicates")
        return self
    
    def load_embeddings(self):
        """Load entity embeddings from kb_entity.npz"""
        print("Loading embeddings...")
        
        embedding_file = self.data_path / "kb_entity.npz"
        if embedding_file.exists():
            self.embeddings = np.load(embedding_file)
            print(f"Loaded embeddings for {len(self.embeddings.files)} entities")
        else:
            print("No embeddings file found")
        
        return self
    
    def load_relations(self):
        """Load relations from kb.txt with case normalization"""
        print("Loading relations from kb.txt...")
        
        relation_file = self.data_path / "kb.txt"
        relations_loaded = 0
        
        with open(relation_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if '|' in line:
                    parts = line.split('|')
                    if len(parts) == 3:
                        subject, relation, obj = parts
                        
                        # 使用大小写映射获取规范化的实体名称
                        subject_normalized = self.entity_case_map.get(subject.lower())
                        obj_normalized = self.entity_case_map.get(obj.lower())
                        
                        if subject_normalized and obj_normalized:
                            self.graph.add_edge(subject_normalized, obj_normalized, relation=relation)
                            self.relation_types.add(relation)
                            relations_loaded += 1
        
        print(f"Loaded {relations_loaded} relations")
        print(f"Relation types: {sorted(self.relation_types)}")
        return self
    
    def add_entity_nodes(self):
        """Add all entities as nodes to the graph"""
        print("Adding entity nodes...")
        
        nodes_added = 0
        for entity_name in self.entity_names:
            if not self.graph.has_node(entity_name):
                node_type = self._classify_entity(entity_name)
                self.graph.add_node(entity_name, type=node_type)
                nodes_added += 1
        
        print(f"Added {nodes_added} nodes to graph")
        print(f"Total graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        return self
    
    def _classify_entity(self, name):
        """Classify entity type based on name"""
        name_lower = name.lower()
        
        # Years
        if name.isdigit() and 1900 <= int(name) <= 2030:
            return 'year'
        
        # Common genres
        genres = {'drama', 'comedy', 'action', 'horror', 'thriller', 'romance', 
                 'war', 'documentary', 'adventure', 'crime', 'fantasy', 'mystery',
                 'sci-fi', 'western', 'animation', 'family', 'biography'}
        if name_lower in genres:
            return 'genre'
        
        # Languages
        languages = {'english', 'french', 'spanish', 'german', 'italian', 
                    'japanese', 'chinese', 'russian', 'portuguese', 'korean'}
        if name_lower in languages:
            return 'language'
        
        # Person names (heuristic: contains space and proper case)
        if (' ' in name and 
            len(name.split()) >= 2 and 
            any(word[0].isupper() for word in name.split())):
            return 'person'
        
        # Default to movie/entity
        return 'movie'
    
    def analyze_graph(self):
        """Analyze the constructed graph"""
        print("\n=== Graph Analysis ===")
        
        # Basic stats
        num_nodes = self.graph.number_of_nodes()
        num_edges = self.graph.number_of_edges()
        
        print(f"Nodes: {num_nodes:,}")
        print(f"Edges: {num_edges:,}")
        print(f"Relation types: {len(self.relation_types)}")
        
        # Node type distribution
        node_types = defaultdict(int)
        for _, data in self.graph.nodes(data=True):
            node_types[data.get('type', 'unknown')] += 1
        
        print(f"\nNode type distribution:")
        for node_type, count in sorted(node_types.items(), key=lambda x: -x[1]):
            print(f"  {node_type}: {count:,}")
        
        # Relation distribution
        relations = [data['relation'] for _, _, data in self.graph.edges(data=True)]
        relation_counts = Counter(relations)
        
        print(f"\nRelation distribution:")
        for relation, count in relation_counts.most_common():
            print(f"  {relation}: {count:,}")
        
        # Connectivity analysis
        print(f"\nConnectivity Analysis:")
        
        # Graph is already undirected
        undirected = self.graph
        
        # Connected components
        components = list(nx.connected_components(undirected))
        num_components = len(components)
        
        print(f"  Connected components: {num_components:,}")
        
        if components:
            # Largest component
            largest_component = max(components, key=len)
            largest_size = len(largest_component)
            
            print(f"  Largest component: {largest_size:,} nodes ({largest_size/num_nodes*100:.2f}%)")
            
            # Component size distribution
            comp_sizes = sorted([len(c) for c in components], reverse=True)
            print(f"  Component sizes (top 10): {comp_sizes[:10]}")
            
            # Isolated nodes
            isolated = [c for c in components if len(c) == 1]
            print(f"  Isolated nodes: {len(isolated):,}")
        
        # Degree analysis (undirected graph)
        degrees = dict(self.graph.degree())
        
        if degrees:
            print(f"\nDegree Statistics:")
            print(f"  Average degree: {sum(degrees.values())/len(degrees):.2f}")
            print(f"  Max degree: {max(degrees.values()):,}")
            
            # Top nodes by degree
            top_degree = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]
            print(f"\n  Top 10 nodes by total degree:")
            for entity_name, degree in top_degree:
                node_type = self.graph.nodes[entity_name].get('type', 'unknown')
                print(f"    {entity_name} ({node_type}): {degree}")
        
        return {
            'nodes': num_nodes,
            'edges': num_edges,
            'components': num_components,
            'largest_component': largest_size if components else 0,
            'isolated_nodes': len(isolated) if components else 0,
            'node_types': dict(node_types),
            'relation_types': list(self.relation_types)
        }
    
    def show_examples(self):
        """Show some example subgraphs"""
        print(f"\n=== Example Subgraphs ===")
        
        # Find some interesting nodes
        examples = []
        
        # Find a movie with many connections
        degrees = dict(self.graph.degree())
        top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
        
        for entity_name, degree in top_nodes[:5]:
            node_type = self.graph.nodes[entity_name].get('type', 'unknown')
            if node_type == 'movie' and degree > 10:
                examples.append(entity_name)
                break
        
        # Find a person
        for entity_name, data in self.graph.nodes(data=True):
            if data.get('type') == 'person':
                examples.append(entity_name)
                break
        
        # Show examples
        for i, entity_name in enumerate(examples[:2]):
            node_type = self.graph.nodes[entity_name].get('type', 'unknown')
            
            print(f"\nExample {i+1}: {entity_name} ({node_type})")
            
            # Show connected edges (undirected graph)
            edges = list(self.graph.edges(entity_name, data=True))[:5]
            if edges:
                print("  Connected relations:")
                for source, target, data in edges:
                    other_entity = target if source == entity_name else source
                    relation = data['relation']
                    print(f"    {entity_name} --[{relation}]-- {other_entity}")
    
    def save_graph(self):
        """Save the graph and related data"""
        print(f"\nSaving graph to {self.output_path}...")
        
        # 1. Save the main graph
        graph_file = self.output_path / "knowledge_graph.pkl"
        with open(graph_file, 'wb') as f:
            pickle.dump(self.graph, f)
        print(f"[OK] Saved graph: {graph_file}")
        
        # 2. Save entity names list
        entity_file = self.output_path / "entity_names.json"
        with open(entity_file, 'w', encoding='utf-8') as f:
            json.dump(sorted(list(self.entity_names)), f, ensure_ascii=False, indent=2)
        print(f"[OK] Saved entity names: {entity_file}")
        
        # 3. Skip embeddings (not available)
        # embeddings functionality removed since kb_entity.npz doesn't exist
        
        # 4. Save graph statistics
        stats = self.analyze_graph()
        stats_file = self.output_path / "graph_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"[OK] Saved statistics: {stats_file}")
        
        # 5. Save relation types
        relations_file = self.output_path / "relations.json"
        relations_data = {
            'relation_types': sorted(list(self.relation_types)),
            'total_relations': len(list(self.graph.edges()))
        }
        with open(relations_file, 'w') as f:
            json.dump(relations_data, f, indent=2)
        print(f"[OK] Saved relations: {relations_file}")
        
        return self.output_path

def main():
    """Main function to build the complete knowledge graph"""
    print("=== Building Complete Knowledge Graph ===")
    print("Using files: kb_entity_dict.txt, kb.txt")
    
    try:
        # Build the graph
        builder = CompleteKnowledgeGraphBuilder()
        
        builder.load_entity_names()
        builder.load_embeddings() 
        builder.load_relations()
        builder.add_entity_nodes()
        
        # Analyze the graph
        stats = builder.analyze_graph()
        builder.show_examples()
        
        # Save the graph
        output_path = builder.save_graph()
        
        print(f"\n=== COMPLETE ===")
        print(f"[OK] Knowledge graph built successfully!")
        print(f"[OK] {stats['nodes']:,} nodes, {stats['edges']:,} edges")
        print(f"[OK] {stats['components']:,} connected components")
        print(f"[OK] Largest component: {stats['largest_component']:,} nodes ({stats['largest_component']/stats['nodes']*100:.1f}%)")
        print(f"[OK] {stats['isolated_nodes']:,} isolated nodes")
        print(f"[OK] {len(stats['relation_types'])} relation types")
        print(f"[OK] Files saved to: {output_path}")
        
        return stats
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()