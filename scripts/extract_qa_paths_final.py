import json
import pickle
import networkx as nx
import random
from collections import defaultdict

class QAPathExtractor:
    def __init__(self):
        self.graph = None
        self.qa_data = defaultdict(list)
        
    def load_graph(self):
        """Load the knowledge graph"""
        print("Loading knowledge graph...")
        with open('graph/knowledge_graph.pkl', 'rb') as f:
            self.graph = pickle.load(f)
        
        print(f"Graph loaded: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        return self
    
    def load_qa_data(self):
        """Load and organize QA data by type"""
        print("Loading QA data...")
        
        with open('extracted_qa_normalized.json', 'r', encoding='utf-8') as f:
            for line in f:
                query = json.loads(line.strip())
                query_type = query['type']
                self.qa_data[query_type].append(query)
        
        print("QA data loaded:")
        for query_type, queries in self.qa_data.items():
            print(f"  {query_type}: {len(queries)} queries")
        
        return self
    
    def find_all_shortest_paths(self, source, target):
        """Find all shortest paths between source and target"""
        try:
            all_paths = list(nx.all_shortest_paths(self.graph, source, target))
            return all_paths
        except nx.NetworkXNoPath:
            return []
    
    def path_to_string(self, path):
        """Convert path nodes to relationship chain string"""
        if len(path) < 2:
            return ""
        
        path_parts = [path[0]]  # Start with first node
        
        for i in range(len(path) - 1):
            source_node = path[i]
            target_node = path[i + 1]
            
            # Get edge data
            edge_data = self.graph[source_node][target_node]
            relation = edge_data.get('relation', 'unknown')
            
            path_parts.append(relation)
            path_parts.append(target_node)
        
        return ".".join(path_parts)
    
    def extract_paths_for_query(self, query):
        """Extract paths for a single query"""
        question_entity = query['question_entity']
        answer_entities = query['answer_entities']
        
        # Check if question entity exists in graph
        if question_entity not in self.graph:
            return None
        
        paths_info = {}
        
        for answer_entity in answer_entities:
            if answer_entity not in self.graph:
                continue
                
            # Find all shortest paths
            all_paths = self.find_all_shortest_paths(question_entity, answer_entity)
            
            if all_paths:
                # Convert paths to string format
                path_strings = []
                for path in all_paths:
                    path_string = self.path_to_string(path)
                    if path_string:
                        path_strings.append(path_string)
                
                if path_strings:
                    paths_info[answer_entity] = path_strings
        
        if not paths_info:
            return None
        
        return {
            "type": query['type'],
            "question": query['question'],
            "question_entity": question_entity,
            "answer_entities": list(paths_info.keys()),
            "paths": paths_info
        }
    
    def test_small_sample(self):
        """Test path extraction on a small sample"""
        print("\n=== Testing on small sample ===")
        
        test_results = []
        
        for query_type in ['1hop_train', '2hop_train', '3hop_train']:
            if query_type in self.qa_data:
                sample_queries = self.qa_data[query_type][:1]  # Just 1 query per type
                
                print(f"\nTesting {query_type}: {len(sample_queries)} queries")
                
                for i, query in enumerate(sample_queries):
                    result = self.extract_paths_for_query(query)
                    if result:
                        test_results.append(result)
                        print(f"  Query {i+1}: Found paths for {len(result['answer_entities'])} answers")
                        
                        # Show path examples
                        for answer, paths in result['paths'].items():
                            print(f"    {answer}: {len(paths)} path(s)")
                            for path in paths[:2]:  # Show first 2 paths
                                print(f"      {path}")
                    else:
                        print(f"  Query {i+1}: No paths found")
        
        # Save test results in compact format
        if test_results:
            with open('test_paths_compact.json', 'w', encoding='utf-8') as f:
                for result in test_results:
                    f.write("{\n")
                    f.write(f'  "type": "{result["type"]}",\n')
                    f.write(f'  "question": "{result["question"]}",\n')
                    f.write(f'  "question_entity": "{result["question_entity"]}",\n')
                    f.write(f'  "answer_entities": {json.dumps(result["answer_entities"])},\n')
                    f.write('  "paths": {\n')
                    
                    path_items = list(result["paths"].items())
                    for j, (answer, paths) in enumerate(path_items):
                        f.write(f'    "{answer}": {json.dumps(paths)}')
                        if j < len(path_items) - 1:
                            f.write(',')
                        f.write('\n')
                    
                    f.write('  }\n')
                    f.write('}\n\n')
            
            print(f"\nSaved {len(test_results)} test results to test_paths_compact.json")
        
        return test_results
    
    def extract_final_dataset(self):
        """Extract final dataset with specified sample sizes"""
        print("\n=== Extracting Final Dataset ===")
        
        # Sample sizes as requested
        sample_config = {
            '1hop_train': 30000,
            '1hop_test': 5000,
            '2hop_train': 30000, 
            '2hop_test': 5000,
            '3hop_train': 10000,
            '3hop_test': 5000
        }
        
        all_results = []
        
        for query_type, sample_size in sample_config.items():
            if query_type not in self.qa_data:
                print(f"Warning: {query_type} not found in data")
                continue
                
            print(f"\nProcessing {query_type}: {sample_size} samples")
            
            # Shuffle and sample
            queries = self.qa_data[query_type].copy()
            random.shuffle(queries)
            sampled_queries = queries[:sample_size]
            
            success_count = 0
            
            for i, query in enumerate(sampled_queries):
                if i % 5000 == 0 and i > 0:
                    print(f"  Processed {i}/{len(sampled_queries)} queries...")
                
                result = self.extract_paths_for_query(query)
                if result:
                    all_results.append(result)
                    success_count += 1
            
            print(f"  Successfully extracted paths for {success_count}/{len(sampled_queries)} queries")
        
        # Save final results
        output_file = 'qa_with_paths.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in all_results:
                f.write("{\n")
                f.write(f'  "type": "{result["type"]}",\n')
                f.write(f'  "question": "{result["question"]}",\n')
                f.write(f'  "question_entity": "{result["question_entity"]}",\n')
                f.write(f'  "answer_entities": {json.dumps(result["answer_entities"])},\n')
                f.write('  "paths": {\n')
                
                path_items = list(result["paths"].items())
                for j, (answer, paths) in enumerate(path_items):
                    f.write(f'    "{answer}": {json.dumps(paths)}')
                    if j < len(path_items) - 1:
                        f.write(',')
                    f.write('\n')
                
                f.write('  }\n')
                f.write('}\n\n')
        
        print(f"\n=== Final Results ===")
        print(f"Total queries with paths: {len(all_results)}")
        print(f"Saved to: {output_file}")
        
        # Summary by type
        type_counts = defaultdict(int)
        for result in all_results:
            type_counts[result['type']] += 1
        
        print("\nBreakdown by type:")
        for query_type, count in sorted(type_counts.items()):
            print(f"  {query_type}: {count}")
        
        return all_results

def main():
    """Main function"""
    extractor = QAPathExtractor()
    
    # Load data
    extractor.load_graph()
    extractor.load_qa_data()
    
    # Test on small sample first
    print("Running test on small sample...")
    test_results = extractor.test_small_sample()
    
    if test_results:
        print("\nTest successful! Ready to proceed with full extraction.")
        
        # Ask user confirmation (for now just proceed)
        print("\nProceeding with full dataset extraction...")
        final_results = extractor.extract_final_dataset()
        
        print("\n=== Complete ===")
        print("Path extraction finished successfully!")
    else:
        print("Test failed - check the setup")

if __name__ == "__main__":
    main()