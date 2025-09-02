import json
from collections import defaultdict

def analyze_and_clean_paths():
    """分析路径长度分布并清理数据"""
    
    print("Loading qa_with_paths.json...")
    
    valid_queries = []
    hop_analysis = defaultdict(lambda: defaultdict(int))
    type_counts = defaultdict(int)
    
    with open('qa_with_paths.json', 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Split by double newlines to get individual queries
    query_blocks = content.strip().split('\n\n')
    
    print(f"Found {len(query_blocks)} query blocks")
    
    for i, block in enumerate(query_blocks):
        if not block.strip():
            continue
            
        try:
            query = json.loads(block)
            
            # Check if query has paths
            if not query.get('paths') or len(query['paths']) == 0:
                continue  # Skip queries without paths
            
            query_type = query['type']
            type_counts[query_type] += 1
            
            # Analyze path lengths for this query
            path_lengths = []
            
            for answer_entity, paths in query['paths'].items():
                for path_string in paths:
                    # Count dots to determine path length
                    # Format: entity.relation.entity.relation.entity
                    # 1-hop: A.rel.B (1 relation, 2 entities)
                    # 2-hop: A.rel.B.rel.C (2 relations, 3 entities)
                    parts = path_string.split('.')
                    num_relations = (len(parts) - 1) // 2
                    path_lengths.append(num_relations)
            
            if path_lengths:
                # Record path length distribution for this query type
                for length in set(path_lengths):  # Unique lengths in this query
                    hop_analysis[query_type][length] += 1
                
                valid_queries.append(query)
        
        except json.JSONDecodeError as e:
            print(f"JSON decode error in block {i}: {e}")
            continue
    
    print(f"\n=== Analysis Results ===")
    print(f"Total valid queries: {len(valid_queries)}")
    
    print(f"\nQueries by type:")
    for query_type, count in sorted(type_counts.items()):
        print(f"  {query_type}: {count}")
    
    print(f"\n=== Path Length Analysis ===")
    
    for query_type in sorted(hop_analysis.keys()):
        print(f"\n{query_type}:")
        total_queries = type_counts[query_type]
        
        length_counts = hop_analysis[query_type]
        
        # Expected hop length based on query type
        expected_hops = int(query_type[0])  # Get first character (1, 2, or 3)
        
        correct_hop_count = length_counts.get(expected_hops, 0)
        correct_percentage = (correct_hop_count / total_queries) * 100 if total_queries > 0 else 0
        
        print(f"  Total queries: {total_queries}")
        print(f"  Expected {expected_hops}-hop paths: {correct_hop_count} ({correct_percentage:.1f}%)")
        
        print(f"  Path length distribution:")
        for length in sorted(length_counts.keys()):
            count = length_counts[length]
            percentage = (count / total_queries) * 100
            marker = " [CORRECT]" if length == expected_hops else ""
            print(f"    {length}-hop: {count} ({percentage:.1f}%){marker}")
    
    # Save cleaned data
    output_file = 'qa_with_paths_cleaned.json'
    print(f"\nSaving cleaned data to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for query in valid_queries:
            f.write("{\n")
            f.write(f'  "type": "{query["type"]}",\n')
            f.write(f'  "question": "{query["question"]}",\n')
            f.write(f'  "question_entity": "{query["question_entity"]}",\n')
            f.write(f'  "answer_entities": {json.dumps(query["answer_entities"])},\n')
            f.write('  "paths": {\n')
            
            path_items = list(query["paths"].items())
            for j, (answer, paths) in enumerate(path_items):
                f.write(f'    "{answer}": {json.dumps(paths)}')
                if j < len(path_items) - 1:
                    f.write(',')
                f.write('\n')
            
            f.write('  }\n')
            f.write('}\n\n')
    
    # Summary statistics
    print(f"\n=== Final Summary ===")
    print(f"Cleaned queries saved: {len(valid_queries)}")
    
    total_original = sum(type_counts.values())
    print(f"Success rate: {len(valid_queries)}/{total_original} queries had valid paths")
    
    # Overall hop accuracy
    print(f"\n=== Hop Accuracy Summary ===")
    for query_type in ['1hop_train', '1hop_test', '2hop_train', '2hop_test', '3hop_train', '3hop_test']:
        if query_type in hop_analysis:
            expected_hops = int(query_type[0])
            total = type_counts[query_type]
            correct = hop_analysis[query_type].get(expected_hops, 0)
            accuracy = (correct / total) * 100 if total > 0 else 0
            print(f"  {query_type}: {correct}/{total} ({accuracy:.1f}%) correct hop length")
    
    return valid_queries, hop_analysis, type_counts

def show_path_examples():
    """Show examples of different path lengths"""
    print("\n=== Path Examples ===")
    
    examples_found = defaultdict(list)
    
    with open('qa_with_paths_cleaned.json', 'r', encoding='utf-8') as f:
        content = f.read()
    
    query_blocks = content.strip().split('\n\n')
    
    for block in query_blocks[:100]:  # Check first 100 queries
        if not block.strip():
            continue
            
        try:
            query = json.loads(block)
            query_type = query['type']
            
            for answer_entity, paths in query['paths'].items():
                for path_string in paths:
                    parts = path_string.split('.')
                    num_relations = (len(parts) - 1) // 2
                    
                    if len(examples_found[num_relations]) < 2:  # Keep 2 examples per hop length
                        examples_found[num_relations].append({
                            'type': query_type,
                            'question': query['question'],
                            'path': path_string
                        })
        except json.JSONDecodeError:
            continue
    
    for hop_length in sorted(examples_found.keys()):
        print(f"\n{hop_length}-hop examples:")
        for example in examples_found[hop_length]:
            print(f"  Type: {example['type']}")
            print(f"  Question: {example['question']}")
            print(f"  Path: {example['path']}")
            print()

if __name__ == "__main__":
    valid_queries, hop_analysis, type_counts = analyze_and_clean_paths()
    show_path_examples()