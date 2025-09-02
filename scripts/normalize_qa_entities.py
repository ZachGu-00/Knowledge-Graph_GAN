import json

def normalize_qa_entities():
    """Normalize QA entities to match graph entity names"""
    
    # Load graph entities (normalized)
    with open('graph/entity_names.json', 'r', encoding='utf-8') as f:
        graph_entities = json.load(f)
    
    # Create case-insensitive mapping from graph entities
    entity_mapping = {entity.lower(): entity for entity in graph_entities}
    
    print(f"Loaded {len(graph_entities)} graph entities")
    print(f"Created case mapping for {len(entity_mapping)} entities")
    
    # Read and normalize QA data
    normalized_queries = []
    normalization_count = 0
    
    print("Normalizing QA entities...")
    
    with open('query/extracted_qa.json', 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 50000 == 0:
                print(f"  Processed {line_num} queries...")
            
            query = json.loads(line.strip())
            
            # Normalize question entity
            original_q_entity = query['question_entity']
            normalized_q_entity = entity_mapping.get(original_q_entity.lower(), original_q_entity)
            
            if normalized_q_entity != original_q_entity:
                normalization_count += 1
            
            # Normalize answer entities
            normalized_answers = []
            for answer in query['answer_entities']:
                normalized_answer = entity_mapping.get(answer.lower(), answer)
                normalized_answers.append(normalized_answer)
                
                if normalized_answer != answer:
                    normalization_count += 1
            
            # Create normalized query
            normalized_query = {
                'type': query['type'],
                'question': query['question'].replace(f'[{original_q_entity}]', f'[{normalized_q_entity}]'),
                'question_entity': normalized_q_entity,
                'answer_entities': normalized_answers
            }
            
            normalized_queries.append(normalized_query)
    
    print(f"Applied {normalization_count} entity normalizations")
    
    # Save normalized QA data
    output_file = 'extracted_qa_normalized.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        for query in normalized_queries:
            json.dump(query, f, ensure_ascii=False, separators=(',', ':'))
            f.write('\n')
    
    print(f"Saved normalized QA data to {output_file}")
    
    # Verification check
    print("\nRunning verification check...")
    
    with open('graph/entity_names.json', 'r', encoding='utf-8') as f:
        graph_entities_set = set(json.load(f))
    
    unmatched_entities = set()
    
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            query = json.loads(line.strip())
            
            # Check question entity
            if query['question_entity'] not in graph_entities_set:
                unmatched_entities.add(query['question_entity'])
            
            # Check answer entities
            for answer in query['answer_entities']:
                if answer not in graph_entities_set:
                    unmatched_entities.add(answer)
    
    print(f"Unmatched entities after normalization: {len(unmatched_entities)}")
    
    if unmatched_entities:
        print("Sample unmatched entities:")
        for entity in sorted(unmatched_entities)[:10]:
            print(f"  '{entity}'")
    else:
        print("All entities now match the graph!")
    
    return {
        'total_normalizations': normalization_count,
        'unmatched_after': len(unmatched_entities),
        'success': len(unmatched_entities) == 0
    }

if __name__ == "__main__":
    result = normalize_qa_entities()
    
    print(f"\n=== NORMALIZATION SUMMARY ===")
    print(f"Total normalizations applied: {result['total_normalizations']}")
    print(f"Unmatched entities remaining: {result['unmatched_after']}")
    print(f"Normalization successful: {result['success']}")