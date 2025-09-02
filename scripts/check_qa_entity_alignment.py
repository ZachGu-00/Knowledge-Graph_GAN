import json

def check_qa_entity_alignment():
    """Check alignment between QA entities and graph entities"""
    
    # Load graph entities (normalized)
    with open('graph/entity_names.json', 'r', encoding='utf-8') as f:
        graph_entities = set(json.load(f))
    
    print(f"Graph entities: {len(graph_entities)}")
    
    # Create case-insensitive mapping
    graph_entities_lower = {entity.lower(): entity for entity in graph_entities}
    
    # Check QA entities
    qa_question_entities = set()
    qa_answer_entities = set()
    
    missing_question_entities = set()
    missing_answer_entities = set()
    
    print("Checking QA entities...")
    
    with open('query/extracted_qa.json', 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 50000 == 0:
                print(f"  Processed {line_num} QA entries...")
                
            query = json.loads(line.strip())
            
            # Question entity
            q_entity = query['question_entity']
            qa_question_entities.add(q_entity)
            
            if q_entity not in graph_entities:
                if q_entity.lower() in graph_entities_lower:
                    # Case mismatch found
                    pass
                else:
                    missing_question_entities.add(q_entity)
            
            # Answer entities
            for answer in query['answer_entities']:
                qa_answer_entities.add(answer)
                
                if answer not in graph_entities:
                    if answer.lower() in graph_entities_lower:
                        # Case mismatch found
                        pass
                    else:
                        missing_answer_entities.add(answer)
    
    print(f"\n=== ALIGNMENT REPORT ===")
    print(f"QA question entities: {len(qa_question_entities)}")
    print(f"QA answer entities: {len(qa_answer_entities)}")
    print(f"Total unique QA entities: {len(qa_question_entities | qa_answer_entities)}")
    
    print(f"\nMissing question entities: {len(missing_question_entities)}")
    print(f"Missing answer entities: {len(missing_answer_entities)}")
    
    # Check case mismatches
    q_case_mismatches = 0
    a_case_mismatches = 0
    
    for q_entity in qa_question_entities:
        if q_entity not in graph_entities and q_entity.lower() in graph_entities_lower:
            q_case_mismatches += 1
    
    for answer in qa_answer_entities:
        if answer not in graph_entities and answer.lower() in graph_entities_lower:
            a_case_mismatches += 1
    
    print(f"\nCase mismatches:")
    print(f"Question entities: {q_case_mismatches}")
    print(f"Answer entities: {a_case_mismatches}")
    print(f"Total case mismatches: {q_case_mismatches + a_case_mismatches}")
    
    # Show examples
    if missing_question_entities:
        print(f"\nSample missing question entities:")
        for entity in sorted(missing_question_entities)[:10]:
            print(f"  '{entity}'")
    
    if missing_answer_entities:
        print(f"\nSample missing answer entities:")
        for entity in sorted(missing_answer_entities)[:10]:
            print(f"  '{entity}'")
    
    # Show case mismatch examples
    print(f"\nSample case mismatches:")
    count = 0
    for q_entity in sorted(qa_question_entities):
        if q_entity not in graph_entities and q_entity.lower() in graph_entities_lower:
            correct_case = graph_entities_lower[q_entity.lower()]
            print(f"  QA: '{q_entity}' -> Graph: '{correct_case}'")
            count += 1
            if count >= 5:
                break
    
    for answer in sorted(qa_answer_entities):
        if count >= 10:
            break
        if answer not in graph_entities and answer.lower() in graph_entities_lower:
            correct_case = graph_entities_lower[answer.lower()]
            print(f"  QA: '{answer}' -> Graph: '{correct_case}'")
            count += 1
    
    return {
        'qa_entities_total': len(qa_question_entities | qa_answer_entities),
        'missing_total': len(missing_question_entities) + len(missing_answer_entities),
        'case_mismatches': q_case_mismatches + a_case_mismatches,
        'needs_normalization': (q_case_mismatches + a_case_mismatches) > 0
    }

if __name__ == "__main__":
    result = check_qa_entity_alignment()
    print(f"\n=== SUMMARY ===")
    print(f"Needs entity normalization: {result['needs_normalization']}")