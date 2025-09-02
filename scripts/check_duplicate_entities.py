import json
from collections import Counter, defaultdict

def check_duplicates():
    """检查实体重复情况"""
    
    # 1. 从kb_entity_dict.txt加载所有实体
    entities = []
    with open('data/kb_entity_dict.txt', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if '\t' in line:
                parts = line.split('\t')
                if len(parts) >= 2:
                    entity = parts[1]
                    entities.append(entity)
    
    print(f"Loaded {len(entities)} entities from kb_entity_dict.txt")
    
    # 2. Check exact duplicates
    entity_counts = Counter(entities)
    exact_duplicates = {entity: count for entity, count in entity_counts.items() if count > 1}
    
    print(f"Exact duplicates: {len(exact_duplicates)}")
    if exact_duplicates:
        print("Top 10 duplicate entities:")
        for entity, count in sorted(exact_duplicates.items(), key=lambda x: -x[1])[:10]:
            print(f"  '{entity}': {count} times")
    
    # 3. Check case differences
    case_groups = defaultdict(list)
    for entity in set(entities):  # Remove duplicates first
        key = entity.lower()
        case_groups[key].append(entity)
    
    case_duplicates = {key: values for key, values in case_groups.items() if len(values) > 1}
    
    print(f"\nCase-different entity groups: {len(case_duplicates)}")
    if case_duplicates:
        print("Top 10 case duplicate groups:")
        for i, (key, variants) in enumerate(sorted(case_duplicates.items())[:10]):
            print(f"  '{key}': {variants}")
    
    # 4. Calculate unique counts
    unique_entities_exact = len(set(entities))
    unique_entities_case_insensitive = len(set(entity.lower() for entity in entities))
    
    print(f"\n=== SUMMARY ===")
    print(f"Original entity count: {len(entities)}")
    print(f"After removing exact duplicates: {unique_entities_exact}")
    print(f"After removing case duplicates: {unique_entities_case_insensitive}")
    print(f"Exact duplicate loss: {len(entities) - unique_entities_exact}")
    print(f"Case duplicate loss: {unique_entities_exact - unique_entities_case_insensitive}")
    
    # 5. Check entities in kb.txt vs dictionary
    kb_entities = set()
    
    with open('data/kb.txt', 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 50000 == 0:
                print(f"Processing kb.txt line {line_num}...")
            line = line.strip()
            if '|' in line:
                parts = line.split('|')
                if len(parts) == 3:
                    subject, relation, obj = parts
                    kb_entities.add(subject)
                    kb_entities.add(obj)
    
    # Check which entities in kb.txt are missing from dictionary
    entity_dict_set = set(entities)
    missing_entities = kb_entities - entity_dict_set
    
    print(f"\nEntities in kb.txt: {len(kb_entities)}")
    print(f"Missing from dictionary: {len(missing_entities)}")
    
    if missing_entities:
        print("Top 10 missing entities:")
        for entity in sorted(missing_entities)[:10]:
            print(f"  '{entity}'")

if __name__ == "__main__":
    check_duplicates()