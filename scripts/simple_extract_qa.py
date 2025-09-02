import json
import os
import re

def extract_qa_from_file(filepath, hop_type, split_type):
    """Extract Q&A data from a single file"""
    qa_data = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                # Remove line number prefix (e.g., "1→")
                if '→' in line:
                    line = line.split('→', 1)[1]
                
                # Split question and answers
                if '\t' in line:
                    question, answers = line.split('\t', 1)
                else:
                    continue
                
                # Extract question entity (text between [ and ])
                question_entity_match = re.search(r'\[([^\]]+)\]', question)
                if question_entity_match:
                    question_entity = question_entity_match.group(1)
                else:
                    continue
                
                # Split multiple answers by |
                answer_entities = []
                if answers:
                    answer_entities = [answer.strip() for answer in answers.split('|') if answer.strip()]
                
                # Create query data
                query_data = {
                    "type": f"{hop_type}_{split_type}",
                    "question": question,
                    "question_entity": question_entity,
                    "answer_entities": answer_entities
                }
                
                qa_data.append(query_data)
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
    
    return qa_data

def main():
    all_qa_data = []
    
    # Process all hop directories and files
    hop_dirs = ['1hop', '2hop', '3hop']
    split_types = ['train', 'test']
    
    for hop_dir in hop_dirs:
        for split_type in split_types:
            filepath = f'query/{hop_dir}/qa_{split_type}.txt'
            
            if os.path.exists(filepath):
                print(f"Processing {filepath}...")
                qa_data = extract_qa_from_file(filepath, hop_dir, split_type)
                all_qa_data.extend(qa_data)
                print(f"  Added {len(qa_data)} queries")
            else:
                print(f"Warning: {filepath} not found")
    
    # Save to JSON file with compact format (one query per line)
    output_file = 'extracted_qa.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        for query in all_qa_data:
            json.dump(query, f, ensure_ascii=False, separators=(',', ':'))
            f.write('\n')
    
    print(f"\nExtracted {len(all_qa_data)} queries to {output_file}")
    
    # Print summary statistics
    type_counts = {}
    for query in all_qa_data:
        query_type = query['type']
        type_counts[query_type] = type_counts.get(query_type, 0) + 1
    
    print("\nSummary by type:")
    for query_type, count in sorted(type_counts.items()):
        print(f"  {query_type}: {count} queries")

if __name__ == "__main__":
    main()