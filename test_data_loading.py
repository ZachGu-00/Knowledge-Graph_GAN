"""
测试数据加载 - 验证完整数据集是否正确加载
"""

import json
from pathlib import Path

def load_multiline_json(file_path: str):
    """加载多行JSON格式的文件"""
    data = []
    current_obj = ""
    brace_count = 0
    
    print(f"Loading {file_path}...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                current_obj += line + "\n"
                
                # 计算大括号
                brace_count += line.count('{') - line.count('}')
                
                # 当大括号平衡时，说明一个JSON对象结束
                if brace_count == 0 and current_obj.strip():
                    try:
                        obj = json.loads(current_obj.strip())
                        data.append(obj)
                        current_obj = ""
                    except json.JSONDecodeError as e:
                        print(f"JSON parse error at line {line_num}: {e}")
                        print(f"Object preview: {current_obj[:200]}...")
                        current_obj = ""
                        
    except Exception as e:
        print(f"File read error: {e}")
        return []
    
    print(f"Loaded {len(data)} objects")
    return data

def analyze_dataset():
    """分析数据集结构"""
    print("Analyzing Complete Dataset")
    print("="*50)
    
    # 1. 加载QA数据
    qa_file = "query/qa_with_paths_cleaned.json"
    if not Path(qa_file).exists():
        print(f"❌ QA dataset not found: {qa_file}")
        return
    
    all_data = load_multiline_json(qa_file)
    
    if not all_data:
        print("❌ No data loaded")
        return
    
    # 2. 统计类型分布
    type_counts = {}
    for item in all_data:
        type_field = item.get('type', 'unknown')
        type_counts[type_field] = type_counts.get(type_field, 0) + 1
    
    print(f"\nData Type Distribution:")
    for type_field, count in sorted(type_counts.items()):
        print(f"   {type_field}: {count}")
    
    print(f"\nTotal samples: {len(all_data)}")
    
    # 3. 筛选训练数据
    train_data = {
        '1hop': [],
        '2hop': [], 
        '3hop': []
    }
    
    for item in all_data:
        type_field = item.get('type', '')
        if 'train' in type_field:
            if '1hop' in type_field:
                train_data['1hop'].append(item)
            elif '2hop' in type_field:
                train_data['2hop'].append(item)
            elif '3hop' in type_field:
                train_data['3hop'].append(item)
    
    print(f"\nTraining Data Distribution:")
    for hop_type, data in train_data.items():
        print(f"   {hop_type}: {len(data)} samples")
    
    total_train = sum(len(data) for data in train_data.values())
    print(f"   Total training: {total_train}")
    
    # 4. 显示样本示例
    print(f"\nSample Examples:")
    for hop_type, data in train_data.items():
        if data:
            sample = data[0]
            print(f"\n{hop_type} example:")
            print(f"   Question: {sample['question']}")
            print(f"   Start entity: {sample['question_entity']}")
            print(f"   Answer entities: {sample['answer_entities'][:2]}...")  # 只显示前2个
            if sample['paths']:
                first_answer = list(sample['paths'].keys())[0]
                first_path = sample['paths'][first_answer][0]
                print(f"   Sample path: {first_path}")
    
    # 5. 检查其他必要文件
    print(f"\nChecking Required Files:")
    required_files = {
        "graph/knowledge_graph.pkl": "Knowledge Graph",
        "graph/entity_names.json": "Entity Names",
        "embeddings/entity_embeddings.pt": "Entity Embeddings"
    }
    
    for file_path, description in required_files.items():
        if Path(file_path).exists():
            print(f"   OK {description}: {file_path}")
        else:
            print(f"   MISSING {description}: {file_path}")

if __name__ == "__main__":
    analyze_dataset()