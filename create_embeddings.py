import torch
from sentence_transformers import SentenceTransformer
import json
from pathlib import Path
from tqdm import tqdm

def create_entity_embeddings():
    """创建统一的实体embedding文件"""
    
    print("Creating unified entity embeddings...")
    
    # 检查输入文件
    entity_names_file = "graph/entity_names.json"
    output_path = "embeddings/entity_embeddings.pt"
    
    if not Path(entity_names_file).exists():
        print(f"Error: {entity_names_file} not found!")
        return
    
    # 创建输出目录
    Path("embeddings").mkdir(exist_ok=True)
    
    # 加载实体名称
    print(f"Loading entity names from {entity_names_file}")
    with open(entity_names_file, 'r', encoding='utf-8') as f:
        entity_names = json.load(f)
    
    print(f"Found {len(entity_names)} entities")
    
    # 初始化SBERT
    print("Initializing SBERT model...")
    sbert_model_name = 'all-MiniLM-L6-v2'
    sbert = SentenceTransformer(sbert_model_name)
    
    # 批量编码
    batch_size = 256
    entity_embeddings = {}
    
    print("Encoding entities in batches...")
    
    for i in tqdm(range(0, len(entity_names), batch_size), desc="Encoding"):
        batch_entities = entity_names[i:i+batch_size]
        
        # 编码当前批次
        batch_embeds = sbert.encode(
            batch_entities, 
            convert_to_tensor=True,
            show_progress_bar=False
        )
        
        # 存储embedding
        for entity, embed in zip(batch_entities, batch_embeds):
            entity_embeddings[entity] = embed.cpu()
    
    # 保存embeddings
    print(f"Saving embeddings to {output_path}")
    torch.save(entity_embeddings, output_path)
    
    # 验证
    loaded_embeddings = torch.load(output_path)
    print(f"[OK] Successfully created embeddings for {len(loaded_embeddings)} entities")
    print(f"[OK] Saved to: {output_path}")
    print(f"[OK] File size: {Path(output_path).stat().st_size / (1024*1024):.1f} MB")
    
    # 显示一些示例
    print(f"\nSample entities:")
    sample_entities = list(loaded_embeddings.keys())[:5]
    for entity in sample_entities:
        embed_shape = loaded_embeddings[entity].shape
        print(f"  '{entity}': {embed_shape}")
    
    return output_path

if __name__ == "__main__":
    create_entity_embeddings()