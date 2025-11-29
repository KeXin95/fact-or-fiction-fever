from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import glob
import json
import pandas as pd
import torch 
import math
import numpy as np

from huggingface_hub import login
import os

login(token="")
os.environ["HF_TOKEN"] = ''
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4,5,6,7'

# Configuration
WIKIS_PATH = "../wiki-pages/wiki-*.jsonl"
FAISS_INDEX_DIR = "faiss_wiki_index"
TRAIN_FILE = 'data/fever-data/dev.jsonl' #'../dev.jsonl'
OUTPUT = 'dev_claim_retrieved_docs_dense_qwen3_top5.json'
BATCH_SIZE = 4          # Reduced to prevent OOM
MAX_SEQ_LENGTH = 1024   # Limit text length to save memory
SENTENCE_TRANSFORMER_MODEL = "Qwen/Qwen3-Embedding-0.6B"
top_k = 5

# Helper function to encode in chunks
def encode_with_progress(data, pool, batch_size, desc="Encoding"):
    chunk_size = 10000  # Adjust based on your RAM (10k is usually safe)
    steps = math.ceil(len(data) / chunk_size)
    
    all_embeddings = []
    
    # Iterate with tqdm
    for i in tqdm(range(steps), desc=desc):
        # Slice the data
        start = i * chunk_size
        end = min((i + 1) * chunk_size, len(data))
        batch_text = data[start:end]
        
        # Encode this chunk using all GPUs
        # Note: encode_multi_process returns a numpy array
        batch_emb = model.encode_multi_process(
            batch_text, 
            pool, 
            batch_size=batch_size
        )
        all_embeddings.append(batch_emb)

    # Combine all chunks back into one big matrix
    return np.vstack(all_embeddings)
    
if __name__ == '__main__':
    
    # 1. Load Model
    print("Loading model...")
    model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL, trust_remote_code=True)
    model.max_seq_length = MAX_SEQ_LENGTH
    
    # 2. Load Documents
    wiki_files = sorted(glob.glob(WIKIS_PATH))
    print(f"Found {len(wiki_files)} Wikipedia files")

    documents = []
    doc_ids = []
    
    print("Reading Wikipedia files...")
    for file in tqdm(wiki_files):
        with open(file, 'r') as f:
            for line in f:
                article = json.loads(line)
                if article['id'] and article['text']:
                    documents.append(article['text'])
                    doc_ids.append(article['id'])
    
    print(f"Total documents: {len(documents)}")

    # 3. Load Claims
    print("Reading Claims...")
    f = open(TRAIN_FILE, 'r')
    l = f.readlines()
    res = []
    for l_ in l:
        res.append(json.loads(l_))
    f.close()
    
    df = pd.DataFrame(res)
    claims = df['claim'].tolist()
    print(f"Total claims: {len(claims)}")
    
    # 4. Encode using Multi-GPU Pool (Fixes OOM & Speed)
    print("Starting Multi-GPU Encoding...")
    
    pool = model.start_multi_process_pool()
    
    # --- USAGE ---
    # documents = documents[:10]
    # claims = claims[:10]
    print(f"Encoding {len(documents)} documents...")
    document_embeddings = encode_with_progress(
        documents, 
        pool, 
        BATCH_SIZE, 
        desc="Docs"
    )
    
    print(f"Encoding {len(claims)} claims...")
    # Note: encode_multi_process doesn't support 'prompt_name' in older versions.
    # If you need the prompt, you might need to manually add it to the string:
    # claims_with_prompt = [f"query: {c}" for c in claims]
    query_embeddings = encode_with_progress(
        claims, 
        pool, 
        BATCH_SIZE, 
        desc="Claims"
    )
    
    model.stop_multi_process_pool(pool)

    # 5. Semantic Search (Chunked to save RAM)
    print(f"Calculating Similarity & Retrieving Top {top_k}...")
    
    # We use util.semantic_search instead of manual torch.topk
    # This automatically handles batching to prevent RAM crash
    from sentence_transformers import util
    
    hits = util.semantic_search(
        query_embeddings, 
        document_embeddings, 
        top_k=top_k,
        query_chunk_size=100,  # Process 100 queries at a time
        corpus_chunk_size=50000 # Search against 50k docs at a time
    )

    # 6. Format Results
    claim_docs_dense = []
    
    for query_idx, hit_list in tqdm(enumerate(hits), total=len(hits), desc="Formatting"):
        claim_text = claims[query_idx]
        
        retrieved_docs = []
        for hit in hit_list:
            doc_idx = hit['corpus_id']
            score = hit['score']
            
            retrieved_docs.append({
                'doc_id': doc_ids[doc_idx],
                'text': documents[doc_idx],
                'score': score
            })

        claim_docs_dense.append({
            'claim_id': query_idx,
            'claim': claim_text,
            'retrieved_documents': retrieved_docs
        })

    # 7. Save
    print(f"Saving to {OUTPUT}...")
    with open(OUTPUT, 'w') as f:
        json.dump(claim_docs_dense, f, indent=2)
    
    
    print(f"Results saved to {OUTPUT}")
    print(f"Sample entry:\n{json.dumps(claim_docs_dense[0], indent=2)}")
    
    # Summary
    print("\n=== Summary ===")
    print(f"Model used: {SENTENCE_TRANSFORMER_MODEL}")
    print(f"Claims processed: {len(claims)}")
    print(f"Output file: {OUTPUT}")
    print("="*50)

# nohup python dense_qwen.py >> dev_logs/log_dense_qwen_dev_top5.log 2>&1&
