import json
import pandas as pd
import numpy as np
import pickle
from pprint import pprint
import glob
from tqdm import tqdm
from pathlib import Path
from pyserini.search.lucene import LuceneSearcher
import subprocess
import os

WIKIS_PATH = "../wiki-pages/wiki-*.jsonl"
# Create JSONL for Pyserini
JSONL_DIR = "wiki_jsonl_files"
JSONL_FILE = "wiki_for_pyserini.jsonl"
PYSERINI_DIR = "pyserini_wiki_index"
TRAIN_FILE = 'data/fever-data/dev.jsonl'#'../train.jsonl'
OUTPUT = 'dev_claim_retrieved_docs_bm25_top5.json'
TOP_K = 5

def build_lucern_idx(documents, doc_ids):
    
    print("Building Pyserini index from documents...")
    # Create directory for JSONL
    os.makedirs(JSONL_DIR, exist_ok=True)
    jsonl_path = os.path.join(JSONL_DIR, JSONL_FILE)
    
    print("Step 1: Creating JSONL file...")
    with open(jsonl_path, 'w') as f:
        for doc_id, doc_text in tqdm(zip(doc_ids, documents), total=len(documents)):
            json.dump({"id": str(doc_id), "contents": doc_text.lower()}, f, ensure_ascii=False)
            f.write('\n')
    print(f"{jsonl_path} generated")
        
    print("Step 2: Building Lucene index (this may take 10-30 minutes)...")    
    
    # Use command-line indexing (most reliable method)
    print("Building index using Pyserini command-line tool...")
    cmd = [
        "python", "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", JSONL_DIR, #THIS MUST BE A FOLDER
        "--index", PYSERINI_DIR,
        "--generator", "DefaultLuceneDocumentGenerator",
        # "--analyzer", "WhitespaceAnalyzer"  # <-- The crucial part
        "--threads", "4",  # Use multiple threads for faster indexing
        "--storePositions", "--storeDocvectors", "--storeRaw"
        
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(f"Pyserini index created at: {PYSERINI_DIR}")
    if result.stdout:
        print("Indexing output:", result.stdout[-500:])  # Last 500 chars

def retrieve_docs_pyserini(claim, searcher, top_k=TOP_K):
    """
    Retrieve top-k documents using Pyserini by correctly parsing the raw JSON
    stored in the Lucene index.
    # Result might be slightly diff but should be more reliable: https://gemini.google.com/u/1/app/1ea3903ad504b413?pageId=none
    """
    # Lowercase the claim to match the lowercased documents in the index
    hits = searcher.search(claim.lower(), k=top_k)
    results = []
    
    for i, hit in enumerate(hits):
        doc = searcher.doc(hit.docid)
        if not doc:
            continue # Skip if doc not found

        # 1. Get the raw JSON string from the document
        raw_json_string = doc.raw()
        if not raw_json_string:
            continue # Skip if no raw content

        # 2. Parse the JSON string into a Python dictionary
        doc_data = json.loads(raw_json_string)
        
        # 3. Access the fields from the dictionary
        doc_text = doc_data.get('contents')
        doc_id = doc_data.get('id')
        
        if doc_text and doc_id:
            results.append({
                'doc_id': doc_id,
                'text': doc_text,
                'score': float(hit.score)
            })
            
    return results    

if __name__ == '__main__':
    # Load Wikipedia data
    wiki_files = sorted(glob.glob(WIKIS_PATH))
    print(f"Found {len(wiki_files)} Wikipedia files")
    # Load all Wikipedia articles
    wiki_data = []
    for file in tqdm(wiki_files):
        with open(file, 'r') as f:
            for line in f:
                wiki_data.append(json.loads(line))
    
    print(f"Loaded {len(wiki_data)} Wikipedia articles")
    # Filter out empty articles and prepare documents
    documents = []
    doc_ids = []
    
    for article in tqdm(wiki_data):
        if article['id'] and article['text']:
            documents.append(article['text'])
            doc_ids.append(article['id'])
    
    print(f"Total documents: {len(documents)}")
    build_lucern_idx(documents, doc_ids)

    # Processing claims
    print("Start extracting most relevant wiki docs...")
    f = open(TRAIN_FILE,'r')
    l = f.readlines()
    res =[ ]
    for l_ in l:
        res.append(json.loads(l_))
    f.close()
    df = pd.DataFrame(res)
    claims = df['claim'].tolist()
    print(f"Total claims: {len(claims)}")

    searcher = LuceneSearcher(PYSERINI_DIR)
    searcher.set_bm25(k1=1.5, b=0.75) # Use the same BM25 params
    claim_docs_pyserini = []
    for i, claim in enumerate(tqdm(claims)):
        retrieved_docs = retrieve_docs_pyserini(claim, searcher, top_k=TOP_K)
        claim_docs_pyserini.append({
            'claim_id': i,
            'claim': claim,
            'retrieved_documents': retrieved_docs
        })
    
    print(f"Retrieved documents for {len(claim_docs_pyserini)} claims using Pyserini")
    # Save results to JSON
    with open(OUTPUT, 'w') as f:
        json.dump(claim_docs_pyserini, f, indent=2)
    
    print(f"Results saved to {OUTPUT}")
    print(f"Sample entry:\n{json.dumps(claim_docs_pyserini[0], indent=2)}")
        
    # Summary
    print("\n=== Summary ===")
    print(f"Total Wikipedia articles: {len(wiki_data)}")
    print(f"Indexed documents: {len(documents)}")
    print(f"Claims processed: {len(claims)}")
    print(f"Output file: {OUTPUT}")


# nohup python build_and_search_with_lucern_idx.py >> dev_logs/log_bm25_dev.log 2>&1&
