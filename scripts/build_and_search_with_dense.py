import json
import pandas as pd
import numpy as np
import pickle
from pprint import pprint
import glob
from tqdm import tqdm
from pathlib import Path
import os
import faiss
from sentence_transformers import SentenceTransformer
import torch

from huggingface_hub import login
import os

login(token="")
os.environ["HF_TOKEN"] = ''
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4,5'

# Configuration
WIKIS_PATH = "../../wiki-pages/wiki-*.jsonl"
FAISS_INDEX_DIR = "faiss_wiki_index"
TRAIN_FILE = 'data/fever-data/dev.jsonl' #'../train.jsonl'
OUTPUT = 'dev_claim_retrieved_docs_dense_miniLM_top5.json'
TOP_K = 5

# Model configuration
# Using 'all-mpnet-base-v2' for better quality, or 'all-MiniLM-L6-v2' for faster/smaller
SENTENCE_TRANSFORMER_MODEL = 'all-MiniLM-L6-v2'

# FAISS configuration
FAISS_METRIC = faiss.METRIC_INNER_PRODUCT  # For cosine similarity (after normalization)
# Alternative: faiss.METRIC_L2  # For L2 distance

# GPU configuration
USE_GPU = True  # Set to False to force CPU usage
GPU_ID = 0  # Which GPU to use (if multiple GPUs available)
BATCH_SIZE = 2048  # Increased for better GPU utilization (adjust based on GPU memory)

def cleanup_gpu_memory():
    """Clean up GPU memory by clearing PyTorch cache and running garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        import gc
        gc.collect()
        print("GPU memory cleaned up")

def get_faiss_gpu_resources():
    """Get FAISS GPU resources if available."""
    try:
        if faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            return res
        else:
            return None
    except:
        return None

def build_faiss_index(documents, doc_ids, model, batch_size=BATCH_SIZE):
    """
    Build a FAISS index from documents using sentence transformers.
    
    Args:
        documents: List of document texts
        doc_ids: List of document IDs
        model: SentenceTransformer model
        batch_size: Batch size for encoding
    
    Returns:
        faiss_index: FAISS index
        id_to_doc: Dictionary mapping index position to (doc_id, doc_text)
    """
    print(f"Building FAISS index with model: {SENTENCE_TRANSFORMER_MODEL}")
    print(f"Total documents: {len(documents)}")
    
    # Create output directory
    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
    
    # Step 1: Encode all documents
    print("Step 1: Encoding documents with SentenceTransformer...")
    print(f"This may take 30-120 minutes for {len(documents):,} documents.")
    print("Progress bar will show below. The process is working, please be patient...")
    
    # SentenceTransformers automatically uses GPU if available
    if torch.cuda.is_available():
        print(f"GPU detected: Using CUDA for faster encoding")
    else:
        print("No GPU detected: Using CPU (this will be slower)")
    
    embeddings = model.encode(
        documents,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # Normalize for cosine similarity
    )
    
    print(f"Embeddings shape: {embeddings.shape}")
    dimension = embeddings.shape[1]
    print(f"Embedding dimension: {dimension}")
    
    # Clean up GPU memory after encoding (embeddings are now in CPU memory as numpy array)
    cleanup_gpu_memory()
    
    # Step 2: Create FAISS index
    print("Step 2: Building FAISS index...")
    # Check GPU availability
    gpu_res = get_faiss_gpu_resources()
    use_gpu_for_index = USE_GPU and (gpu_res is not None)
    
    if use_gpu_for_index:
        print(f"Using GPU for FAISS operations (GPU {GPU_ID})")
        print(f"Available GPUs: {faiss.get_num_gpus()}")
    else:
        print("Using CPU for FAISS operations")
    
    # Create index (will be on CPU first, then moved to GPU if needed)
    if FAISS_METRIC == faiss.METRIC_INNER_PRODUCT:
        index_cpu = faiss.IndexFlatIP(dimension)
    else:
        index_cpu = faiss.IndexFlatL2(dimension)
    
    # Add embeddings to CPU index first
    embeddings_f32 = embeddings.astype('float32')
    index_cpu.add(embeddings_f32)
    print(f"Index size: {index_cpu.ntotal}")
    
    # Move to GPU if available
    # Note: Large indexes may not fit in GPU memory
    # For 5.4M documents with 384-dim embeddings, index is ~8.3GB
    if use_gpu_for_index:
        try:
            # Clean up GPU memory first
            cleanup_gpu_memory()
            
            # Check GPU memory before moving
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(GPU_ID).total_memory / 1e9
                gpu_allocated = torch.cuda.memory_allocated(GPU_ID) / 1e9
                gpu_free = gpu_memory - gpu_allocated
                index_size_gb = (index_cpu.ntotal * dimension * 4) / 1e9  # float32 = 4 bytes
                
                print(f"GPU {GPU_ID} total memory: {gpu_memory:.2f} GB")
                print(f"GPU {GPU_ID} allocated: {gpu_allocated:.2f} GB")
                print(f"GPU {GPU_ID} free: {gpu_free:.2f} GB")
                print(f"Estimated index size: {index_size_gb:.2f} GB")
                
                if index_size_gb > gpu_free * 0.9:  # Leave 10% buffer
                    print(f"WARNING: Index size ({index_size_gb:.2f} GB) is too large for available GPU memory ({gpu_free:.2f} GB)")
                    print("Keeping index on CPU. Search will be slower but will work.")
                    index = index_cpu
                else:
                    print("Moving index to GPU...")
                    index = faiss.index_cpu_to_gpu(gpu_res, GPU_ID, index_cpu)
                    print("Index moved to GPU successfully")
            else:
                index = index_cpu
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "alloc fail" in str(e).lower():
                print(f"WARNING: GPU out of memory when moving index to GPU: {e}")
                print("Keeping index on CPU. Search will be slower but will work.")
                cleanup_gpu_memory()  # Clean up after failure
                index = index_cpu
            else:
                raise
    else:
        index = index_cpu
    
    # Step 3: Save index and metadata
    print("Step 3: Saving index and metadata...")
    
    # Save metadata FIRST (before index) to ensure it's saved even if process is interrupted
    # Metadata is critical for retrieval, and rebuilding it is expensive
    print("Saving metadata first (most critical)...")
    id_to_doc = {i: (doc_ids[i], documents[i]) for i in range(len(doc_ids))}
    metadata_path = os.path.join(FAISS_INDEX_DIR, "doc_metadata.pkl")
    try:
        with open(metadata_path, 'wb') as f:
            pickle.dump(id_to_doc, f)
        print(f"Metadata saved to: {metadata_path}")
    except Exception as e:
        print(f"ERROR saving metadata: {e}")
        raise
    
    # Save model info (small, fast)
    model_info_path = os.path.join(FAISS_INDEX_DIR, "model_info.json")
    try:
        with open(model_info_path, 'w') as f:
            json.dump({
                'model_name': SENTENCE_TRANSFORMER_MODEL,
                'dimension': dimension,
                'metric': 'cosine' if FAISS_METRIC == faiss.METRIC_INNER_PRODUCT else 'l2',
                'num_documents': len(documents)
            }, f, indent=2)
        print(f"Model info saved to: {model_info_path}")
    except Exception as e:
        print(f"WARNING: Could not save model info: {e}")
        # Non-critical, continue
    
    # Save FAISS index last (can be rebuilt if needed, though it's expensive)
    print("Saving FAISS index...")
    faiss_index_path = os.path.join(FAISS_INDEX_DIR, "faiss.index")
    try:
        # If index is on GPU, move back to CPU for saving
        if use_gpu_for_index:
            print("Moving index back to CPU for saving...")
            index_cpu = faiss.index_gpu_to_cpu(index)
            faiss.write_index(index_cpu, faiss_index_path)
        else:
            faiss.write_index(index, faiss_index_path)
        print(f"FAISS index saved to: {faiss_index_path}")
    except Exception as e:
        print(f"ERROR saving FAISS index: {e}")
        raise
    
    return index, id_to_doc

def load_faiss_index(index_dir=FAISS_INDEX_DIR, use_gpu=True):
    """
    Load a previously built FAISS index and metadata.
    
    Args:
        index_dir: Directory containing the index
        use_gpu: Whether to use GPU if available
    
    Returns:
        index: FAISS index (on GPU if available and requested)
        id_to_doc: Dictionary mapping index position to (doc_id, doc_text)
        model: SentenceTransformer model
    """
    print(f"Loading FAISS index from: {index_dir}")
    
    # Load model info
    model_info_path = os.path.join(index_dir, "model_info.json")
    if os.path.exists(model_info_path):
        with open(model_info_path, 'r') as f:
            model_info = json.load(f)
        model_name = model_info['model_name']
        print(f"Using model: {model_name}")
    else:
        model_name = SENTENCE_TRANSFORMER_MODEL
        print(f"Model info not found, using default: {model_name}")
    
    # Load model
    model = SentenceTransformer(model_name)
    
    # Load index from disk (always on CPU)
    faiss_index_path = os.path.join(index_dir, "faiss.index")
    index_cpu = faiss.read_index(faiss_index_path)
    print(f"Loaded index with {index_cpu.ntotal} documents")
    
    # Move to GPU if available and requested
    if use_gpu:
        gpu_res = get_faiss_gpu_resources()
        if gpu_res is not None:
            try:
                # Clean up GPU memory first
                cleanup_gpu_memory()
                
                # Check GPU memory before moving
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.get_device_properties(GPU_ID).total_memory / 1e9
                    gpu_allocated = torch.cuda.memory_allocated(GPU_ID) / 1e9
                    gpu_free = gpu_memory - gpu_allocated
                    index_size_gb = (index_cpu.ntotal * index_cpu.d * 4) / 1e9  # float32 = 4 bytes
                    
                    print(f"GPU {GPU_ID} total memory: {gpu_memory:.2f} GB")
                    print(f"GPU {GPU_ID} allocated: {gpu_allocated:.2f} GB")
                    print(f"GPU {GPU_ID} free: {gpu_free:.2f} GB")
                    print(f"Estimated index size: {index_size_gb:.2f} GB")
                    
                    if index_size_gb > gpu_free * 0.9:  # Leave 10% buffer
                        print(f"WARNING: Index size ({index_size_gb:.2f} GB) is too large for available GPU memory ({gpu_free:.2f} GB)")
                        print("Keeping index on CPU. Search will be slower but will work.")
                        index = index_cpu
                    else:
                        print(f"Moving index to GPU (GPU {GPU_ID})...")
                        index = faiss.index_cpu_to_gpu(gpu_res, GPU_ID, index_cpu)
                        print("Index on GPU, ready for fast search")
                else:
                    index = index_cpu
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "alloc fail" in str(e).lower():
                    print(f"WARNING: GPU out of memory when moving index to GPU: {e}")
                    print("Keeping index on CPU. Search will be slower but will work.")
                    cleanup_gpu_memory()  # Clean up after failure
                    index = index_cpu
                else:
                    raise
        else:
            print("GPU not available, keeping index on CPU")
            index = index_cpu
    else:
        index = index_cpu
    
    # Load metadata
    metadata_path = os.path.join(index_dir, "doc_metadata.pkl")
    with open(metadata_path, 'rb') as f:
        id_to_doc = pickle.load(f)
    print(f"Loaded metadata for {len(id_to_doc)} documents")
    
    return index, id_to_doc, model

def retrieve_docs_dense(claim, model, index, id_to_doc, top_k=TOP_K):
    """
    Retrieve top-k documents using dense retrieval (SentenceTransformer + FAISS).
    
    Args:
        claim: The claim to search for
        model: SentenceTransformer model
        index: FAISS index
        id_to_doc: Dictionary mapping index position to (doc_id, doc_text)
        top_k: Number of documents to retrieve
    
    Returns:
        List of dictionaries with 'doc_id', 'text', and 'score'
    """
    # Encode the claim
    claim_embedding = model.encode(
        [claim],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype('float32')
    
    # Search in FAISS index
    scores, indices = index.search(claim_embedding, top_k)
    
    results = []
    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx < 0:  # Invalid index
            continue
        
        doc_id, doc_text = id_to_doc[idx]
        results.append({
            'doc_id': doc_id,
            'text': doc_text,
            'score': float(score)  # Cosine similarity score (higher is better)
        })
    
    return results

if __name__ == '__main__':
    # Check if index already exists
    index_exists = os.path.exists(os.path.join(FAISS_INDEX_DIR, "faiss.index"))
    
    if index_exists:
        print(f"Found existing FAISS index at {FAISS_INDEX_DIR}")
        print("Loading existing index...")
        index, id_to_doc, model = load_faiss_index(use_gpu=USE_GPU)
    else:
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
        # import pdb; pdb.set_trace()
        # documents = documents[:100]
        # doc_ids = doc_ids[:100]
        
        # Load sentence transformer model
        print(f"Loading SentenceTransformer model: {SENTENCE_TRANSFORMER_MODEL}")
        model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
        
        # Build FAISS index
        index, id_to_doc = build_faiss_index(documents, doc_ids, model)
        # Note: index is already on GPU if available, no need to reload
    
    
    
    # Processing claims
    print("\n" + "="*50)
    print("Start extracting most relevant wiki docs using dense retrieval...")
    print("="*50)
    
    f = open(TRAIN_FILE, 'r')
    l = f.readlines()
    res = []
    for l_ in l:
        res.append(json.loads(l_))
    f.close()
    df = pd.DataFrame(res)
    claims = df['claim'].tolist()
    print(f"Total claims: {len(claims)}")
    
    # Retrieve documents for each claim
    claim_docs_dense = []
    for i, claim in enumerate(tqdm(claims, desc="Retrieving documents")):
        retrieved_docs = retrieve_docs_dense(claim, model, index, id_to_doc, top_k=TOP_K)
        claim_docs_dense.append({
            'claim_id': i,
            'claim': claim,
            'retrieved_documents': retrieved_docs
        })
        # break
    
    print(f"Retrieved documents for {len(claim_docs_dense)} claims using dense retrieval")
    
    # Save results to JSON
    with open(OUTPUT, 'w') as f:
        json.dump(claim_docs_dense, f, indent=2)
    
    print(f"Results saved to {OUTPUT}")
    print(f"Sample entry:\n{json.dumps(claim_docs_dense[0], indent=2)}")
    
    # Summary
    print("\n=== Summary ===")
    print(f"Model used: {SENTENCE_TRANSFORMER_MODEL}")
    print(f"Index directory: {FAISS_INDEX_DIR}")
    gpu_res = get_faiss_gpu_resources()
    if gpu_res is not None and USE_GPU:
        print(f"GPU: Enabled (GPU {GPU_ID}, {faiss.get_num_gpus()} GPU(s) available)")
    else:
        print("GPU: Disabled or not available (using CPU)")
    print(f"Claims processed: {len(claims)}")
    print(f"Output file: {OUTPUT}")
    print("="*50)



# nohup python build_and_search_with_dense.py >> dev_logs/log_dense_miniLM_dev.log 2>&1&
