import subprocess
import os
import time
import argparse
import json
from typing import List, Dict
import re, time, httpx, json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import cycle
import pandas as pd

LABELS = {"SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"}
LABEL_RE = re.compile(r'^\s*(SUPPORTS|REFUTES|NOT[_ ]ENOUGH[_ ]INFO)\s*$', re.I)

CLOSED_BOOK_TMPL = (
    "You are a strict fact verifier.\n"
    "Without external sources, decide if the claim is true, false, or unknown.\n"
    "If you do not know for sure, output NOT_ENOUGH_INFO.\n\n"
    "Claim: \"{claim}\"\n\n"
    "Answer with exactly ONE of these labels on a single line:\n"
    "SUPPORTS\nREFUTES\nNOT_ENOUGH_INFO\n\n"
    "Answer:\n "
)

def format_closed_book_llama3_prompt(claim: str):
    """Format prompt for llama3 models using system/user message format."""
    system_prompt = """You are a precise fact-checking system.
You will be provided with a CLAIM.

Your goal is to determine if the claim is true, false, or unknown based on your knowledge.

RULES:
- "SUPPORTS": The claim is true based on your knowledge.
- "REFUTES": The claim is explicitly false based on your knowledge.
- "NOT ENOUGH INFO": You do not have enough information to determine if the claim is true or false.
- Without external sources, decide if the claim is true, false, or unknown.
- If you do not know for sure, output NOT ENOUGH INFO.
"""

    user_prompt = f"""
<claim>
{claim}
</claim>

Your response (ONLY respond with one word/phrase: SUPPORTS, REFUTES, or NOT ENOUGH INFO):
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    return messages

def format_closed_book_general_prompt(claim: str):
    """Format prompt for non-llama3 models."""
    return CLOSED_BOOK_TMPL.format(claim=claim)

def read_fever_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def norm_gold_label(lbl: str) -> str:
    return lbl.replace(' ', '_').upper()

def normalize_label(text: str):
    m = LABEL_RE.search(text or "")
    if not m:
        return None
    lab = m.group(1).upper().replace(' ', '_')
    if lab == 'NOT_ENOUGH_INFO' or lab in {'SUPPORTS', 'REFUTES'}:
        return lab
    return None

def ask_ollama_http(model: str, prompt_or_messages, port: int, temperature: float = 0, retry: int = 1, use_chat_format: bool = False) -> str:
    """Call Ollama via HTTP API to specific port (for multi-GPU support)."""
    if use_chat_format:
        url = f"http://127.0.0.1:{port}/api/chat"
        payload = {
            "model": model,
            "messages": prompt_or_messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": 0.1,
            }
        }
    else:
        url = f"http://127.0.0.1:{port}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt_or_messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": 0.1,
            }
        }
    
    for _ in range(retry + 1):
        try:
            with httpx.Client(timeout=120.0) as client:
                resp = client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                if use_chat_format:
                    out = (data.get("message", {}).get("content") or "").strip()
                else:
                    out = (data.get("response") or "").strip()
                lab = normalize_label(out)
                if lab:
                    return lab
        except Exception as e:
            print(e)
            time.sleep(0.5)
            import pdb;pdb.set_trace()
    return "NOT_ENOUGH_INFO"  # safe fallback

def get_next_port():
    """Thread-safe round-robin port selection."""
    with _port_lock:
        return next(_port_cycle)

def process_single_row(args):
    """Process a single row - used by thread pool workers."""
    idx, r, model = args
    claim = r["claim"]
    gold = norm_gold_label(r.get("label", "NOT ENOUGH INFO"))
    
    # Use llama3 format if model contains "llama3"
    use_chat_format = 'llama3' in model.lower()
    if use_chat_format:
        prompt_or_messages = format_closed_book_llama3_prompt(claim)
    else:
        prompt_or_messages = format_closed_book_general_prompt(claim)
    
    port = get_next_port()  # Round-robin across GPUs
    pred = ask_ollama_http(model, prompt_or_messages, port=port, temperature=0, retry=1, use_chat_format=use_chat_format)
    return idx, r.get("id", None), claim, gold, pred

def run_closed_book_parallel(rows, model="llama3.1", max_n=None, num_workers=None):
    """
    Run closed-book inference in parallel across multiple GPUs.
    
    Args:
        rows: List of FEVER data rows
        model: Model name to use
        max_n: Max number of rows to process (None = all)
        num_workers: Number of parallel workers (default: NUM_GPUS * 2)
    """
    use_rows = rows if max_n is None else rows[:max_n]
    
    # Use 2 workers per GPU for better throughput (overlap compute & network)
    if num_workers is None:
        num_workers = len(GPU_IDS) * 2
    
    print(f"Running inference on {len(use_rows)} samples using {len(GPU_IDS)} GPU(s) with {num_workers} workers...")
    
    # Prepare arguments for parallel processing
    work_items = [(i, r, model) for i, r in enumerate(use_rows)]
    
    # Results storage (will be sorted by index later)
    results = [None] * len(use_rows)
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_row, item): item[0] for item in work_items}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Inference"):
            idx, claim_id, claim, gold, pred = future.result()
            results[idx] = (claim_id, claim, gold, pred)
    
    # Build dataframe from ordered results
    df = pd.DataFrame(results, columns=["id", "claim", "gold", "pred"])
    return df

# ===== Single-GPU fallback (if you only have 1 GPU) =====
def run_closed_book_sequential(rows, model="llama3.1", max_n=None):
    """Original sequential version for single GPU."""
    preds, golds, claim_ids, claims = [], [], [], []
    use_rows = rows if max_n is None else rows[:max_n]
    port = OLLAMA_PORTS[0]
    
    # Use llama3 format if model contains "llama3"
    use_chat_format = 'llama3' in model.lower()
    
    for r in tqdm(use_rows):
        claim = r["claim"]
        gold = norm_gold_label(r.get("label", "NOT ENOUGH INFO"))
        
        if use_chat_format:
            prompt_or_messages = format_closed_book_llama3_prompt(claim)
        else:
            prompt_or_messages = format_closed_book_general_prompt(claim)
        
        pred = ask_ollama_http(model, prompt_or_messages, port=port, temperature=0.1, retry=1, use_chat_format=use_chat_format)
        preds.append(pred)
        golds.append(gold)
        claim_ids.append(r.get("id", None))
        claims.append(claim)
    df = pd.DataFrame({"id": claim_ids, "claim": claims, "gold": golds, "pred": preds})
    return df



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process claims with LLM in chunks')
    parser.add_argument('--ollama_bin', type=str, help='Path to ollama bin')
    parser.add_argument('--models_path', type=str, help='Models Path')
    parser.add_argument('--log_dir', type=str, help='Path to log directory')
    parser.add_argument('--output_file', type=str, help='Path to output csv file')
    parser.add_argument('--gpus', type=list, default = [0, 1, 2, 3, 4, 5, 6, 7], help='gpu ids')
    parser.add_argument('--dev_file_path', type=str, help='fever dev file path')
    parser.add_argument('--model_name', type=str, default = 'llama3.1', help='model name')
    parser.add_argument('--port', type=int, default=11434, help='Ollama API port (default: 11434)')

    args = parser.parse_args()

    GPU_IDS = args.gpus
    OLLAMA_MODELS_PATH = args.models_path
    print(f"\n=== Detected {GPU_IDS} GPU(s) ===")
    
    # Start multiple Ollama instances, one per GPU on different ports
    BASE_PORT = args.port
    OLLAMA_PORTS = []

    os.makedirs(args.log_dir, exist_ok=True)

    for gpu_id in GPU_IDS:
        port = BASE_PORT + gpu_id
        OLLAMA_PORTS.append(port)
        # env_vars = f"CUDA_VISIBLE_DEVICES={gpu_id} OLLAMA_HOST=0.0.0.0:{port}"
        env_vars = f"CUDA_VISIBLE_DEVICES={gpu_id} OLLAMA_HOST=0.0.0.0:{port} OLLAMA_MODELS={OLLAMA_MODELS_PATH}"
        cmd = f"{env_vars} nohup {args.ollama_bin} serve > {args.log_dir}/ollama_serve_{gpu_id}.log 2>&1 &"
        os.system(cmd)
        print(f"Started Ollama on GPU {gpu_id} at port {port}")
    
    time.sleep(5)  # Give servers time to start
    
    print(f'\nOllama installed with {GPU_IDS} instance(s). Ports: {OLLAMA_PORTS}')

    dev_rows = read_fever_jsonl(args.dev_file_path)
    print('Loaded dev rows:', len(dev_rows))
    print('Keys:', list(dev_rows[0].keys()))

    # Round-robin port selector for load balancing across GPUs
    _port_cycle = cycle(OLLAMA_PORTS)
    _port_lock = __import__('threading').Lock()

    # ===== Run inference =====
    # Use parallel version for multi-GPU, sequential for single GPU
    if len(GPU_IDS) > 1:
        print(f"Multi-GPU mode: Using {len(GPU_IDS)} GPUs in parallel")
        df_dev = run_closed_book_parallel(dev_rows, model=args.model_name, max_n=None)
    else:
        print("Single-GPU mode: Running sequentially")
        df_dev = run_closed_book_sequential(dev_rows, model=args.model_name, max_n=None)
    

    df_dev.to_csv(args.output_file, index=False)

# python close_book_predict_result.py --ollama_bin='./bin/ollama' --log_dir='./ollama_servers/close_book' --dev_file_path='./data/fever-data/dev.jsonl' --output_file='./results/dev_res_llama3_newprompt/fever_closedbook_llama3.1_8b_instruct_dev_subset.csv' --models_path='./models/'
