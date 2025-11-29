import pandas as pd
import json
import requests
import re
import argparse
import time
from typing import Literal
from tqdm import tqdm

MODEL = 'llama3.1'#'qwen2.5:7b-instruct'
FILE = ''

def format_llama3_promt(doc_texts, claim):
    # Format the context using XML tags for Llama 3.1
    formatted_docs = []
    for i, doc in enumerate(doc_texts):
        # We add an index so the model can cite specific documents if needed
        formatted_docs.append(f'<doc index="{i+1}">\n{doc}\n</doc>')

    # Join them clearly
    context_block = "\n".join(formatted_docs)
    
    system_prompt = """You are a precise fact-checking system.
You will be provided with a CLAIM and a set of retrieved DOCUMENTS.

Your goal is to output a JSON object with two keys:
1. "reasoning": A brief explanation of which document supports or refutes the claim.
2. "label": One of "SUPPORTS", "REFUTES", or "NOT ENOUGH INFO".

RULES:
- "SUPPORTS": The claim is fully supported by the information in the documents.
- "REFUTES": The claim is explicitly contradicted by the documents.
- "NOT ENOUGH INFO": The documents do not contain the answer, or the information is unrelated.
- Do not use outside knowledge. Rely ONLY on the provided <doc> tags.
"""

    user_prompt = f"""
<documents>
{context_block}
</documents>

<claim>
{claim}
</claim>

Your response (ONLY respond with one word/phrase: SUPPORTS, REFUTES, or NOT ENOUGH INFO):
"""

    # If using a chat template (Recommended):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    return messages

def format_general_prompt(doc_texts, claim):
    # Combine documents into context
    context = "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(doc_texts)])
    
    # Create a strict prompt that constrains output to only 3 categories
    prompt = f"""You are a fact-checking assistant. Your task is to determine if the provided documents SUPPORT, REFUTE, or provide NOT ENOUGH INFO for the given claim.

CRITICAL INSTRUCTIONS:
- You MUST respond with ONLY one of these three exact phrases (case-insensitive):
  1. SUPPORTS
  2. REFUTES  
  3. NOT ENOUGH INFO

- SUPPORTS: The documents clearly support or confirm the claim
- REFUTES: The documents clearly contradict or refute the claim
- NOT ENOUGH INFO: The documents don't provide sufficient information to determine support or refutation

Claim: {claim}

Retrieved Documents:
{context}

Based on the above documents, determine if they SUPPORT, REFUTE, or provide NOT ENOUGH INFO for the claim.

Your response (ONLY respond with one word/phrase: SUPPORTS, REFUTES, or NOT ENOUGH INFO):"""
    return prompt

def check_claim_with_llm(claim: str, retrieved_documents: list, model: str = "qwen2:7b", request_num: int = 0, log_frequency: int = 50, port: int = 11434, sleep_after: float = 0.1) -> Literal["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]:
    """
    Check if the claim is supported, refuted, or has not enough info based on retrieved documents.
    
    Args:
        claim: The claim to verify
        retrieved_documents: List of retrieved documents (strings or dicts with text)
        model: Ollama model name to use
    
    Returns:
        One of: "SUPPORTS", "REFUTES", "NOT ENOUGH INFO"
    """
    # Extract text from documents if they're dicts
    doc_texts = []
    for doc in retrieved_documents:
        if isinstance(doc, dict):
            # Try common keys for document text
            text = doc.get('text', doc.get('content', doc.get('body', str(doc))))
        else:
            text = str(doc)
        doc_texts.append(text)
    
    # Determine if we should use chat format or prompt format
    use_chat_format = False
    messages = None
    prompt = None
    
    # Use chat format for Llama models (they use chat templates)
    if 'llama3' in model.lower() or 'llama' in model.lower():
        messages = format_llama3_promt(doc_texts, claim)
        use_chat_format = True
    else:
        prompt = format_general_prompt(doc_texts, claim)
        use_chat_format = False
    
    # Call Ollama API
    try:
        if use_chat_format:
            # Use /api/chat endpoint for chat format
            ollama_url = f"http://localhost:{port}/api/chat"
            # import pdb; pdb.set_trace()
            response = requests.post(
                ollama_url,
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.0,  # Very low temperature for deterministic output
                        "top_p": 0.1,
                        "num_predict": 10  # Limit output length to just the category
                    }
                },
                timeout=60
            )
        else:
            # Use /api/generate endpoint for prompt format
            ollama_url = f"http://localhost:{port}/api/generate"
            response = requests.post(
                ollama_url,
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.0,  # Very low temperature for deterministic output
                        "top_p": 0.1,
                        # "num_predict": 10  # Limit output length to just the category
                    }
                },
                timeout=60
            )
        response.raise_for_status()
        result = response.json()
        
        # Handle both /api/generate and /api/chat response formats
        if use_chat_format:
            # Chat API returns response in message.content
            answer = result.get("message", {}).get("content", "")
        else:
            # Generate API returns response directly
            answer = result.get("response", "").strip()
        
        # Log the full Ollama response periodically (first request and every Nth request)
        should_log = (request_num == 0) or (request_num % log_frequency == 0)
        if should_log:
            print(f"\n[Request #{request_num}] Ollama Response:")
            print(f"  Full output: {answer}")
            print(f"  Raw JSON keys: {list(result.keys())}")
            if 'done' in result:
                print(f"  Done: {result.get('done')}")
            if 'context' in result:
                print(f"  Context length: {len(result.get('context', []))}")
            if 'total_duration' in result:
                duration_ms = result.get('total_duration', 0) / 1_000_000
                print(f"  Total duration: {duration_ms:.2f}ms")
            if 'load_duration' in result:
                load_ms = result.get('load_duration', 0) / 1_000_000
                print(f"  Load duration: {load_ms:.2f}ms")
            if 'prompt_eval_count' in result:
                print(f"  Prompt eval count: {result.get('prompt_eval_count')}")
            if 'eval_count' in result:
                print(f"  Eval count: {result.get('eval_count')}")
        
        # Normalize to uppercase for matching
        answer_upper = answer.upper()
        
        # Strict parsing: look for exact matches (case-insensitive)
        # Use regex to find the category even if there's extra text
        result_value = ""
        if re.search(r'\bSUPPORTS\b', answer_upper):
            result_value = "SUPPORTS"
        elif re.search(r'\bREFUTES\b', answer_upper):
            result_value = "REFUTES"
        elif re.search(r'\bNOT\s+ENOUGH\s+INFO\b', answer_upper) or re.search(r'\bNOT_ENOUGH_INFO\b', answer_upper) or re.search(r'\bNOTENOUGHINFO\b', answer_upper):
            result_value = "NOT ENOUGH INFO"
        else:
            # # Last resort: check if any word matches
            # words = re.findall(r'\b\w+\b', answer_upper)
            # for word in words:
            #     if word == "SUPPORTS":
            #         return "SUPPORTS"
            #     elif word == "REFUTES":
            #         return "REFUTES"
            
            # If no match found, default to NOT ENOUGH INFO
            print(f"Warning: Could not parse response '{answer}', defaulting to empty string")
            result_value = ""
        
        # Sleep after each request to avoid overwhelming the server
        if sleep_after > 0:
            time.sleep(sleep_after)
        
        return result_value
            
    except requests.exceptions.RequestException as e:
        print(f"Error calling Ollama: {e}")
        return ""
    except Exception as e:
        print(f"Unexpected error: {e}")
        import pdb; pdb.set_trace()
        return ""



if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Process claims with LLM in chunks')
    parser.add_argument('--file', type=str, default=FILE, help='Path to input JSON file')
    parser.add_argument('--model', type=str, default=MODEL, help='Ollama model name')
    parser.add_argument('--port', type=int, default=11434, help='Ollama API port (default: 11434)')
    parser.add_argument('--chunk-size', type=int, default=None, help='Number of claims to process per chunk')
    parser.add_argument('--start', type=int, default=0, help='Start index (0-based)')
    parser.add_argument('--end', type=int, default=None, help='End index (exclusive, None for all)')
    parser.add_argument('--output-csv', type=str, default='llm_classification_results.csv', help='Output CSV filename')
    parser.add_argument('--output-json', type=str, default='llm_classification_results.json', help='Output JSON filename')
    parser.add_argument('--log-ollama-frequency', type=int, default=50, help='Log Ollama responses every N requests (default: 50, first request always logged)')
    
    args = parser.parse_args()
    
    # Load data
    all_claims_facts = json.load(open(args.file, 'r'))
    total_items = len(all_claims_facts)
    
    # Determine processing range
    start_idx = args.start
    end_idx = args.end if args.end is not None else total_items
    end_idx = min(end_idx, total_items)
    
    print(f"Total items in file: {total_items}")
    print(f"Processing range: {start_idx} to {end_idx} (total: {end_idx - start_idx} items)")
    print(f"Ollama API: http://localhost:{args.port} | Model: {args.model}")
    print(f"Ollama responses will be logged for: first request and every {args.log_ollama_frequency} requests")
    
    # Process in chunks if chunk_size is specified
    results = []
    items_to_process = all_claims_facts[start_idx:end_idx]
    
    if args.chunk_size:
        num_chunks = (len(items_to_process) + args.chunk_size - 1) // args.chunk_size
        print(f"Processing in {num_chunks} chunks of size {args.chunk_size}")
        
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * args.chunk_size
            chunk_end = min(chunk_start + args.chunk_size, len(items_to_process))
            chunk = items_to_process[chunk_start:chunk_end]
            
            print(f"\nProcessing chunk {chunk_idx + 1}/{num_chunks} (items {chunk_start} to {chunk_end - 1})")
            
            for i, item in enumerate(tqdm(chunk, desc=f"Chunk {chunk_idx + 1}")):
                actual_idx = start_idx + chunk_start + i
                claim = item.get('claim', item.get('query', ''))
                retrieved_docs = item.get('retrieved_documents', item.get('documents', item.get('docs', [])))
                result = check_claim_with_llm(claim, retrieved_docs, args.model, actual_idx, args.log_ollama_frequency, args.port)
                results.append({
                    'index': actual_idx,
                    'claim': claim,
                    'classification': result
                })
    else:
        # Process all items without chunking
        print("Processing all items sequentially...")
        for i, item in enumerate(tqdm(items_to_process, desc="Processing claims")):
            actual_idx = start_idx + i
            claim = item.get('claim', item.get('query', ''))
            retrieved_docs = item.get('retrieved_documents', item.get('documents', item.get('docs', [])))
            result = check_claim_with_llm(claim, retrieved_docs, args.model, actual_idx, args.log_ollama_frequency)
            results.append({
                'index': actual_idx,
                'claim': claim,
                'classification': result
            })
    
    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    print(f"\nProcessed {len(results)} claims")
    print(df_results.head())

    # Save results to files
    df_results.to_csv(args.output_csv, index=False)
    print(f"\nResults saved to {args.output_csv}")
    
    df_results.to_json(args.output_json, orient='records', indent=2)
    print(f"Results saved to {args.output_json}")
    

# python pred_results.py --file dev_claim_retrieved_docs_dense_qwen3_top5.json --model llama3.1 --port 11434 --chunk-size 1000 --start 0 --end 1000 --output-csv dev_res_llama3/dev_llm_classification_results_merged_dense_qwen3_top5_temp.csv --output-json dev_res_llama3/dev_llm_classification_results_merged_dense_qwen3_top5_temp.json
