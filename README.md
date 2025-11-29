# FEVER Hallucination Detection with RAG

This project implements a Retrieval-Augmented Generation (RAG) approach to detect hallucinations in claims using the FEVER (Fact Extraction and VERification) dataset. The system uses Pyserini and FAISS for document retrieval and Ollama (`qwen2.5:7b-instruct`, `llama3.1`) for LLM-based fact-checking.

## Overview

The system works in two main stages:

1. **Document Retrieval**: Uses Pyserini with BM25 to retrieve relevant Wikipedia documents for each claim
2. **Fact-Checking**: Uses an LLM (via Ollama) to classify claims as:
   - `SUPPORTS`: The retrieved documents support the claim
   - `REFUTES`: The retrieved documents refute the claim
   - `NOT ENOUGH INFO`: Insufficient information to determine support or refutation

## Requirements

- Python 3.7+
- Ollama installed and configured
- `qwen2.5:7b-instruct`, `llama3.1` model downloaded in Ollama
- CUDA-capable GPU (recommended for faster inference)

## Reproducible steps

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Install and set up Ollama:
   - Download from [ollama.ai](https://ollama.ai)
   - Pull the qwen2.5:7b-instruct model:
   ```bash
   ollama pull qwen2.5:7b-instruct
   ```
   - Pull the llama3.1 model:
   ```bash
   ollama pull llama3.1
   ```

3. Prepare dataset:
   - Download [shared_task_dev.jsonl](https://fever.ai/dataset/fever.html) and rename as `dev.jsonl` into `data/fever-data/`

4. Build index and retrieve:
   - Created retrieved documents results with BM25: 
      ```bash
      cd scripts && python build_and_search_with_lucern_idx.py
      ```
      This should produce `dev_claim_retrieved_docs_bm25_top5.json` (set desired output name from the script)
   - Created retrieved documents results with Dense:
      - For `all-MiniLM-L6-v2`:
         ```bash
         cd scripts && python build_and_search_with_dense.py
         ```
         This should produce `dev_claim_retrieved_docs_dense_miniLM_top5.json`
      - For `Qwen/Qwen3-Embedding-0.6B`:
         ```bash
         cd scripts && python dense_qwen.py
         ```
         This should produce `dev_claim_retrieved_docs_dense_qwen3_top5.json`

5. To get classification result from ollama, use these 2 scripts:
   - start 8 Ollama services concurrently on 8 different ports on server with GPUs (configurable in .sh script for the amount of GPU):
         ```bash
         cd scripts && start_ollama_servers.sh
         ```
   - Set configuration (most importantly, `RETRIEVAL_METHOD`, `MODEL` and `MODEL_PROMPT`) in `run_parallel.sh`. This will split the input file into #NUM_PROCESSES batches, pass it to `pred_results.py`, `pred_results.py` will then output to interim chunks files, and combine all interim chunk files back to one resulted classification file specified in `$MERGED_CSV`. Command to run:
         ```bash
         cd scripts && run_parallel.sh
         ```

## Data

Results can be downloaded from Google Drive:
- `results/`: https://drive.google.com/drive/folders/1JR1Mhr_-y_4oDFFtjUbRFY5jlD2oEZoj?usp=sharing
- `results-interim-data/`: https://drive.google.com/drive/folders/1kfpB7YDnRFF_bWD2est8psMWz0xysYyu?usp=sharing

## Results
- Analysis used in the report can be found in [`notebooks/analysis_dev_final.ipynb`](notebooks/analysis_dev_final.ipynb)

## Project structures

```text
fact-or-fiction-fever/
├─ data/
│  └─ fever-data/
│     ├─ dev.jsonl
│     └─ train.jsonl
├─ dev_logs/
│  ├─ log_bm25_dev.log
│  ├─ log_dense_miniLM_dev.log
│  ├─ log_dense_qwen_dev_top5.log
│  ├─ log_par_bm25_llama3_newprompt.log
│  ├─ log_par_bm25_top5_llama3.log
│  ├─ log_par_bm25_top5.log
│  ├─ log_par_miniLM_top5_llama3.log
│  ├─ log_par_miniLM_top5.log
│  ├─ log_par_qwen3_top5_llama3_newprompt.log
│  ├─ log_par_qwen3_top5_llama3.log
│  └─ log_par_qwen3_top5.log
├─ notebooks/
│  └─ analysis_dev_final.ipynb
├─ results/
│  ├─ dev_claim_retrieved_docs_bm25_top5.json
│  ├─ dev_claim_retrieved_docs_dense_miniLM_top5.json
│  ├─ dev_claim_retrieved_docs_dense_qwen3_top5.json
│  ├─ dev_res_llama3_newprompt/
│  │  ├─ dev_llm_classification_results_merged_bm25_top5.csv
│  │  ├─ dev_llm_classification_results_merged_dense_miniLM_top5.csv
│  │  ├─ dev_llm_classification_results_merged_dense_qwen3_top5.csv
│  │  └─ fever_closedbook_llama3.1_8b_instruct_dev_subset.csv
│  └─ dev_res_qwen25/
│     ├─ dev_llm_classification_results_merged_bm25_top5.csv
│     ├─ dev_llm_classification_results_merged_dense_miniLM_top5.csv
│     ├─ dev_llm_classification_results_merged_dense_qwen3_top5.csv
│     └─ fever_closedbook_qwen2.5_7b_instruct_dev_subset.csv
├─ results-interim-data/
│  ├─ parallel_results_bm25_top5_llama3_newprompt/
│  │  ├─ process_0.log
│  │  ├─ process_1.log
│  │  ├─ process_2.log
│  │  ├─ process_3.log
│  │  ├─ process_4.log
│  │  ├─ process_5.log
│  │  ├─ process_6.log
│  │  ├─ process_7.log
│  │  ├─ results_part_0.csv
│  │  ├─ results_part_0.json
│  │  ├─ results_part_1.csv
│  │  ├─ results_part_1.json
│  │  ├─ results_part_2.csv
│  │  ├─ results_part_2.json
│  │  ├─ results_part_3.csv
│  │  ├─ results_part_3.json
│  │  ├─ results_part_4.csv
│  │  ├─ results_part_4.json
│  │  ├─ results_part_5.csv
│  │  ├─ results_part_5.json
│  │  ├─ results_part_6.csv
│  │  ├─ results_part_6.json
│  │  ├─ results_part_7.csv
│  │  └─ results_part_7.json
│  ├─ parallel_results_dense_miniLM_top5_llama3_newprompt/
│  │  ├─ process_0.log
│  │  ├─ process_1.log
│  │  ├─ process_2.log
│  │  ├─ process_3.log
│  │  ├─ process_4.log
│  │  ├─ process_5.log
│  │  ├─ process_6.log
│  │  ├─ process_7.log
│  │  ├─ results_part_0.csv
│  │  ├─ results_part_0.json
│  │  ├─ results_part_1.csv
│  │  ├─ results_part_1.json
│  │  ├─ results_part_2.csv
│  │  ├─ results_part_2.json
│  │  ├─ results_part_3.csv
│  │  ├─ results_part_3.json
│  │  ├─ results_part_4.csv
│  │  ├─ results_part_4.json
│  │  ├─ results_part_5.csv
│  │  ├─ results_part_5.json
│  │  ├─ results_part_6.csv
│  │  ├─ results_part_6.json
│  │  ├─ results_part_7.csv
│  │  └─ results_part_7.json
│  └─ parallel_results_dense_qwen3_top5_llama3_newprompt/
│     └─ ...
├─ scripts/
│  ├─ build_and_search_with_dense.py
│  ├─ build_and_search_with_lucern_idx.py
│  ├─ close_book_predict_result.py
│  ├─ dense_qwen.py
│  ├─ pred_results.py
│  ├─ run_parallel.sh
│  └─ start_ollama_servers.sh
├─ README.md
└─ requirements.txt
```

