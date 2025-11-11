# FEVER Hallucination Detection with RAG

This project implements a Retrieval-Augmented Generation (RAG) approach to detect hallucinations in claims using the FEVER (Fact Extraction and VERification) dataset. The system uses Pyserini for document retrieval and Ollama (Qwen2:7b) for LLM-based fact-checking.

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
- Qwen2:7b model downloaded in Ollama
- CUDA-capable GPU (recommended for faster inference)

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Install and set up Ollama:
   - Download from [ollama.ai](https://ollama.ai)
   - Pull the Qwen2:7b model:
   ```bash
   ollama pull qwen2:7b
   ```

3. Prepare your data:
   - Place FEVER training data as `train.jsonl` in the parent directory
   - Place Wikipedia pages as `wiki-pages/wiki-*.jsonl` in the parent directory

## Data

All pre-processed data files are available for download from Google Drive:

**[Download all data files from Google Drive](https://drive.google.com/drive/folders/1koNF2Sr-D-ltDxUUn_EAY1pt2M0tAe3t?usp=sharing)**

The folder contains:
- `claim_retrieved_docs_bm25.json` - Retrieved documents for each claim using BM25
- `pyserini/` - Pre-built Pyserini index folder
- `llm_classification_results_merged.csv` - Merged classification results (CSV format)
- `llm_classification_results_merged.json` - Merged classification results (JSON format)
- `results_part_0.csv`, `results_part_0.json` - Partial results from process 0
- `results_part_1.csv`, `results_part_1.json` - Partial results from process 1
- `results_part_2.csv`, `results_part_2.json` - Partial results from process 2
- `results_part_3.csv`, `results_part_3.json` - Partial results from process 3
- `results_part_4.csv`, `results_part_4.json` - Partial results from process 4
- `results_part_5.csv`, `results_part_5.json` - Partial results from process 5
- `results_part_6.csv`, `results_part_6.json` - Partial results from process 6
- `results_part_7.csv`, `results_part_7.json` - Partial results from process 7


## Usage

### Step 1: Build Index and Retrieve Documents

Build a Lucene index from Wikipedia articles and retrieve relevant documents for each claim:

```bash
python build_and_search_with_lucern_idx.py >> log.log 2>&1 &
```

This will:
- Load Wikipedia articles from `../wiki-pages/wiki-*.jsonl`
- Build a Pyserini Lucene index
- Retrieve top-k relevant documents for each claim using BM25
- Save results to `claim_retrieved_docs_bm25.json`

### Step 2: Start Ollama Servers (for parallel processing)

Start multiple Ollama server instances on different ports:

```bash
bash start_ollama_servers.sh >> log_servers.log 2>&1&
```

This script starts 8 Ollama servers (default) on ports 11434-11441. Adjust the `NUM_SERVERS` and `CUDA_DEVICES` variables in the script to match your setup.

### Step 3: Run Fact-Checking

#### Single Process Mode

Process claims sequentially with a single Ollama server:

```bash
python pred_results.py --file claim_retrieved_docs_bm25.json --model qwen2:7b --port 11434
```

#### Parallel Processing Mode

Process claims in parallel across multiple Ollama servers:

```bash
bash run_parallel.sh >> log_par.log 2>&1&
```

This will:
- Split the dataset across multiple processes
- Each process connects to a different Ollama server port
- Merge results into `llm_classification_results_merged.csv` and `llm_classification_results_merged.json`

### Command-Line Options

For `pred_results.py`:

- `--file`: Path to input JSON file (default: `./claim_retrieved_docs_bm25.json`)
- `--model`: Ollama model name (default: `qwen2:7b`)
- `--port`: Ollama API port (default: `11434`)
- `--chunk-size`: Number of claims per chunk (optional)
- `--start`: Start index (0-based, default: 0)
- `--end`: End index (exclusive, default: None for all)
- `--output-csv`: Output CSV filename (default: `llm_classification_results.csv`)
- `--output-json`: Output JSON filename (default: `llm_classification_results.json`)
- `--log-ollama-frequency`: Log Ollama responses every N requests (default: 50)

## Project Structure

```
server/
├── build_and_search_with_lucern_idx.py  # Build index and retrieve documents
├── pred_results.py                       # LLM-based fact-checking
├── run_parallel.sh                       # Parallel processing script
├── start_ollama_servers.sh               # Start multiple Ollama servers
├── eval.ipynb                            # Evaluation notebook
├── parallel_results/                     # Output directory for parallel processing
└── logs/                                 # Log files
```

## Output Format

The fact-checking results are saved as CSV and JSON files with the following structure:

```json
{
  "index": 0,
  "claim": "Example claim text",
  "classification": "SUPPORTS"
}
```

## Notes

- The system uses BM25 with parameters (k1=1.5, b=0.75)
- LLM temperature is set to 0.0 for deterministic outputs
- Processing large datasets may take several hours depending on hardware
- Monitor logs in `parallel_results/` directory during parallel processing

