#!/bin/bash

# Configuration
# =============
# Change these variables to update all file paths at once
RETRIEVAL_METHOD="bm25_top5"  # e.g., "bm25_top5", "dense_miniLM_top5"
MODEL_PROMPT="llama3_newprompt"  # e.g., "llama3_newprompt"
MODEL='llama3.1'
# "qwen2.5:7b-instruct"
INPUT_FILE="../results/dev_claim_retrieved_docs_${RETRIEVAL_METHOD}.json"
TOTAL_ENTRIES=19998
NUM_PROCESSES=8
SCRIPT_PATH="./pred_results.py"
BASE_PORT=11434  # Base port for Ollama (matches start_ollama_servers.sh)
USE_DIFFERENT_PORTS=true  # Use different ports for each process (one per Ollama server)
# Merge all CSV files
MERGED_CSV="../results/dev_res_${MODEL_PROMPT}/dev_llm_classification_results_merged_${RETRIEVAL_METHOD}.csv"
MERGED_JSON="../results/dev_res_${MODEL_PROMPT}/dev_llm_classification_results_merged_${RETRIEVAL_METHOD}.json"
# =============

# Calculate chunk size per process
CHUNK_SIZE=$((TOTAL_ENTRIES / NUM_PROCESSES))
REMAINDER=$((TOTAL_ENTRIES % NUM_PROCESSES))



echo "=========================================="
echo "Parallel Processing Setup"
echo "=========================================="
echo "Total entries: $TOTAL_ENTRIES"
echo "Number of processes: $NUM_PROCESSES"
echo "Chunk size per process: ~$CHUNK_SIZE"
if [ "$USE_DIFFERENT_PORTS" = "true" ]; then
    echo "Port configuration: Different port per process (starting from $BASE_PORT)"
else
    echo "Port configuration: All processes use port $BASE_PORT"
fi
echo "=========================================="
echo ""

# Create output directory for individual results
OUTPUT_DIR="parallel_results_${RETRIEVAL_METHOD}_${MODEL_PROMPT}"
mkdir -p "$OUTPUT_DIR"

# PID file to track processes (useful if script is interrupted)
PID_FILE="$OUTPUT_DIR/running_pids.txt"
> "$PID_FILE"  # Clear/create PID file

# Array to store process IDs
PIDS=()

# Function to calculate start and end for each process
calculate_range() {
    local process_num=$1
    local start=$((process_num * CHUNK_SIZE))
    local end=$((start + CHUNK_SIZE))
    
    # Add remainder to the last process
    if [ $process_num -eq $((NUM_PROCESSES - 1)) ]; then
        end=$((end + REMAINDER))
    fi
    
    echo "$start $end"
}

# Launch parallel processes
echo "Launching $NUM_PROCESSES parallel processes..."
echo ""

for i in $(seq 0 $((NUM_PROCESSES - 1))); do
    read start end <<< $(calculate_range $i)
    
    CSV_OUTPUT="$OUTPUT_DIR/results_part_${i}.csv"
    JSON_OUTPUT="$OUTPUT_DIR/results_part_${i}.json"
    
    # Calculate port for this process
    if [ "$USE_DIFFERENT_PORTS" = "true" ]; then
        PORT=$((BASE_PORT + i))
    else
        PORT=$BASE_PORT
    fi
    
    echo "Process $i: Processing entries $start to $((end - 1))"
    echo "  Output: $CSV_OUTPUT"
    echo "  Ollama port: $PORT"
    
    # Run the Python script in background with setsid (new session) and nohup
    # setsid creates a new session, making the process independent of the terminal
    setsid nohup python3 "$SCRIPT_PATH" \
        --file "$INPUT_FILE" \
        --model "$MODEL" \
        --port "$PORT" \
        --chunk-size "$CHUNK_SIZE" \
        --start "$start" \
        --end "$end" \
        --output-csv "$CSV_OUTPUT" \
        --output-json "$JSON_OUTPUT" \
        > "$OUTPUT_DIR/process_${i}.log" 2>&1 &
    
    PID=$!
    PIDS+=($PID)
    echo "$PID" >> "$PID_FILE"  # Save PID to file
    echo "  Started with PID: $PID"
    echo ""
    
    # Small delay to avoid overwhelming the system
    sleep 1
done

echo "=========================================="
echo "All processes launched!"
echo "=========================================="
echo "Process IDs: ${PIDS[@]}"
echo "PID file saved to: $PID_FILE"
echo ""
echo "Note: Processes are running in independent sessions and will"
echo "      continue even if you log out. If the script is interrupted,"
echo "      check $PID_FILE for running process IDs."
echo ""
echo "Monitor progress with:"
echo "  tail -f $OUTPUT_DIR/process_*.log"
echo ""
echo "Waiting for all processes to complete..."
echo ""

# Wait for all processes to complete
FAILED=0
for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    wait $pid
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "✓ Process $i (PID $pid) completed successfully"
    else
        echo "✗ Process $i (PID $pid) failed with exit code $exit_code"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "=========================================="
if [ $FAILED -eq 0 ]; then
    echo "All processes completed successfully!"
    echo "=========================================="
    echo ""
    echo "Merging results..."
    
    
    
    # Check if Python has pandas (for merging)
    python3 -c "import pandas as pd" 2>/dev/null
    if [ $? -eq 0 ]; then
        # Use Python to merge and sort by index
        python3 << EOF
import pandas as pd
import json
import glob
import os

# Merge CSV files
csv_files = sorted(glob.glob("$OUTPUT_DIR/results_part_*.csv"))
dfs = []
for f in csv_files:
    df = pd.read_csv(f)
    dfs.append(df)

merged_df = pd.concat(dfs, ignore_index=True)
merged_df = merged_df.sort_values('index').reset_index(drop=True)
merged_df.to_csv("$MERGED_CSV", index=False)
print(f"Merged CSV saved to: $MERGED_CSV")

# Merge JSON files
json_files = sorted(glob.glob("$OUTPUT_DIR/results_part_*.json"))
all_records = []
for f in json_files:
    with open(f, 'r') as jf:
        records = json.load(jf)
        all_records.extend(records)

# Sort by index
all_records.sort(key=lambda x: x.get('index', 0))
with open("$MERGED_JSON", 'w') as jf:
    json.dump(all_records, jf, indent=2)
print(f"Merged JSON saved to: $MERGED_JSON")

print(f"\nTotal records merged: {len(merged_df)}")
EOF
    else
        # Fallback: simple concatenation (won't be sorted)
        echo "Warning: pandas not available, using simple concatenation"
        head -1 "$OUTPUT_DIR/results_part_0.csv" > "$MERGED_CSV"
        for f in "$OUTPUT_DIR"/results_part_*.csv; do
            tail -n +2 "$f" >> "$MERGED_CSV"
        done
        echo "Merged CSV saved to: $MERGED_CSV (not sorted by index)"
    fi
    
    echo ""
    echo "=========================================="
    echo "Results:"
    echo "  Individual results: $OUTPUT_DIR/"
    echo "  Merged CSV: $MERGED_CSV"
    echo "  Merged JSON: $MERGED_JSON"
    echo "=========================================="
    
    # Clean up PID file
    rm -f "$PID_FILE"
else
    echo "$FAILED process(es) failed. Check logs in $OUTPUT_DIR/"
    echo "=========================================="
    echo "Note: Some processes may still be running. Check $PID_FILE for PIDs."
    exit 1
fi

