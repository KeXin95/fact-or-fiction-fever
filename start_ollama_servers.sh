#!/bin/bash

# Configuration
BASE_PORT=11434
NUM_SERVERS=8
OLLAMA_BIN="./ollama"
MODEL="qwen2:7b"
OLLAMA_CONTEXT_LENGTH=16384
OLLAMA_MODELS="./models/"

# CUDA device allocation
# Format: "device1,device2" for each server
# NOTE: If you only have devices 2,3 available, you can use the same for all servers
#       or distribute them. Ollama will handle the load balancing.
# Example distributions:
#   - All servers use same devices: all "2,3"
#   - Split devices: "2" for some, "3" for others
#   - Use all available: distribute across 0,1,2,3,4,5,6,7
CUDA_DEVICES=(
    "0"      # Server 0 (adjust based on your GPU setup)
    "1"      # Server 1
    "2"      # Server 2
    "3"      # Server 3
    "4"      # Server 4
    "5"      # Server 5
    "6"      # Server 6
    "7"
)

# Log directory
LOG_DIR="./NLP/ollama_servers"
mkdir -p "$LOG_DIR"

# Check CUDA devices if nvidia-smi is available
if command -v nvidia-smi &> /dev/null; then
    echo "Available CUDA Devices:"
    nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader | while IFS=, read -r idx name mem; do
        echo "  GPU $idx: $name (Free: $mem)"
    done
    echo ""
fi

echo "=========================================="
echo "Starting $NUM_SERVERS Ollama Servers"
echo "=========================================="
echo "Base port: $BASE_PORT"
echo "Model: $MODEL"
echo "Context length: $OLLAMA_CONTEXT_LENGTH"
echo "Models directory: $OLLAMA_MODELS"
echo "Log directory: $LOG_DIR"
echo "=========================================="
echo ""

# Array to store server PIDs
SERVER_PIDS=()

# Function to check if a port is already in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        return 1  # Port is in use
    else
        return 0  # Port is free
    fi
}

# Start each Ollama server
for i in $(seq 0 $((NUM_SERVERS - 1))); do
    PORT=$((BASE_PORT + i))
    CUDA_DEVICE="${CUDA_DEVICES[$i]}"
    
    # Check if port is already in use
    if ! check_port $PORT; then
        echo "⚠️  Warning: Port $PORT is already in use. Skipping server $i."
        continue
    fi
    
    echo "Starting Ollama Server $i:"
    echo "  Port: $PORT"
    echo "  CUDA devices: $CUDA_DEVICE"
    echo "  Serve log: $LOG_DIR/ollama_serve_${i}.log"
    echo "  Model log: $LOG_DIR/qwen_${i}.log"
    
    # Set environment variables for this server
    export OLLAMA_HOST="http://localhost:${PORT}"
    export OLLAMA_CONTEXT_LENGTH=$OLLAMA_CONTEXT_LENGTH
    export OLLAMA_MODELS=$OLLAMA_MODELS
    export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE
    
    # Start Ollama serve in background with setsid (new session) and nohup
    # setsid creates a new session, making the process independent of the terminal
    setsid nohup $OLLAMA_BIN serve >> "$LOG_DIR/ollama_serve_${i}.log" 2>&1 &
    SERVE_PID=$!
    SERVER_PIDS+=($SERVE_PID)
    disown $SERVE_PID 2>/dev/null || true  # Remove from shell job table
    
    # Wait a bit for server to start
    sleep 3
    
    # Start the model in background with setsid (new session) and nohup
    setsid nohup $OLLAMA_BIN run $MODEL >> "$LOG_DIR/qwen_${i}.log" 2>&1 &
    MODEL_PID=$!
    disown $MODEL_PID 2>/dev/null || true  # Remove from shell job table
    
    echo "  Serve PID: $SERVE_PID"
    echo "  Model PID: $MODEL_PID"
    echo ""
    
    # Small delay between starting servers
    sleep 2
done

echo "=========================================="
echo "All servers started!"
echo "=========================================="
echo "Server PIDs: ${SERVER_PIDS[@]}"
echo ""
echo "Ports in use:"
for i in $(seq 0 $((NUM_SERVERS - 1))); do
    PORT=$((BASE_PORT + i))
    if check_port $PORT; then
        echo "  Port $PORT: ❌ Not listening"
    else
        echo "  Port $PORT: ✅ Active"
    fi
done
echo ""
echo "Monitor logs with:"
echo "  tail -f $LOG_DIR/ollama_serve_*.log"
echo "  tail -f $LOG_DIR/qwen_*.log"
echo ""
echo "To stop all servers, run:"
echo "  pkill -f 'ollama serve'"
echo "  pkill -f 'ollama run'"
echo "=========================================="

