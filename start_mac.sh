#!/bin/bash

# =====================================================
#   PII Shield - Sensitive Data Masker
#   Mac Startup Script (Apple Silicon)
#   Powered by llama.cpp (llama-server) — 100% Offline
# =====================================================

set -e  # Exit on error

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_ALIAS="${LLM_MODEL:-qwen2.5}"
MODEL_FILE="${LLAMA_MODEL_PATH:-$HOME/models/qwen2.5-7b-instruct-q4_K_M.gguf}"
LLAMA_PORT=8080
LLAMA_PARALLEL="${LLM_MAX_CONCURRENT:-1}"   # dev: 1 = full bandwidth single-user; prod: set LLM_MAX_CONCURRENT=8
LLAMA_CTX=16384

echo ""
echo -e "${CYAN}====================================================="
echo "   PII Shield — Sensitive Data Masker"
echo "   Powered by llama-server (llama.cpp) — 100% Offline"
echo -e "=====================================================${NC}"
echo ""

# ── Detect chip type ──────────────────────────────────────────────────────────
CHIP=$(uname -m)
if [ "$CHIP" = "arm64" ]; then
    echo -e "${GREEN}[INFO] Apple Silicon detected — Metal GPU backend active ⚡${NC}"
else
    echo -e "${YELLOW}[INFO] Intel Mac detected — CPU inference only${NC}"
fi
echo ""

# ── Check Homebrew ────────────────────────────────────────────────────────────
if ! command -v brew &> /dev/null; then
    echo -e "${YELLOW}[WARN] Homebrew not found. Installing...${NC}"
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    if [ "$CHIP" = "arm64" ]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
fi

# ── Check Python 3 ────────────────────────────────────────────────────────────
if ! command -v python3 &> /dev/null; then
    echo -e "${YELLOW}[INFO] Python3 not found. Installing via Homebrew...${NC}"
    brew install python3
fi
echo -e "${GREEN}[INFO] Python: $(python3 --version 2>&1)${NC}"

# ── Check / Install llama.cpp ─────────────────────────────────────────────────
if ! command -v llama-server &> /dev/null; then
    echo -e "${YELLOW}[INFO] llama-server not found. Installing via Homebrew...${NC}"
    brew install llama.cpp
    echo -e "${GREEN}[INFO] llama-server installed ✓${NC}"
else
    echo -e "${GREEN}[INFO] llama-server found: $(llama-server --version 2>&1 | head -1)${NC}"
fi

# ── Check model file ──────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}[INFO] Model file: $MODEL_FILE${NC}"
if [ ! -f "$MODEL_FILE" ]; then
    echo -e "${RED}[ERROR] Model file not found: $MODEL_FILE${NC}"
    echo ""
    echo "  Option A — extract from existing Ollama install:"
    echo "    ollama show --modelfile qwen2.5:7b-instruct-q4_K_M"
    echo "    # copy the blob path shown after 'FROM' to $MODEL_FILE"
    echo ""
    echo "  Option B — download from HuggingFace:"
    echo "    mkdir -p ~/models"
    echo "    huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF \\"
    echo "      qwen2.5-7b-instruct-q4_k_m.gguf --local-dir ~/models"
    echo ""
    echo "  Then re-run: ./start_mac.sh"
    exit 1
fi
echo -e "${GREEN}[INFO] Model file found ✓${NC}"

# ── Start llama-server ────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}[INFO] Starting llama-server in background...${NC}"

# Kill any existing llama-server on our port
pkill -f "llama-server" 2>/dev/null || true
sleep 1

llama-server \
    --model "$MODEL_FILE" \
    --alias "$MODEL_ALIAS" \
    --n-gpu-layers 999 \
    --ctx-size "$LLAMA_CTX" \
    --parallel "$LLAMA_PARALLEL" \
    --port "$LLAMA_PORT" \
    --host 127.0.0.1 \
    > /tmp/llama_server_pii.log 2>&1 &
LLAMA_PID=$!
echo -e "${GREEN}[INFO] llama-server started (PID: $LLAMA_PID)${NC}"

# Wait for llama-server to be ready
echo -e "${CYAN}[INFO] Waiting for llama-server to load model...${NC}"
for i in {1..30}; do
    if curl -s http://localhost:$LLAMA_PORT/health 2>/dev/null | grep -q "ok"; then
        echo -e "${GREEN}[INFO] llama-server ready ✓${NC}"
        break
    fi
    sleep 2
    if [ $i -eq 30 ]; then
        echo -e "${YELLOW}[WARN] llama-server slow to start — app will run regex-only mode${NC}"
        echo -e "${YELLOW}       Check logs: tail -f /tmp/llama_server_pii.log${NC}"
    fi
done

# ── Install Python dependencies ───────────────────────────────────────────────
echo ""
echo -e "${CYAN}[INFO] Installing Python dependencies...${NC}"

VENV_DIR="$(dirname "$0")/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${CYAN}[INFO] Creating virtual environment...${NC}"
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
pip install -q --upgrade pip
pip install -q -r "$(dirname "$0")/requirements.txt"
echo -e "${GREEN}[INFO] Dependencies ready ✓${NC}"

# ── Launch app ────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}====================================================="
echo "   PII Shield is starting..."
echo "   Open your browser at: http://localhost:5000"
echo "   Model: $MODEL_ALIAS | Parallel slots: $LLAMA_PARALLEL"
echo ""
echo "   Press Ctrl+C to stop"
echo -e "=====================================================${NC}"
echo ""

# Open browser after 2 seconds
(sleep 2 && open http://localhost:5000) &

# Start Flask app
cd "$(dirname "$0")"
python3 app.py

# Cleanup on exit
echo ""
echo -e "${YELLOW}[INFO] Shutting down llama-server...${NC}"
kill $LLAMA_PID 2>/dev/null || pkill -f "llama-server" 2>/dev/null || true
echo -e "${GREEN}[INFO] Goodbye!${NC}"
