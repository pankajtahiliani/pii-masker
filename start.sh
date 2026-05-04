#!/bin/bash
# RunPod startup script — runs llama-server then Flask/gunicorn
# Set as Container Start Command in RunPod dashboard: bash /workspace/pii-masker/start.sh

set -e

REPO_DIR="/workspace/pii-masker"
MODEL_DIR="/workspace/models"
MODEL_PATH="${MODEL_DIR}/qwen2.5-7b-instruct-q4_k_m.gguf"
MODEL_ALIAS="qwen2.5"
LLAMA_PORT=8080
FLASK_PORT=5000

# ── 1. Pull latest code ───────────────────────────────────────────────────────
echo "=== Pulling latest code ==="
cd "${REPO_DIR}"
git pull origin main || echo "⚠️  git pull failed — running existing code"

# ── 2. Install / upgrade Python deps ─────────────────────────────────────────
echo "=== Installing dependencies ==="
pip install -q --ignore-installed -r "${REPO_DIR}/requirements.txt"

# ── 3. Install llama-server if missing ───────────────────────────────────────
if ! command -v llama-server &>/dev/null; then
    echo "=== Building llama-server (CUDA) — ~15 min first time ==="
    apt-get update -qq && apt-get install -y -qq cmake build-essential
    cd /tmp
    rm -rf llama.cpp
    git clone --depth 1 https://github.com/ggerganov/llama.cpp
    cd llama.cpp
    cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
    cmake --build build --config Release -j$(nproc)
    cp build/bin/llama-server /usr/local/bin/
    chmod +x /usr/local/bin/llama-server
    echo "llama-server installed ✓"
    cd "${REPO_DIR}"
else
    echo "llama-server already installed ✓"
fi

# ── 4. Ensure model directory exists ─────────────────────────────────────────
mkdir -p "${MODEL_DIR}"

# ── 4. Download model if missing ─────────────────────────────────────────────
if [ ! -f "${MODEL_PATH}" ]; then
    echo "=== Downloading Qwen2.5 7B Q4 model (~4.5 GB) ==="
    wget --show-progress \
        "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q3_k_m.gguf" \
        -O "${MODEL_PATH}"
    echo "Model downloaded ✓"
else
    echo "Model already exists ✓"
fi

# ── 5. Start llama-server ─────────────────────────────────────────────────────
echo "=== Starting llama-server on :${LLAMA_PORT} ==="
llama-server \
    --model   "${MODEL_PATH}" \
    --alias   "${MODEL_ALIAS}" \
    --n-gpu-layers 999 \
    --ctx-size    16384 \
    --parallel    4 \
    --port        ${LLAMA_PORT} \
    --host        0.0.0.0 \
    > /tmp/llama.log 2>&1 &

LLAMA_PID=$!
echo "llama-server PID: ${LLAMA_PID}"

# ── 6. Wait for llama-server ready ───────────────────────────────────────────
echo "Waiting for llama-server..."
for i in $(seq 1 40); do
    if curl -sf "http://localhost:${LLAMA_PORT}/health" | grep -q "ok"; then
        echo "llama-server ready ✓ (${i}s)"
        break
    fi
    if [ "$i" -eq 40 ]; then
        echo "❌ llama-server failed to start. Check /tmp/llama.log"
        cat /tmp/llama.log
        exit 1
    fi
    sleep 3
done

# ── 7. Start Flask via gunicorn ───────────────────────────────────────────────
echo "=== Starting Flask on :${FLASK_PORT} ==="
cd "${REPO_DIR}"
gunicorn \
    --workers     1 \
    --threads     16 \
    --timeout     600 \
    --worker-class gthread \
    --bind        0.0.0.0:${FLASK_PORT} \
    --access-logfile /tmp/gunicorn-access.log \
    --error-logfile  /tmp/gunicorn-error.log \
    app:app &

GUNICORN_PID=$!
echo "gunicorn PID: ${GUNICORN_PID}"

echo ""
echo "✅ All services running"
echo "   Flask  → http://0.0.0.0:${FLASK_PORT}"
echo "   llama  → http://0.0.0.0:${LLAMA_PORT}"
echo ""

# Keep container alive — stream Flask logs to stdout
tail -f /tmp/gunicorn-error.log /tmp/llama.log
