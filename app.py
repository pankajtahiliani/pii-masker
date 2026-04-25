"""
PII Masker — Agile Suite
Flask application entry point.

Architecture: thin orchestrator — creates Flask app, registers Blueprints,
and hosts the two shared routes (/api/status, /api/models) that span modules.

Modules:
  modules/pii/           — PII detection & masking
  modules/project_docs/  — Agile artifact generation (backlog, sprints, etc.)
  modules/call_summarizer/ — Call transcript analysis
  modules/chat/          — Freeform AI chat (streaming, stop, edit)
  llm/client.py          — LLM backend dispatch (llama-server + OpenRouter)
  config.py              — Shared env config, semaphore, connection pool, rate limiter
"""

import os
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS

# ── Shared infrastructure ─────────────────────────────────────────────────────
from config import (
    USE_CLOUD, OPENROUTER_MODEL,
    LLM_TIMEOUT, LLM_CHUNK,
    _LLM_MAX_CONCURRENT, _LLM_SEM,
    _HAS_LIMITER, limiter, rate_limit,
    _LLAMA_SESSION, LLAMA_MODELS_URL,
)
from llm.client import (
    get_model, check_backend_available,
    _call_llm, _call_llama_server, _call_openrouter,
    _stream_llama_server, _stream_openrouter,
    _model_profile, _model_timeout,
)
from modules.project_docs.parser import (
    _parse_json_response, _close_truncated_json, _extract_complete_objects,
)
from modules.chat.session import _CHAT_SESSIONS, _CHAT_SYSTEM_PROMPT

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder='static')
CORS(app)

if limiter is not None:
    limiter.init_app(app)

# ── Register Blueprints ───────────────────────────────────────────────────────
from modules.pii.routes import pii_bp
from modules.project_docs.routes import project_docs_bp
from modules.call_summarizer.routes import call_summarizer_bp
from modules.chat.routes import chat_bp

app.register_blueprint(pii_bp)
app.register_blueprint(project_docs_bp)
app.register_blueprint(call_summarizer_bp)
app.register_blueprint(chat_bp)


# ── Shared routes ─────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/api/status')
def status():
    backend_info = check_backend_available()
    active = OPENROUTER_MODEL if USE_CLOUD else get_model()
    return jsonify({
        "status":             "ok",
        "backend":            "openrouter" if USE_CLOUD else "llamacpp",
        "llama_server":       backend_info,
        "active_model":       active,
        "cloud_enabled":      USE_CLOUD,
        "max_file_size_mb":   50,
        "ai_timeout_seconds": LLM_TIMEOUT,
        "chunk_size":         LLM_CHUNK,
    })


@app.route('/api/models')
def list_models():
    """Return loaded llama-server models with profile hints for the UI dropdown."""
    try:
        r = _LLAMA_SESSION.get(LLAMA_MODELS_URL, timeout=3)
        if r.status_code != 200:
            return jsonify({"models": [], "default": None,
                            "error": "llama-server unreachable"}), 200
        raw = [m.get("id", "") for m in r.json().get("data", []) if m.get("id")]
        enriched = []
        for name in raw:
            prof = _model_profile(name)
            enriched.append({
                "name":         name,
                "timeout":      prof["timeout"],
                "predict_mult": prof["predict_mult"],
            })
        default = get_model()
        return jsonify({"models": enriched, "default": default})
    except Exception as e:
        return jsonify({"models": [], "default": None, "error": str(e)}), 200


# ── Dev server entry point ────────────────────────────────────────────────────
if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    print("🚀 Agile Suite starting on http://localhost:5000")
    print("🔒 LOCAL MODE — all inference via llama-server, zero data leaves device")
    print("   Run: llama-server --model ~/models/qwen2.5-7b-instruct-q4_K_M.gguf \\")
    print("        --alias qwen2.5 --n-gpu-layers 999 --ctx-size 16384 --parallel 4 --port 8080")
    print(f"   Model: {get_model()} | Max concurrent: {_LLM_MAX_CONCURRENT}")
    if USE_CLOUD:
        print(f"⚠️  CLOUD OPT-IN ACTIVE — OpenRouter ({OPENROUTER_MODEL})")
    # Production: gunicorn --workers 1 --threads 16 --timeout 600 \
    #             --worker-class gthread --bind 0.0.0.0:5000 app:app
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
