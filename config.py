"""
Shared configuration, infrastructure, and rate-limiting helpers.

All modules import from here — never from app.py — to avoid circular deps.
"""
import os
import threading
import requests
from requests.adapters import HTTPAdapter

# ── llama-server (llama.cpp) — OpenAI-compatible endpoint ───────────────────
_LLAMA_BASE      = os.environ.get("LLAMA_SERVER_URL", "http://localhost:8080")
LLAMA_CHAT_URL   = f"{_LLAMA_BASE}/v1/chat/completions"
LLAMA_MODELS_URL = f"{_LLAMA_BASE}/v1/models"
LLAMA_HEALTH_URL = f"{_LLAMA_BASE}/health"

# ── Optional cloud inference (OpenRouter) ───────────────────────────────────
# LOCAL-ONLY by default. Privacy = value prop. Cloud would send PII off-device.
# Opt-in only: export PII_CLOUD_OPT_IN=1 OPENROUTER_API_KEY=sk-or-v1-...
_CLOUD_OPT_IN      = os.environ.get("PII_CLOUD_OPT_IN", "").strip() == "1"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip() if _CLOUD_OPT_IN else ""
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL   = os.environ.get("OPENROUTER_MODEL", "google/gemma-3-27b-it:free")
USE_CLOUD          = bool(OPENROUTER_API_KEY)

# ── LLM options (for PII NER — short, deterministic) ────────────────────────
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5")
LLM_TIMEOUT = 15    # hard timeout per chunk — 15s plenty with num_ctx=512
LLM_CHUNK   = 500   # smaller prompt = ~3x faster response
LLM_OPTIONS = {
    "max_tokens":     150,
    "temperature":    0,
    "top_k":          1,
    "top_p":          1.0,
    "repeat_penalty": 1.0,
}

# ── File storage ─────────────────────────────────────────────────────────────
MAX_FILE_SIZE = 50 * 1024 * 1024
# Fixed shared dir — all gunicorn threads use same path, no cross-worker miss.
UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", "/tmp/pii_uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Concurrency guardrails ───────────────────────────────────────────────────
# Semaphore caps concurrent LLM calls to match llama-server --parallel slots.
# Set LLM_MAX_CONCURRENT = --parallel value on the server (default 4 Mac, 8 GPU).
_LLM_MAX_CONCURRENT = int(os.environ.get("LLM_MAX_CONCURRENT", "1"))
_LLM_SEM = threading.BoundedSemaphore(_LLM_MAX_CONCURRENT)

# ── Persistent HTTP connection pool ─────────────────────────────────────────
# Reuses TCP connections to llama-server and OpenRouter — avoids handshake
# overhead on every LLM call. pool_maxsize = parallel slots + headroom.
_LLAMA_SESSION = requests.Session()
_LLAMA_SESSION.mount('http://', HTTPAdapter(
    pool_connections=4,
    pool_maxsize=_LLM_MAX_CONCURRENT + 2,
    max_retries=0,
))
_LLAMA_SESSION.mount('https://', HTTPAdapter(
    pool_connections=2,
    pool_maxsize=4,
    max_retries=0,
))

# ── Rate limiter (flask-limiter, optional) ───────────────────────────────────
# Uses init_app pattern so Limiter can be created before Flask app is created.
_HAS_LIMITER = False
limiter = None

try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    _HAS_LIMITER = True
    limiter = Limiter(
        get_remote_address,
        default_limits=["120 per minute", "2000 per hour"],
        storage_uri=os.environ.get("LIMITER_STORAGE", "memory://"),
    )
except ImportError:
    print("⚠️  flask-limiter not installed — rate limiting DISABLED. Run: pip install flask-limiter")


def rate_limit(spec):
    """No-op decorator when flask-limiter absent; real per-IP limit when present."""
    if _HAS_LIMITER:
        return limiter.limit(spec)
    return lambda f: f
