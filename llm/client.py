"""
LLM backend client — llama-server (local) and OpenRouter (cloud).

All callers import from here. The module-level names (_LLM_SEM, USE_CLOUD,
_call_llama_server, etc.) are the correct patch targets for unit tests.
"""
import json
import requests

from config import (
    _LLM_SEM,
    _LLAMA_SESSION,
    USE_CLOUD,
    OPENROUTER_API_KEY,
    OPENROUTER_URL,
    OPENROUTER_MODEL,
    LLAMA_CHAT_URL,
    LLAMA_HEALTH_URL,
    LLM_MODEL,
)


# ── Model helpers ─────────────────────────────────────────────────────────────

def get_model() -> str:
    """Return configured model alias. llama-server loads one model — no discovery needed."""
    return LLM_MODEL


def check_backend_available() -> dict:
    """Check llama-server health and return loaded model info."""
    try:
        r = _LLAMA_SESSION.get(LLAMA_HEALTH_URL, timeout=3)
        if r.status_code == 200:
            return {"available": True, "models": [LLM_MODEL]}
    except requests.RequestException:
        pass
    return {"available": False, "models": []}


def _model_timeout(model_name: str) -> int:
    """Return appropriate request timeout in seconds based on model size."""
    name = (model_name or "").lower()
    if "gemma4" in name:
        return 360
    if any(x in name for x in ("27b", "13b", "12b", "9b")):
        return 600
    if "7b" in name or "8b" in name:
        return 480
    # Known 7B aliases without size suffix
    if any(x in name for x in ("qwen2.5", "qwen3", "llama3", "mistral")):
        return 480
    if "4b" in name:
        return 240
    return 120   # tiny models (1b, tinyllama)


def _model_profile(model_name: str) -> dict:
    """
    Per-model tuning profile. Tokens/sec and JSON reliability differ by model,
    so max_tokens + timeout scale accordingly.
    """
    name = (model_name or "").lower()
    profile = {
        "predict_mult": 1.0,
        "timeout":      _model_timeout(model_name),
        "top_k":        1,
        "temperature":  0,
    }
    if "gemma4" in name:
        profile["top_k"]       = 40
        profile["temperature"] = 0.1
    if "gemma3" in name and "4b" in name:
        profile["predict_mult"] = 0.85
    if "1b" in name or "tinyllama" in name:
        profile["predict_mult"] = 0.6
    if "27b" in name or "13b" in name or "12b" in name:
        profile["predict_mult"] = 1.1
    return profile


# ── SSE stream parser (shared — eliminates duplicate logic) ──────────────────

def _parse_sse_stream(response):
    """
    Generator: yield text tokens from an OpenAI-compatible SSE response.
    Extracted from _stream_llama_server / _stream_openrouter to eliminate
    the duplicated 15-line parsing loop (DRY fix).
    """
    for line in response.iter_lines():
        if not line:
            continue
        decoded = line.decode("utf-8") if isinstance(line, bytes) else line
        if not decoded.startswith("data: "):
            continue
        data_str = decoded[6:].strip()
        if data_str == "[DONE]":
            return
        try:
            chunk = json.loads(data_str)
            token = chunk["choices"][0]["delta"].get("content", "")
            if token:
                yield token
        except (KeyError, json.JSONDecodeError):
            pass


# ── Streaming generators ──────────────────────────────────────────────────────

def _stream_llama_server(messages: list, model: str, timeout: int = 120):
    """Generator: yield text tokens from llama-server SSE stream (multi-turn)."""
    payload = {
        "model":       model,
        "messages":    messages,
        "max_tokens":  2048,
        "temperature": 0.7,
        "stream":      True,
    }
    with _LLAMA_SESSION.post(LLAMA_CHAT_URL, json=payload, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        yield from _parse_sse_stream(r)


def _stream_openrouter(messages: list, timeout: int = 120):
    """Generator: yield text tokens from OpenRouter SSE stream (multi-turn)."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "http://localhost:5000",
        "X-Title":       "Agile Suite — Chat",
    }
    payload = {
        "model":       OPENROUTER_MODEL,
        "messages":    messages,
        "max_tokens":  2048,
        "temperature": 0.7,
        "stream":      True,
    }
    # Use _LLAMA_SESSION for HTTPS pooling — avoids new TCP handshake per stream
    with _LLAMA_SESSION.post(OPENROUTER_URL, headers=headers, json=payload,
                             stream=True, timeout=timeout) as r:
        r.raise_for_status()
        yield from _parse_sse_stream(r)


# ── Non-streaming callers ────────────────────────────────────────────────────

def _call_openrouter(prompt: str, timeout: int = 60,
                     format_json: bool = False, max_tokens: int = 1500) -> str:
    """
    OpenRouter cloud inference — free tier Gemma, OpenAI-compatible.
    Sub-10s per artifact vs. 2-3 min local.
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "http://localhost:5000",
        "X-Title":       "Agile Suite — Project Docs",
    }
    payload = {
        "model":       OPENROUTER_MODEL,
        "messages":    [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens":  max_tokens,
    }
    if format_json:
        payload["response_format"] = {"type": "json_object"}
    r = _LLAMA_SESSION.post(OPENROUTER_URL, headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def _call_llama_server(model: str, prompt: str, timeout: int = None,
                       options_override: dict = None, format_json: bool = False) -> str:
    """
    Call llama-server via OpenAI-compatible /v1/chat/completions.
    Uses _LLAMA_SESSION connection pool (was using requests.post directly — bug fix).

    Inference tuning (context size, GPU layers, threads) are llama-server
    startup flags — not per-request. Only generation params sent here.
    """
    if timeout is None:
        timeout = _model_timeout(model)

    params = {
        "max_tokens":     1200,
        "temperature":    0,
        "top_k":          1,
        "top_p":          1.0,
        "repeat_penalty": 1.0,
    }
    if options_override:
        params.update(options_override)

    payload = {
        "model":    model,
        "messages": [{"role": "user", "content": prompt}],
        **params,
    }
    if format_json:
        payload["response_format"] = {"type": "json_object"}

    r = _LLAMA_SESSION.post(LLAMA_CHAT_URL, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def _call_llm(model: str, prompt: str, timeout: int = None,
              options_override: dict = None, format_json: bool = False) -> str:
    """
    Dispatch to OpenRouter if USE_CLOUD, else llama-server local.
    Semaphore-guarded: caps global concurrent LLM calls to _LLM_MAX_CONCURRENT
    matching llama-server --parallel slots to prevent VRAM thrash.
    """
    with _LLM_SEM:
        if USE_CLOUD:
            max_tok = (options_override or {}).get("max_tokens", 1500)
            t       = timeout or 60
            try:
                return _call_openrouter(prompt, timeout=t,
                                        format_json=format_json, max_tokens=max_tok)
            except Exception as e:
                print(f"[openrouter] failed ({e}), falling back to llama-server")
        return _call_llama_server(model, prompt, timeout=timeout,
                                  options_override=options_override,
                                  format_json=format_json)
