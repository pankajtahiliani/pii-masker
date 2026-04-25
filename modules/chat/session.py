"""
Chat session store — thread-safe, TTL-evicting, capacity-capped.

Design:
  - In-memory only (ephemeral by design — privacy).
  - Sessions expire after CHAT_SESSION_TTL seconds of inactivity (default 1h).
  - Hard cap at CHAT_SESSION_MAX concurrent sessions (default 500) — DoS guard.
  - All mutations lock individually (no long-held locks across LLM calls).
"""
import os
import threading
import time


_SESSION_TTL = int(os.environ.get("CHAT_SESSION_TTL", "3600"))   # seconds of inactivity
_SESSION_MAX = int(os.environ.get("CHAT_SESSION_MAX", "500"))    # max concurrent sessions


class _SessionStore:
    """
    Dict-like store with TTL eviction and capacity cap.

    Each entry: {"msgs": [{"role", "content"}, ...], "ts": float}
    All methods are atomic under self._lock.
    """

    def __init__(self, ttl: int = _SESSION_TTL, max_sessions: int = _SESSION_MAX):
        self._store: dict = {}
        self._ttl  = ttl
        self._max  = max_sessions
        self._lock = threading.Lock()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _evict(self) -> None:
        """Remove sessions idle longer than TTL. Call while holding _lock."""
        cutoff  = time.time() - self._ttl
        expired = [k for k, v in self._store.items() if v["ts"] < cutoff]
        for k in expired:
            del self._store[k]

    def _touch(self, key: str) -> None:
        """Refresh last-access time. Call while holding _lock."""
        if key in self._store:
            self._store[key]["ts"] = time.time()

    # ── Public interface ──────────────────────────────────────────────────────

    def __setitem__(self, key: str, messages: list) -> None:
        """Directly set the message list for a session (test helper / compat)."""
        with self._lock:
            self._evict()
            if key not in self._store and len(self._store) >= self._max:
                raise MemoryError(f"Session limit ({self._max}) reached")
            self._store[key] = {"msgs": list(messages), "ts": time.time()}

    def __contains__(self, key: str) -> bool:
        with self._lock:
            self._evict()
            return key in self._store

    def __delitem__(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def __getitem__(self, key: str) -> list:
        """Return a shallow copy of messages. Raises KeyError if session not found."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                raise KeyError(key)
            self._touch(key)
            return list(entry["msgs"])

    def get(self, key: str, default=None) -> list:
        """Return a shallow copy of messages, or default if not found."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return [] if default is None else default
            self._touch(key)
            return list(entry["msgs"])

    def append_message(self, session_id: str, message: dict) -> None:
        """Append message, creating session if needed. Raises MemoryError if at cap."""
        with self._lock:
            self._evict()
            if session_id not in self._store:
                if len(self._store) >= self._max:
                    raise MemoryError(
                        f"Session limit ({self._max}) reached — try again later"
                    )
                self._store[session_id] = {"msgs": [], "ts": time.time()}
            entry = self._store[session_id]
            entry["msgs"].append(message)
            entry["ts"] = time.time()

    def pop_last_user(self, session_id: str) -> bool:
        """Remove last message if it is a user message. Returns True if removed."""
        with self._lock:
            entry = self._store.get(session_id)
            if entry and entry["msgs"] and entry["msgs"][-1]["role"] == "user":
                entry["msgs"].pop()
                return True
        return False

    def truncate(self, session_id: str, keep: int) -> None:
        """Keep only first `keep` messages — used by Edit to discard subsequent turns."""
        with self._lock:
            entry = self._store.get(session_id)
            if entry:
                entry["msgs"] = entry["msgs"][:keep]
                entry["ts"]   = time.time()

    def clear(self) -> None:
        """Wipe all sessions — used by tests only."""
        with self._lock:
            self._store.clear()


# Module-level singleton — all routes share one store.
_CHAT_SESSIONS = _SessionStore()

_CHAT_SYSTEM_PROMPT = (
    "You are a helpful AI assistant for software teams. "
    "Help with requirements analysis, effort estimation, user story writing, "
    "risk analysis, Q&A, and any other software-related needs. "
    "Be concise, structured, and practical in your responses."
)
