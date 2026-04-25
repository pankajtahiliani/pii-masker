"""
Chat session store — in-memory only, intentionally ephemeral (privacy).
Lost on server restart by design.
"""

# {session_id: [{role, content}, ...]}
_CHAT_SESSIONS: dict = {}

_CHAT_SYSTEM_PROMPT = (
    "You are a helpful AI assistant for software teams. "
    "Help with requirements analysis, effort estimation, user story writing, "
    "risk analysis, Q&A, and any other software-related needs. "
    "Be concise, structured, and practical in your responses."
)
