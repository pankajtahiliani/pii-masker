"""
PII detection — regex engine and LLM-assisted entity extraction.
"""
import re
import requests

from config import (
    LLAMA_CHAT_URL,
    LLM_OPTIONS,
    LLM_TIMEOUT,
    _LLAMA_SESSION,
)
from modules.pii.patterns import (
    REGEX_PATTERNS,
    CONTEXT_LABELS,
    NAME_PATTERNS,
    NOT_NAMES,
    HEADER_WORDS,
    DATE_PATTERN,
    HIGH_PRIORITY_TYPES,
)


def is_safe_to_mask(val: str, pii_type: str) -> bool:
    """
    Returns False if the value should NOT be masked.
    Prevents false positives on column headers and date strings.
    """
    stripped = val.strip()

    # Bug Fix 1: never mask standalone column/header words
    if stripped in HEADER_WORDS:
        return False

    # Bug Fix 2: never treat date-formatted strings as passwords
    if pii_type == 'password' and DATE_PATTERN.match(stripped):
        return False

    # Never mask very short values (1-2 chars) unless specific types
    if len(stripped) <= 2 and pii_type not in ('in_aadhaar', 'uk_nino', 'us_ssn'):
        return False

    # Never mask placeholder/null values
    if stripped.upper() in ('N/A', 'TBD', 'NA', '-', '—', 'NONE', 'NULL', 'YES', 'NO'):
        return False

    return True


def detect_pii_with_regex(text: str) -> list:
    """
    Global PII detection — IN, AU, UK, US, CA, EU, UAE, SG + universal.
    Bug Fix 6: high-priority (longer) patterns run first to prevent
    short patterns (ZIP, bank acct) from fragmenting longer strings.
    """
    found = []

    # Pass 1: High-priority patterns first (phone, API keys, credentials)
    for pii_type in HIGH_PRIORITY_TYPES:
        pattern = REGEX_PATTERNS.get(pii_type)
        if not pattern:
            continue
        try:
            for match in re.finditer(pattern, text):
                val = match.group().strip()
                if len(val) > 1 and is_safe_to_mask(val, pii_type):
                    found.append({"text": val, "type": pii_type, "source": "Regex"})
        except re.error:
            continue

    # Collect already-matched spans to avoid double-matching
    matched_spans = set()
    for item in found:
        for m in re.finditer(re.escape(item["text"]), text):
            matched_spans.add((m.start(), m.end()))

    # Pass 2: Remaining patterns (postcodes, bank accts, IDs etc.)
    for pii_type, pattern in REGEX_PATTERNS.items():
        if pii_type in HIGH_PRIORITY_TYPES:
            continue
        try:
            for match in re.finditer(pattern, text):
                val = match.group().strip()
                if len(val) <= 1:
                    continue
                span = (match.start(), match.end())
                overlaps = any(
                    not (span[1] <= s[0] or span[0] >= s[1])
                    for s in matched_spans
                )
                if overlaps:
                    continue
                if is_safe_to_mask(val, pii_type):
                    found.append({"text": val, "type": pii_type, "source": "Regex"})
                    matched_spans.add(span)
        except re.error:
            continue

    # Context-aware keyword: value extraction
    for ctx_pattern, (val_pattern, pii_type) in CONTEXT_LABELS.items():
        for ctx_match in re.finditer(ctx_pattern, text):
            rest = text[ctx_match.end():]
            val_match = re.match(val_pattern, rest.strip())
            if val_match:
                val = val_match.group().strip().rstrip(',;.')
                if is_safe_to_mask(val, pii_type):
                    found.append({"text": val, "type": pii_type, "source": "Regex"})

    # Proper name detection
    for pattern in NAME_PATTERNS:
        for match in re.finditer(pattern, text):
            words = match.group().strip().split()
            if any(w in NOT_NAMES for w in words):
                continue
            if len(words) < 2 or len(words) > 4:
                continue
            if all(w.isupper() for w in words):
                continue
            val = match.group().strip()
            if is_safe_to_mask(val, 'full_name'):
                found.append({"text": val, "type": "full_name", "source": "Regex"})

    return found


def detect_pii_with_llm(text: str, model: str) -> list:
    """
    Single chunk AI call with hard 15s timeout.
    Returns [] immediately on any error — regex results already saved.
    Uses _LLAMA_SESSION connection pool (was using requests.post directly — bug fix).
    """
    prompt = (
        "PII detector. Find: full names, usernames, physical addresses, gender. "
        "Return ONLY a JSON array. Each item: "
        '{"text": "exact string", "type": "name|username|address|gender"}\n\n'
        f"TEXT:\n{text}\n\nJSON:"
    )
    try:
        payload = {
            "model":       model,
            "messages":    [{"role": "user", "content": prompt}],
            "max_tokens":  LLM_OPTIONS["max_tokens"],
            "temperature": LLM_OPTIONS["temperature"],
            "top_k":       LLM_OPTIONS["top_k"],
            "top_p":       LLM_OPTIONS["top_p"],
        }
        resp = _LLAMA_SESSION.post(LLAMA_CHAT_URL, json=payload, timeout=LLM_TIMEOUT)
        if resp.status_code == 200:
            import json
            raw = resp.json()["choices"][0]["message"]["content"]
            m = re.search(r'\[.*?\]', raw, re.DOTALL)
            if m:
                entities = json.loads(m.group())
                return [e for e in entities
                        if isinstance(e, dict) and "text" in e and "type" in e
                        and len(e["text"].strip()) > 1]
    except requests.exceptions.Timeout:
        print(f"[AI] Timeout after {LLM_TIMEOUT}s — regex results only")
    except Exception as e:
        print(f"[AI] Error: {e} — regex results only")
    return []
