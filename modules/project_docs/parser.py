"""
JSON repair and extraction pipeline for LLM artifact responses.
Handles truncated output, markdown fences, and nested wrapper objects.
"""
import json
import re

from modules.project_docs.prompts import _ARTIFACT_ARRAY_KEYS


def _close_truncated_json(text: str) -> str:
    """Add missing closing brackets/braces to truncated JSON string.
    Handles mid-string truncation by stripping the incomplete string token."""
    stack = []
    in_string = False
    escape = False
    last_closed_pos = 0
    string_start = -1

    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"':
            if in_string:
                in_string = False
                string_start = -1
                if not stack:
                    last_closed_pos = i + 1
            else:
                in_string = True
                string_start = i
            continue
        if in_string:
            continue
        if ch in ('{', '['):
            stack.append('}' if ch == '{' else ']')
        elif ch in ('}', ']'):
            if stack and stack[-1] == ch:
                stack.pop()
                if not stack:
                    last_closed_pos = i + 1

    if in_string:
        if string_start > 0:
            prefix = text[:string_start].rstrip().rstrip(',').rstrip()
            if prefix.endswith(':'):
                end_quote = prefix.rfind('"', 0, len(prefix) - 1)
                if end_quote > 0:
                    start_quote = prefix.rfind('"', 0, end_quote)
                    if start_quote > 0:
                        prefix = prefix[:start_quote].rstrip().rstrip(',').rstrip()
            if prefix:
                return _close_truncated_json(prefix)
        if last_closed_pos > 0:
            return _close_truncated_json(text[:last_closed_pos])
        return text
    if not stack:
        return text
    return text + ''.join(reversed(stack))


def _extract_complete_objects(text: str, array_key: str):
    """Last-resort: scan text for complete {...} objects inside a named array.
    Works even when outer structure is truncated. Returns {array_key: [obj,...]} or None."""
    m = re.search(rf'"{re.escape(array_key)}"\s*:\s*\[', text)
    if not m:
        return None
    pos = m.end()
    objects = []
    depth = 0
    in_string = False
    escape = False
    obj_start = None

    for i in range(pos, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            if depth == 0:
                obj_start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and obj_start is not None:
                try:
                    obj = json.loads(text[obj_start:i + 1])
                    objects.append(obj)
                except json.JSONDecodeError:
                    pass
                obj_start = None
        elif ch == ']' and depth == 0:
            break

    if objects:
        print(f"[parse] extracted {len(objects)} complete objects from '{array_key}' array")
        return {array_key: objects}
    return None


def _parse_json_response(raw: str, artifact_key: str):
    """Best-effort JSON extraction with truncation repair. Returns a dict/list or _raw fallback."""
    print(f"[parse] {artifact_key}: raw_len={len(raw)} first80={repr(raw[:80])}")
    cleaned = raw.strip()

    # Strip markdown fences
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:])
        if cleaned.lstrip().startswith("json"):
            cleaned = cleaned.lstrip()[4:]
    if cleaned.endswith("```"):
        cleaned = "\n".join(cleaned.split("\n")[:-1])
    cleaned = cleaned.strip()

    # Unwrap single-key wrappers e.g. {"risks":[...]} → list
    def _unwrap(obj):
        if not isinstance(obj, dict):
            return obj
        for k in (artifact_key, 'risks', 'epics', 'sprints', 'reviews',
                  'test_cases', 'items', 'data', 'backlog'):
            if k in obj and len(obj) == 1:
                return obj[k]
        return obj

    def _try(fragment):
        try:
            return _unwrap(json.loads(fragment))
        except json.JSONDecodeError:
            return None

    # Pass 1: direct parse of outermost { } or [ ]
    for start_ch, end_ch in [('{', '}'), ('[', ']')]:
        s = cleaned.find(start_ch)
        e = cleaned.rfind(end_ch)
        if s != -1 and e != -1 and e > s:
            result = _try(cleaned[s:e + 1])
            if result is not None:
                print(f"[parse] {artifact_key}: direct parse OK")
                return result

    # Pass 2: close truncated JSON then parse
    for start_ch in ('{', '['):
        s = cleaned.find(start_ch)
        if s != -1:
            repaired = _close_truncated_json(cleaned[s:])
            result = _try(repaired)
            if result is not None:
                print(f"[parse] {artifact_key}: truncation-close repaired OK")
                return result

    # Pass 3: extract complete objects from named array
    array_key_name = _ARTIFACT_ARRAY_KEYS.get(artifact_key)
    if array_key_name:
        result = _extract_complete_objects(cleaned, array_key_name)
        if result is not None:
            return _unwrap(result)

    print(f"[parse] {artifact_key}: all passes failed → _raw")
    return {"_raw": cleaned}
