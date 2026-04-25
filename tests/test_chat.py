"""
Chat module — comprehensive unit tests. No LLM calls, no network.

Coverage:
  Input validation        — empty/whitespace/missing fields on all endpoints
  Session lifecycle       — create, restore, delete, isolation between IDs
  SSE streaming           — format, token content, [DONE] terminator
  Multi-turn history      — system prompt first, all prior turns passed
  Error handling          — LLM error cleans up orphaned user message
  Empty LLM response      — zero-token response cleans up user message (Bug 1)
  Stop endpoint           — removes orphaned user msg, no-op on completed session
  Truncate endpoint       — correct slice, zero, negative clamped, invalid type (Bugs 2&3)
  Edit workflow e2e       — truncate + re-send rebuilds session correctly
  Download DOCX           — rejects empty session, returns DOCX with content
  Cloud path              — routes to _stream_openrouter, not _stream_llama_server
  Rate limit              — @rate_limit applied to chat_stream
  Static asset            — chat.html uses d.llama_server (not d.ollama)
  GET /chat route         — serves HTML page

Patch targets after modularisation:
  modules.chat.routes._stream_llama_server  — local stream mock
  modules.chat.routes._stream_openrouter    — cloud stream mock
  modules.chat.routes.USE_CLOUD             — backend selector
  modules.chat.session._CHAT_SESSIONS       — session dict (also re-exported via app)
"""
import json
import os
import sys
import unittest
from unittest.mock import patch, MagicMock, call

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers

def _sse_tokens(body: str) -> list[str]:
    tokens = []
    for line in body.split('\n'):
        line = line.strip()
        if not line.startswith('data: '):
            continue
        raw = line[6:].strip()
        if raw in ('[DONE]', ''):
            continue
        try:
            obj = json.loads(raw)
            if 'token' in obj:
                tokens.append(obj['token'])
        except json.JSONDecodeError:
            pass
    return tokens


def _sse_error(body: str):
    for line in body.split('\n'):
        line = line.strip()
        if not line.startswith('data: '):
            continue
        raw = line[6:].strip()
        if raw in ('[DONE]', ''):
            continue
        try:
            obj = json.loads(raw)
            if 'error' in obj:
                return obj['error']
        except json.JSONDecodeError:
            pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 1. Input validation
# ─────────────────────────────────────────────────────────────────────────────

class ChatStreamInputTests(unittest.TestCase):
    def setUp(self):
        import app
        from modules.chat.session import _CHAT_SESSIONS
        self.client = app.app.test_client()
        _CHAT_SESSIONS.clear()

    def test_01_empty_message_rejected(self):
        """Empty message string → 400."""
        resp = self.client.post('/api/chat/stream',
                                json={'session_id': 'sid', 'message': ''})
        self.assertEqual(resp.status_code, 400)

    def test_02_whitespace_message_rejected(self):
        """Whitespace-only message → 400."""
        resp = self.client.post('/api/chat/stream',
                                json={'session_id': 'sid', 'message': '   \t\n  '})
        self.assertEqual(resp.status_code, 400)

    def test_03_missing_session_id_rejected(self):
        """No session_id field → 400."""
        resp = self.client.post('/api/chat/stream', json={'message': 'hello'})
        self.assertEqual(resp.status_code, 400)

    def test_04_empty_session_id_rejected(self):
        """Empty string session_id → 400."""
        resp = self.client.post('/api/chat/stream',
                                json={'session_id': '', 'message': 'hello'})
        self.assertEqual(resp.status_code, 400)

    def test_05_null_body_rejected(self):
        """Null / missing JSON body → 400 on both message and session_id."""
        resp = self.client.post('/api/chat/stream',
                                data='', content_type='application/json')
        self.assertEqual(resp.status_code, 400)

    def test_06_whitespace_session_id_rejected(self):
        """session_id that strips to empty → 400."""
        resp = self.client.post('/api/chat/stream',
                                json={'session_id': '   ', 'message': 'hi'})
        self.assertEqual(resp.status_code, 400)


# ─────────────────────────────────────────────────────────────────────────────
# 2. SSE streaming — format and content
# ─────────────────────────────────────────────────────────────────────────────

class ChatStreamFormatTests(unittest.TestCase):
    def setUp(self):
        import app
        from modules.chat.session import _CHAT_SESSIONS
        self.client = app.app.test_client()
        _CHAT_SESSIONS.clear()

    def test_07_sse_lines_start_with_data(self):
        """Every non-blank SSE line must start with 'data: '."""
        import modules.chat.routes as chat_routes
        with patch.object(chat_routes, '_stream_llama_server', return_value=iter(['Hi'])), \
             patch.object(chat_routes, 'USE_CLOUD', False):
            body = self.client.post('/api/chat/stream',
                                    json={'session_id': 's', 'message': 'x'}).get_data(as_text=True)
        for line in body.split('\n'):
            if line.strip():
                self.assertTrue(line.startswith('data: '),
                                f"Non-blank SSE line missing 'data: ' prefix: {line!r}")

    def test_08_sse_contains_done_terminator(self):
        """SSE stream must end with data: [DONE]."""
        import modules.chat.routes as chat_routes
        with patch.object(chat_routes, '_stream_llama_server', return_value=iter(['tok'])), \
             patch.object(chat_routes, 'USE_CLOUD', False):
            body = self.client.post('/api/chat/stream',
                                    json={'session_id': 's2', 'message': 'x'}).get_data(as_text=True)
        self.assertIn('data: [DONE]', body)

    def test_09_sse_token_key_present(self):
        """Each non-DONE SSE event must have a 'token' key."""
        import modules.chat.routes as chat_routes
        with patch.object(chat_routes, '_stream_llama_server', return_value=iter(['A', 'B'])), \
             patch.object(chat_routes, 'USE_CLOUD', False):
            body = self.client.post('/api/chat/stream',
                                    json={'session_id': 's3', 'message': 'x'}).get_data(as_text=True)
        self.assertEqual(_sse_tokens(body), ['A', 'B'])

    def test_10_content_type_is_text_event_stream(self):
        """Response Content-Type must be text/event-stream."""
        import modules.chat.routes as chat_routes
        with patch.object(chat_routes, '_stream_llama_server', return_value=iter(['t'])), \
             patch.object(chat_routes, 'USE_CLOUD', False):
            resp = self.client.post('/api/chat/stream',
                                    json={'session_id': 's4', 'message': 'x'})
        self.assertIn('text/event-stream', resp.content_type)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Session lifecycle
# ─────────────────────────────────────────────────────────────────────────────

class ChatSessionLifecycleTests(unittest.TestCase):
    def setUp(self):
        import app
        from modules.chat.session import _CHAT_SESSIONS
        self.app     = app
        self.client  = app.app.test_client()
        self.sessions = _CHAT_SESSIONS
        _CHAT_SESSIONS.clear()

    def _stream(self, sid, msg, tokens):
        import modules.chat.routes as chat_routes
        with patch.object(chat_routes, '_stream_llama_server', return_value=iter(tokens)), \
             patch.object(chat_routes, 'USE_CLOUD', False):
            return self.client.post('/api/chat/stream',
                                    json={'session_id': sid, 'message': msg}).get_data()

    def test_11_session_created_after_stream(self):
        """Session entry created after first successful stream."""
        self._stream('s1', 'hello', ['Hi'])
        msgs = self.sessions['s1']
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]['role'], 'user')
        self.assertEqual(msgs[1]['role'], 'assistant')

    def test_12_session_stores_exact_content(self):
        """Session stores exact user and assistant text."""
        txt = 'exact text 12345'
        self._stream('exact', txt, [txt[:4], txt[4:]])
        self.assertEqual(self.sessions['exact'][0]['content'], txt)

    def test_13_sessions_are_isolated(self):
        """Two different session IDs don't share messages."""
        self._stream('alpha', 'msg-alpha', ['resp-alpha'])
        self._stream('beta',  'msg-beta',  ['resp-beta'])
        self.assertEqual(self.sessions['alpha'][0]['content'], 'msg-alpha')
        self.assertEqual(self.sessions['beta'][0]['content'],  'msg-beta')
        self.assertNotIn('msg-beta',  str(self.sessions['alpha']))
        self.assertNotIn('msg-alpha', str(self.sessions['beta']))

    def test_14_multi_turn_session_grows(self):
        """Each request adds user + assistant = +2 messages."""
        self._stream('mt', 'turn1', ['resp1'])
        self._stream('mt', 'turn2', ['resp2'])
        msgs = self.sessions['mt']
        self.assertEqual(len(msgs), 4)

    def test_15_delete_clears_session(self):
        """DELETE /api/chat/session removes the session entry."""
        self.sessions['del'] = [{'role': 'user', 'content': 'hi'}]
        self.client.delete('/api/chat/session', json={'session_id': 'del'})
        self.assertNotIn('del', self.sessions)

    def test_16_session_count_correct_after_multi_turn(self):
        """3 turns → 6 messages (3 user + 3 assistant)."""
        for i in range(3):
            self._stream('cnt', f'turn{i}', [f'resp{i}'])
        self.assertEqual(len(self.sessions['cnt']), 6)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Multi-turn history passed to LLM
# ─────────────────────────────────────────────────────────────────────────────

class ChatMultiTurnTests(unittest.TestCase):
    def setUp(self):
        import app
        from modules.chat.session import _CHAT_SESSIONS
        self.app     = app
        self.client  = app.app.test_client()
        self.sessions = _CHAT_SESSIONS
        _CHAT_SESSIONS.clear()

    def _stream(self, sid, msg, tokens):
        import modules.chat.routes as chat_routes
        with patch.object(chat_routes, '_stream_llama_server', return_value=iter(tokens)), \
             patch.object(chat_routes, 'USE_CLOUD', False):
            return self.client.post('/api/chat/stream',
                                    json={'session_id': sid, 'message': msg}).get_data()

    def test_17_system_prompt_is_first_message(self):
        """LLM receives system prompt as first message in every call."""
        from modules.chat.session import _CHAT_SYSTEM_PROMPT
        import modules.chat.routes as chat_routes
        with patch.object(chat_routes, '_stream_llama_server',
                          return_value=iter(['r'])) as mock, \
             patch.object(chat_routes, 'USE_CLOUD', False):
            self.client.post('/api/chat/stream',
                             json={'session_id': 'sp', 'message': 'hi'}).get_data()
            msgs_to_llm = mock.call_args[0][0]
        self.assertEqual(msgs_to_llm[0]['role'], 'system')

    def test_18_system_prompt_matches_config(self):
        """System prompt text matches _CHAT_SYSTEM_PROMPT constant."""
        from modules.chat.session import _CHAT_SYSTEM_PROMPT
        import modules.chat.routes as chat_routes
        with patch.object(chat_routes, '_stream_llama_server',
                          return_value=iter(['r'])) as mock, \
             patch.object(chat_routes, 'USE_CLOUD', False):
            self.client.post('/api/chat/stream',
                             json={'session_id': 'sp2', 'message': 'hi'}).get_data()
            msgs = mock.call_args[0][0]
        self.assertEqual(msgs[0]['content'], _CHAT_SYSTEM_PROMPT)

    def test_19_prior_turns_included_in_second_call(self):
        """Second LLM call includes prior user/assistant exchange."""
        import modules.chat.routes as chat_routes
        with patch.object(chat_routes, '_stream_llama_server',
                          return_value=iter(['A1'])), \
             patch.object(chat_routes, 'USE_CLOUD', False):
            self.client.post('/api/chat/stream',
                             json={'session_id': 'mt2', 'message': 'U1'}).get_data()

        with patch.object(chat_routes, '_stream_llama_server',
                          return_value=iter(['A2'])) as mock2, \
             patch.object(chat_routes, 'USE_CLOUD', False):
            self.client.post('/api/chat/stream',
                             json={'session_id': 'mt2', 'message': 'U2'}).get_data()
            msgs_to_llm = mock2.call_args[0][0]

        # system + U1 + A1 + U2 = 4 messages
        self.assertEqual(len(msgs_to_llm), 4)
        self.assertEqual(msgs_to_llm[1]['content'], 'U1')
        self.assertEqual(msgs_to_llm[2]['content'], 'A1')
        self.assertEqual(msgs_to_llm[3]['content'], 'U2')


# ─────────────────────────────────────────────────────────────────────────────
# 5. Error handling and session cleanup
# ─────────────────────────────────────────────────────────────────────────────

class ChatErrorHandlingTests(unittest.TestCase):
    def setUp(self):
        import app
        from modules.chat.session import _CHAT_SESSIONS
        self.client  = app.app.test_client()
        self.sessions = _CHAT_SESSIONS
        _CHAT_SESSIONS.clear()

    def test_20_llm_error_yields_error_sse_event(self):
        """LLM exception → error SSE event emitted before [DONE]."""
        import modules.chat.routes as chat_routes
        def boom(*a, **kw):
            raise RuntimeError("VRAM OOM")
            yield

        with patch.object(chat_routes, '_stream_llama_server', side_effect=boom), \
             patch.object(chat_routes, 'USE_CLOUD', False):
            body = self.client.post('/api/chat/stream',
                                    json={'session_id': 'e1', 'message': 'hi'}).get_data(as_text=True)
        self.assertIsNotNone(_sse_error(body))
        self.assertIn('[DONE]', body)

    def test_21_llm_error_removes_orphaned_user_message(self):
        """LLM error must clean up the user message appended before generation."""
        import modules.chat.routes as chat_routes
        def boom(*a, **kw):
            raise RuntimeError("timeout")
            yield

        with patch.object(chat_routes, '_stream_llama_server', side_effect=boom), \
             patch.object(chat_routes, 'USE_CLOUD', False):
            self.client.post('/api/chat/stream',
                             json={'session_id': 'e2', 'message': 'hi'}).get_data()
        self.assertEqual(len(self.sessions.get('e2', [])), 0)

    def test_22_llm_error_mid_stream_partial_tokens_lost(self):
        """If LLM raises after yielding some tokens, orphaned user msg is cleaned up."""
        import modules.chat.routes as chat_routes
        def partial_then_fail(*a, **kw):
            yield 'tok1'
            raise RuntimeError("connection reset")

        with patch.object(chat_routes, '_stream_llama_server',
                          side_effect=partial_then_fail), \
             patch.object(chat_routes, 'USE_CLOUD', False):
            body = self.client.post('/api/chat/stream',
                                    json={'session_id': 'e3', 'message': 'hi'}).get_data(as_text=True)
        self.assertIn('tok1', body)
        self.assertEqual(len(self.sessions.get('e3', [])), 0)

    def test_23_empty_llm_response_cleans_up_user_message(self):
        """Bug 1 fix: LLM yields zero tokens → user message must NOT be orphaned."""
        import modules.chat.routes as chat_routes
        with patch.object(chat_routes, '_stream_llama_server', return_value=iter([])), \
             patch.object(chat_routes, 'USE_CLOUD', False):
            self.client.post('/api/chat/stream',
                             json={'session_id': 'empty-resp', 'message': 'hi'}).get_data()
        msgs = self.sessions.get('empty-resp', [])
        self.assertEqual(len(msgs), 0,
                         "Zero-token response must remove the orphaned user message")

    def test_24_empty_llm_response_still_sends_done(self):
        """Zero-token response must still emit [DONE] so client doesn't hang."""
        import modules.chat.routes as chat_routes
        with patch.object(chat_routes, '_stream_llama_server', return_value=iter([])), \
             patch.object(chat_routes, 'USE_CLOUD', False):
            body = self.client.post('/api/chat/stream',
                                    json={'session_id': 'e4', 'message': 'hi'}).get_data(as_text=True)
        self.assertIn('[DONE]', body)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Stop endpoint
# ─────────────────────────────────────────────────────────────────────────────

class ChatStopTests(unittest.TestCase):
    def setUp(self):
        import app
        from modules.chat.session import _CHAT_SESSIONS
        self.client  = app.app.test_client()
        self.sessions = _CHAT_SESSIONS
        _CHAT_SESSIONS.clear()

    def test_25_stop_removes_orphaned_user_message(self):
        """Last msg = user (in-progress) → stop removes it."""
        sid = 'stop1'
        self.sessions[sid] = [{'role': 'user', 'content': 'pending'}]
        resp = self.client.post('/api/chat/stop', json={'session_id': sid})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(self.sessions[sid]), 0)

    def test_26_stop_on_completed_session_is_noop(self):
        """Last msg = assistant (completed) → stop does nothing."""
        sid = 'stop2'
        self.sessions[sid] = [
            {'role': 'user',      'content': 'hi'},
            {'role': 'assistant', 'content': 'hello'},
        ]
        self.client.post('/api/chat/stop', json={'session_id': sid})
        self.assertEqual(len(self.sessions[sid]), 2)

    def test_27_stop_on_unknown_session_ok(self):
        """Stop on non-existent session → 200, no crash."""
        resp = self.client.post('/api/chat/stop', json={'session_id': 'ghost'})
        self.assertEqual(resp.status_code, 200)

    def test_28_stop_mid_multi_turn_preserves_completed_exchanges(self):
        """Stop after 2 completed turns + 1 pending → only pending user removed."""
        sid = 'stopmulti'
        self.sessions[sid] = [
            {'role': 'user',      'content': 'turn1'},
            {'role': 'assistant', 'content': 'reply1'},
            {'role': 'user',      'content': 'turn2'},
            {'role': 'assistant', 'content': 'reply2'},
            {'role': 'user',      'content': 'pending-turn3'},
        ]
        self.client.post('/api/chat/stop', json={'session_id': sid})
        msgs = self.sessions[sid]
        self.assertEqual(len(msgs), 4)
        self.assertEqual(msgs[-1]['content'], 'reply2')

    def test_29_stop_with_missing_session_id_ok(self):
        """Stop with no session_id → 200 (empty string looks up nothing)."""
        resp = self.client.post('/api/chat/stop', json={})
        self.assertEqual(resp.status_code, 200)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Truncate endpoint
# ─────────────────────────────────────────────────────────────────────────────

class ChatTruncateTests(unittest.TestCase):
    def setUp(self):
        import app
        from modules.chat.session import _CHAT_SESSIONS
        self.client  = app.app.test_client()
        self.sessions = _CHAT_SESSIONS
        _CHAT_SESSIONS.clear()

    def _session(self, sid, n_pairs):
        self.sessions[sid] = []
        for i in range(n_pairs):
            self.sessions[sid] += [
                {'role': 'user',      'content': f'u{i}'},
                {'role': 'assistant', 'content': f'a{i}'},
            ]

    def test_30_truncate_keeps_correct_slice(self):
        """keep_up_to=2 keeps first 2 messages."""
        self._session('t1', 3)
        self.client.post('/api/chat/truncate', json={'session_id': 't1', 'keep_up_to': 2})
        msgs = self.sessions['t1']
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[-1]['content'], 'a0')

    def test_31_truncate_to_zero_empties_session(self):
        """keep_up_to=0 removes all messages."""
        self._session('t2', 2)
        self.client.post('/api/chat/truncate', json={'session_id': 't2', 'keep_up_to': 0})
        self.assertEqual(len(self.sessions['t2']), 0)

    def test_32_truncate_negative_clamped_to_zero(self):
        """Bug 3 fix: negative keep_up_to must be clamped to 0, not [:−1]."""
        self._session('t3', 2)
        resp = self.client.post('/api/chat/truncate',
                                json={'session_id': 't3', 'keep_up_to': -5})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(self.sessions['t3']), 0,
                         "Negative keep_up_to must be treated as 0, not a Python negative index")

    def test_33_truncate_invalid_type_returns_400(self):
        """Bug 2 fix: non-integer keep_up_to must return 400, not crash with 500."""
        self._session('t4', 2)
        resp = self.client.post('/api/chat/truncate',
                                json={'session_id': 't4', 'keep_up_to': 'abc'})
        self.assertEqual(resp.status_code, 400,
                         "Non-integer keep_up_to must return 400, not 500")

    def test_34_truncate_float_string_returns_400(self):
        """Float string (e.g. '1.5') is not a valid integer → 400."""
        self._session('t5', 2)
        resp = self.client.post('/api/chat/truncate',
                                json={'session_id': 't5', 'keep_up_to': '1.5'})
        self.assertEqual(resp.status_code, 400)

    def test_35_truncate_unknown_session_ok(self):
        """Truncate on unknown session → 200, no crash."""
        resp = self.client.post('/api/chat/truncate',
                                json={'session_id': 'nobody', 'keep_up_to': 0})
        self.assertEqual(resp.status_code, 200)

    def test_36_truncate_larger_than_session_keeps_all(self):
        """keep_up_to > len(session) must keep all messages (no-op)."""
        self._session('t6', 2)
        self.client.post('/api/chat/truncate',
                         json={'session_id': 't6', 'keep_up_to': 9999})
        self.assertEqual(len(self.sessions['t6']), 4)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Session GET (page-reload restore)
# ─────────────────────────────────────────────────────────────────────────────

class ChatSessionGetTests(unittest.TestCase):
    def setUp(self):
        import app
        from modules.chat.session import _CHAT_SESSIONS
        self.client  = app.app.test_client()
        self.sessions = _CHAT_SESSIONS
        _CHAT_SESSIONS.clear()

    def test_37_session_get_returns_all_messages(self):
        """GET /api/chat/session returns full message list."""
        sid = 'g1'
        self.sessions[sid] = [
            {'role': 'user',      'content': 'req'},
            {'role': 'assistant', 'content': 'resp'},
        ]
        data = self.client.get(f'/api/chat/session?session_id={sid}').get_json()
        self.assertEqual(len(data['messages']), 2)
        self.assertEqual(data['messages'][0]['role'], 'user')
        self.assertEqual(data['messages'][1]['role'], 'assistant')

    def test_38_session_get_unknown_returns_empty_list(self):
        """GET for unknown session_id returns {"messages": []}."""
        data = self.client.get('/api/chat/session?session_id=nobody').get_json()
        self.assertEqual(data['messages'], [])

    def test_39_session_get_no_param_returns_empty_list(self):
        """GET with no session_id param → empty list (not 400 or crash)."""
        data = self.client.get('/api/chat/session').get_json()
        self.assertEqual(data['messages'], [])

    def test_40_session_get_returns_correct_content(self):
        """GET returns exact message content, not references or truncated data."""
        sid      = 'g2'
        long_msg = 'x' * 2000
        self.sessions[sid] = [{'role': 'user', 'content': long_msg}]
        data = self.client.get(f'/api/chat/session?session_id={sid}').get_json()
        self.assertEqual(data['messages'][0]['content'], long_msg)


# ─────────────────────────────────────────────────────────────────────────────
# 9. Edit workflow (truncate + re-stream)
# ─────────────────────────────────────────────────────────────────────────────

class ChatEditWorkflowTests(unittest.TestCase):
    """End-to-end: simulate what the frontend does for Edit (truncate + re-send)."""

    def setUp(self):
        import app
        from modules.chat.session import _CHAT_SESSIONS
        self.client  = app.app.test_client()
        self.sessions = _CHAT_SESSIONS
        _CHAT_SESSIONS.clear()

    def _stream(self, sid, msg, tokens):
        import modules.chat.routes as chat_routes
        with patch.object(chat_routes, '_stream_llama_server', return_value=iter(tokens)), \
             patch.object(chat_routes, 'USE_CLOUD', False):
            return self.client.post('/api/chat/stream',
                                    json={'session_id': sid, 'message': msg}).get_data()

    def test_41_edit_first_message_discards_subsequent_turns(self):
        """Edit U1: truncate to 0, re-send edited text → session rebuilt from scratch."""
        sid = 'edit1'
        self._stream(sid, 'original U1', ['A1'])
        self._stream(sid, 'U2', ['A2'])
        self.client.post('/api/chat/truncate', json={'session_id': sid, 'keep_up_to': 0})
        self._stream(sid, 'edited U1', ['new-A1'])
        msgs = self.sessions[sid]
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]['content'], 'edited U1')
        self.assertEqual(msgs[1]['content'], 'new-A1')

    def test_42_edit_second_message_keeps_first_exchange(self):
        """Edit U2: truncate to 2, re-send → U1/A1 preserved, U2/A2 replaced."""
        sid = 'edit2'
        self._stream(sid, 'U1', ['A1'])
        self._stream(sid, 'U2', ['A2'])
        self.client.post('/api/chat/truncate', json={'session_id': sid, 'keep_up_to': 2})
        self._stream(sid, 'edited U2', ['new-A2'])
        msgs = self.sessions[sid]
        self.assertEqual(len(msgs), 4)
        self.assertEqual(msgs[0]['content'], 'U1')
        self.assertEqual(msgs[1]['content'], 'A1')
        self.assertEqual(msgs[2]['content'], 'edited U2')
        self.assertEqual(msgs[3]['content'], 'new-A2')

    def test_43_re_send_after_edit_passes_correct_history_to_llm(self):
        """After edit, LLM receives correct history (no stale prior turns)."""
        import modules.chat.routes as chat_routes
        sid = 'edit3'
        self._stream(sid, 'U1', ['A1'])
        self._stream(sid, 'U2', ['A2'])
        self.client.post('/api/chat/truncate', json={'session_id': sid, 'keep_up_to': 0})

        with patch.object(chat_routes, '_stream_llama_server',
                          return_value=iter(['new'])) as mock, \
             patch.object(chat_routes, 'USE_CLOUD', False):
            self.client.post('/api/chat/stream',
                             json={'session_id': sid, 'message': 'fresh start'}).get_data()
            msgs_to_llm = mock.call_args[0][0]

        # Only system + new user — no stale U1/A1/U2/A2
        self.assertEqual(len(msgs_to_llm), 2)
        self.assertEqual(msgs_to_llm[0]['role'], 'system')
        self.assertEqual(msgs_to_llm[1]['role'], 'user')
        self.assertEqual(msgs_to_llm[1]['content'], 'fresh start')


# ─────────────────────────────────────────────────────────────────────────────
# 10. Download DOCX
# ─────────────────────────────────────────────────────────────────────────────

class ChatDownloadTests(unittest.TestCase):
    def setUp(self):
        import app
        from modules.chat.session import _CHAT_SESSIONS
        self.client  = app.app.test_client()
        self.sessions = _CHAT_SESSIONS
        _CHAT_SESSIONS.clear()

    def test_44_download_empty_session_rejected(self):
        """No conversation → 400."""
        resp = self.client.post('/api/chat/download-docx', json={'session_id': 'empty'})
        self.assertEqual(resp.status_code, 400)

    def test_45_download_unknown_session_rejected(self):
        """Unknown session_id (no entry) → 400."""
        resp = self.client.post('/api/chat/download-docx', json={'session_id': 'ghost'})
        self.assertEqual(resp.status_code, 400)

    def test_46_download_returns_docx_content_type(self):
        """Valid session → 200 with DOCX MIME type."""
        sid = 'dl1'
        self.sessions[sid] = [
            {'role': 'user',      'content': 'Estimate 3 sprints'},
            {'role': 'assistant', 'content': '42 points'},
        ]
        resp = self.client.post('/api/chat/download-docx', json={'session_id': sid})
        self.assertEqual(resp.status_code, 200)
        self.assertIn('openxmlformats', resp.content_type)

    def test_47_download_docx_is_valid_zip(self):
        """DOCX file (zip-based) must start with PK magic bytes."""
        import io, zipfile
        sid = 'dl2'
        self.sessions[sid] = [{'role': 'user', 'content': 'hello'}]
        resp = self.client.post('/api/chat/download-docx', json={'session_id': sid})
        self.assertTrue(zipfile.is_zipfile(io.BytesIO(resp.data)),
                        "Downloaded file is not a valid ZIP/DOCX")

    def test_48_download_no_session_id_rejected(self):
        """Missing session_id field → 400."""
        resp = self.client.post('/api/chat/download-docx', json={})
        self.assertEqual(resp.status_code, 400)


# ─────────────────────────────────────────────────────────────────────────────
# 11. Cloud path, rate limit, static asset, route
# ─────────────────────────────────────────────────────────────────────────────

class ChatRouteAndConfigTests(unittest.TestCase):
    def setUp(self):
        import app
        from modules.chat.session import _CHAT_SESSIONS
        self.client  = app.app.test_client()
        self.sessions = _CHAT_SESSIONS
        _CHAT_SESSIONS.clear()

    def test_49_cloud_path_uses_stream_openrouter(self):
        """USE_CLOUD=True → _stream_openrouter called, _stream_llama_server not."""
        import modules.chat.routes as chat_routes
        with patch.object(chat_routes, '_stream_openrouter',
                          return_value=iter(['cloud'])) as mock_or, \
             patch.object(chat_routes, '_stream_llama_server') as mock_local, \
             patch.object(chat_routes, 'USE_CLOUD', True):
            self.client.post('/api/chat/stream',
                             json={'session_id': 'c', 'message': 'hi'}).get_data()
            mock_or.assert_called_once()
            mock_local.assert_not_called()

    def test_50_local_path_uses_stream_llama_server(self):
        """USE_CLOUD=False → _stream_llama_server called, _stream_openrouter not."""
        import modules.chat.routes as chat_routes
        with patch.object(chat_routes, '_stream_llama_server',
                          return_value=iter(['local'])) as mock_local, \
             patch.object(chat_routes, '_stream_openrouter') as mock_or, \
             patch.object(chat_routes, 'USE_CLOUD', False):
            self.client.post('/api/chat/stream',
                             json={'session_id': 'l', 'message': 'hi'}).get_data()
            mock_local.assert_called_once()
            mock_or.assert_not_called()

    def test_51_chat_stream_has_rate_limit_decorator(self):
        """chat_stream must carry @rate_limit decorator for multi-user safety."""
        from modules.chat import routes as chat_routes_module
        import inspect
        src = inspect.getsource(chat_routes_module)
        idx = src.find('def chat_stream(')
        self.assertGreater(idx, 0)
        window = src[max(0, idx - 400):idx]
        self.assertIn('rate_limit', window)

    def test_52_get_chat_route_serves_html(self):
        """GET /chat → 200 with HTML body."""
        resp = self.client.get('/chat')
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b'<!DOCTYPE html>', resp.data)

    def test_53_chat_html_no_stale_ollama_reference(self):
        """chat.html must not reference d.ollama — would break status check."""
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(base, 'static', 'chat.html')) as f:
            src = f.read()
        self.assertNotIn('d.ollama', src)
        self.assertIn('llama_server', src)

    def test_54_chat_html_has_stop_button(self):
        """chat.html must contain stop button element for Stop Generation feature."""
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(base, 'static', 'chat.html')) as f:
            src = f.read()
        self.assertIn('stopBtn', src)
        self.assertIn('stopGeneration', src)

    def test_55_chat_html_has_edit_functionality(self):
        """chat.html must contain edit-related functions for Edit Prompt feature."""
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(base, 'static', 'chat.html')) as f:
            src = f.read()
        self.assertIn('startEdit', src)
        self.assertIn('saveEdit', src)
        self.assertIn('cancelEdit', src)

    def test_56_chat_html_has_session_restore(self):
        """chat.html must call GET /api/chat/session for page-reload restore."""
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(base, 'static', 'chat.html')) as f:
            src = f.read()
        self.assertIn('/api/chat/session', src)


if __name__ == '__main__':
    unittest.main(verbosity=2)
