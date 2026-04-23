"""
Call Summarizer migration tests — no LLM calls, no network.
Verifies:
  1. /api/summarize-call uses _call_llm dispatcher (not _call_llama_server direct)
     → ensures semaphore, cloud fallback, rate limit all apply
  2. /api/status returns 'llama_server' key (not 'ollama') for frontend compat
  3. @rate_limit decorator applied to summarize_call
  4. Cloud path + local path both reachable via summarizer
"""
import json
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SummarizerDispatchTests(unittest.TestCase):
    def setUp(self):
        import app
        self.app = app
        self.client = app.app.test_client()

    def test_01_summarizer_routes_through_call_llm(self):
        """/api/summarize-call must go through _call_llm (not _call_llama_server direct)."""
        fake_json = json.dumps({
            "key_takeaways": ["a"], "action_items": [],
            "decisions": [], "risks": [], "next_steps": []
        })
        with patch.object(self.app, '_call_llm', return_value=fake_json) as mock_llm, \
             patch.object(self.app, '_call_llama_server') as mock_direct:
            resp = self.client.post('/api/summarize-call',
                                    json={"source": "hello world transcript"})
            self.assertEqual(resp.status_code, 200)
            mock_llm.assert_called_once()
            mock_direct.assert_not_called()
            # format_json=True + max_tokens override present
            kwargs = mock_llm.call_args.kwargs
            self.assertTrue(kwargs.get("format_json"))
            self.assertEqual(kwargs.get("options_override", {}).get("max_tokens"), 1200)

    def test_02_summarizer_empty_transcript_rejected(self):
        """Empty source returns 400."""
        resp = self.client.post('/api/summarize-call', json={"source": ""})
        self.assertEqual(resp.status_code, 400)

    def test_03_summarizer_cloud_path_works(self):
        """USE_CLOUD=True → _call_llm routes to openrouter under the hood."""
        fake_json = json.dumps({
            "key_takeaways": ["cloud"], "action_items": [],
            "decisions": [], "risks": [], "next_steps": []
        })
        with patch.object(self.app, 'USE_CLOUD', True), \
             patch.object(self.app, 'OPENROUTER_MODEL', 'google/gemma-3-27b-it:free'), \
             patch.object(self.app, '_call_openrouter', return_value=fake_json) as mock_or, \
             patch.object(self.app, '_call_llama_server') as mock_local:
            resp = self.client.post('/api/summarize-call',
                                    json={"source": "meeting transcript"})
            self.assertEqual(resp.status_code, 200)
            mock_or.assert_called_once()
            mock_local.assert_not_called()


class StatusEndpointTests(unittest.TestCase):
    def setUp(self):
        import app
        self.app = app
        self.client = app.app.test_client()

    def test_04_status_returns_llama_server_key(self):
        """/api/status must expose 'llama_server' key for frontend compat."""
        with patch.object(self.app, 'check_backend_available',
                          return_value={"available": True, "models": ["qwen2.5"]}):
            resp = self.client.get('/api/status')
            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()
            self.assertIn("llama_server", data)
            self.assertNotIn("ollama", data)
            self.assertEqual(data["llama_server"]["available"], True)
            self.assertEqual(data["backend"], "llamacpp")

    def test_05_status_cloud_backend_label(self):
        """backend='openrouter' when USE_CLOUD=True."""
        with patch.object(self.app, 'USE_CLOUD', True), \
             patch.object(self.app, 'check_backend_available',
                          return_value={"available": False, "models": []}):
            resp = self.client.get('/api/status')
            data = resp.get_json()
            self.assertEqual(data["backend"], "openrouter")
            self.assertTrue(data["cloud_enabled"])


class RateLimitTests(unittest.TestCase):
    def test_06_summarize_call_has_rate_limit(self):
        """summarize_call must carry @rate_limit for multi-user safety."""
        import app
        import inspect
        src = inspect.getsource(app)
        idx = src.find("def summarize_call(")
        self.assertGreater(idx, 0, "summarize_call not found")
        window = src[max(0, idx - 200):idx]
        self.assertIn("rate_limit", window,
                      "summarize_call missing @rate_limit decorator")


class FrontendAssetsTests(unittest.TestCase):
    """Verify no stale 'ollama' JS key that would break status check."""
    def test_07_frontend_uses_llama_server_key(self):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for f in ["call_summarizer.html", "sanitize.html", "project_docs.html"]:
            path = os.path.join(base, "static", f)
            with open(path) as fh:
                src = fh.read()
            # The broken check pattern "d.ollama" and "d.ollama.available" must be gone
            self.assertNotIn("d.ollama.available", src,
                             f"{f} still reads d.ollama.available — frontend broken")
            self.assertNotIn("d.ollama &&", src,
                             f"{f} still checks d.ollama — use d.llama_server")
            self.assertIn("llama_server", src,
                          f"{f} must check d.llama_server")


if __name__ == "__main__":
    unittest.main(verbosity=2)
