"""
Fast unit tests — no LLM calls, no network.
Verifies:
  1. _call_llm dispatches to openrouter when USE_CLOUD, llama-server otherwise
  2. max_tokens mapping works
  3. format_json → response_format mapping works
  4. JSON repair pipeline handles truncated output
"""
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DispatchTests(unittest.TestCase):
    def test_01_cloud_dispatch(self):
        """USE_CLOUD=True → _call_llm routes to openrouter."""
        import llm.client as llm_client
        with patch.object(llm_client, 'USE_CLOUD', True), \
             patch.object(llm_client, '_call_openrouter', return_value='{"ok":1}') as mock_or, \
             patch.object(llm_client, '_call_llama_server', return_value='should-not-fire') as mock_ls:
            out = llm_client._call_llm("any-model", "test prompt",
                                       timeout=30,
                                       options_override={"max_tokens": 800},
                                       format_json=True)
            mock_or.assert_called_once()
            mock_ls.assert_not_called()
            kwargs = mock_or.call_args.kwargs
            self.assertEqual(kwargs["max_tokens"], 800)
            self.assertTrue(kwargs["format_json"])
            self.assertEqual(out, '{"ok":1}')

    def test_02_local_dispatch(self):
        """USE_CLOUD=False → _call_llm routes to llama-server."""
        import llm.client as llm_client
        with patch.object(llm_client, 'USE_CLOUD', False), \
             patch.object(llm_client, '_call_llama_server', return_value='{"ok":2}') as mock_ls, \
             patch.object(llm_client, '_call_openrouter') as mock_or:
            out = llm_client._call_llm("qwen2.5", "prompt",
                                       options_override={"max_tokens": 1200},
                                       format_json=True)
            mock_ls.assert_called_once()
            mock_or.assert_not_called()
            self.assertEqual(out, '{"ok":2}')

    def test_03_cloud_fallback_on_error(self):
        """OpenRouter exception → graceful fallback to llama-server."""
        import llm.client as llm_client
        with patch.object(llm_client, 'USE_CLOUD', True), \
             patch.object(llm_client, '_call_openrouter',
                          side_effect=RuntimeError("api down")) as mock_or, \
             patch.object(llm_client, '_call_llama_server',
                          return_value='{"ok":3}') as mock_ls:
            out = llm_client._call_llm("qwen2.5", "p",
                                       options_override={"max_tokens": 500})
            mock_or.assert_called_once()
            mock_ls.assert_called_once()
            self.assertEqual(out, '{"ok":3}')


class JsonRepairTests(unittest.TestCase):
    def test_04_mid_string_truncation(self):
        """_close_truncated_json repairs JSON cut mid-string."""
        from modules.project_docs.parser import _close_truncated_json
        truncated = '{"epics":[{"id":"EP-01","title":"Core Website Functionality & Desi'
        repaired  = _close_truncated_json(truncated)
        import json
        parsed = json.loads(repaired)
        self.assertIn("epics", parsed)
        self.assertEqual(len(parsed["epics"]), 1)
        self.assertEqual(parsed["epics"][0]["id"], "EP-01")
        self.assertNotIn("title", parsed["epics"][0])

    def test_05_complete_objects_extraction(self):
        """_extract_complete_objects salvages good objects from truncated array."""
        from modules.project_docs.parser import _extract_complete_objects
        text = '{"risks":[{"id":"R-01","title":"A"},{"id":"R-02","title":"B"},{"id":"R-03","title":"inc'
        out  = _extract_complete_objects(text, "risks")
        self.assertIsNotNone(out)
        self.assertEqual(len(out["risks"]), 2)

    def test_06_parse_json_response_fallback(self):
        """_parse_json_response walks all 3 passes → never returns None."""
        from modules.project_docs.parser import _parse_json_response
        raw = '```json\n{"epics":[{"id":"EP-01"}]}\n```'
        out = _parse_json_response(raw, "backlog")
        self.assertIsInstance(out, list)
        self.assertEqual(out[0]["id"], "EP-01")


class ConfigTests(unittest.TestCase):
    def test_07_max_tokens_budgets_sane(self):
        """Artifact max_tokens budgets sized for qwen2.5:7b at ~7 tok/s on M4."""
        from modules.project_docs.prompts import _ARTIFACT_PREDICT
        self.assertEqual(_ARTIFACT_PREDICT['backlog'], 1500)
        self.assertEqual(_ARTIFACT_PREDICT['test_cases'], 1300)
        for key in ['backlog', 'sprint_plan', 'sprint_review',
                    'retrospective', 'risk_register', 'test_cases']:
            self.assertIn(key, _ARTIFACT_PREDICT)

    def test_08_llama_server_json_mode(self):
        """_call_llama_server sends response_format when format_json=True."""
        import llm.client as llm_client
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "{}"}}]}
        mock_resp.raise_for_status.return_value = None
        with patch.object(llm_client, '_LLAMA_SESSION') as mock_session:
            mock_session.post.return_value = mock_resp
            llm_client._call_llama_server("qwen2.5", "p", timeout=10, format_json=True)
            sent = mock_session.post.call_args.kwargs["json"]
            self.assertEqual(sent["response_format"], {"type": "json_object"})
            self.assertEqual(sent["temperature"], 0)
            self.assertEqual(sent["top_k"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
