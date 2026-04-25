"""
Concurrency guardrail tests — no LLM calls, no network.
Verifies:
  1. Semaphore caps concurrent _call_llm invocations to LLM_MAX_CONCURRENT
  2. Rate limit decorator helper is no-op when flask-limiter absent,
     real when present
  3. Flask app is configured threaded for dev server
  4. Semaphore releases on exception (no deadlock)
"""
import os
import sys
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SemaphoreTests(unittest.TestCase):
    def test_01_semaphore_caps_concurrent_calls(self):
        """_LLM_SEM limits simultaneous _call_llm to configured bound."""
        import llm.client as llm_client

        in_flight = []
        max_seen  = [0]
        lock      = threading.Lock()

        def slow_llama(*a, **kw):
            with lock:
                in_flight.append(1)
                max_seen[0] = max(max_seen[0], len(in_flight))
            time.sleep(0.15)
            with lock:
                in_flight.pop()
            return '{"ok":1}'

        # Shrink semaphore to 3 for deterministic test
        with patch.object(llm_client, '_LLM_SEM', threading.BoundedSemaphore(3)), \
             patch.object(llm_client, 'USE_CLOUD', False), \
             patch.object(llm_client, '_call_llama_server', side_effect=slow_llama):
            threads = [threading.Thread(target=llm_client._call_llm,
                                         args=("m", "p"),
                                         kwargs={"options_override": {"max_tokens": 100}})
                       for _ in range(10)]
            for t in threads: t.start()
            for t in threads: t.join()

        self.assertLessEqual(max_seen[0], 3, f"Semaphore breach: saw {max_seen[0]} concurrent")
        self.assertGreaterEqual(max_seen[0], 2, "Suspicious — no parallelism observed")

    def test_02_semaphore_releases_on_exception(self):
        """Exception in _call_llama_server must not leak semaphore permits."""
        import llm.client as llm_client

        sem = threading.BoundedSemaphore(2)
        with patch.object(llm_client, '_LLM_SEM', sem), \
             patch.object(llm_client, 'USE_CLOUD', False), \
             patch.object(llm_client, '_call_llama_server',
                          side_effect=RuntimeError("boom")):
            for _ in range(5):
                with self.assertRaises(RuntimeError):
                    llm_client._call_llm("m", "p")

        # All 2 permits must be free
        for _ in range(2):
            self.assertTrue(sem.acquire(blocking=False), "Permit leaked")

    def test_03_semaphore_bound_from_env(self):
        """LLM_MAX_CONCURRENT env var sized from import-time os.environ."""
        import config
        self.assertTrue(hasattr(config, '_LLM_SEM'))
        self.assertGreaterEqual(config._LLM_MAX_CONCURRENT, 1)


class RateLimitHelperTests(unittest.TestCase):
    def test_04_rate_limit_noop_when_absent(self):
        """rate_limit returns identity decorator when _HAS_LIMITER=False."""
        import config
        with patch.object(config, '_HAS_LIMITER', False):
            deco = config.rate_limit("5 per minute")
            def sample(): return "hi"
            wrapped = deco(sample)
            self.assertIs(wrapped, sample, "Should pass through when limiter absent")

    def test_05_rate_limit_calls_limiter_when_present(self):
        """rate_limit proxies to limiter.limit when _HAS_LIMITER=True."""
        import config
        called_with = []
        class FakeLimiter:
            def limit(self, spec):
                called_with.append(spec)
                return lambda f: f
        with patch.object(config, '_HAS_LIMITER', True), \
             patch.object(config, 'limiter', FakeLimiter()):
            config.rate_limit("10 per minute")
            self.assertEqual(called_with, ["10 per minute"])


class FlaskConfigTests(unittest.TestCase):
    def test_06_app_run_is_threaded(self):
        """Dev server must be threaded=True for multi-user tolerance."""
        import app
        import inspect
        src = inspect.getsource(app)
        self.assertIn("threaded=True", src,
                      "app.run must have threaded=True for concurrent dev use")

    def test_07_gunicorn_command_documented(self):
        """Gunicorn launch command present in source as a comment."""
        import app
        import inspect
        src = inspect.getsource(app)
        self.assertIn("gunicorn", src.lower())

    def test_08_heavy_endpoints_rate_limited(self):
        """Heavy endpoints carry @rate_limit decorator."""
        from modules.project_docs import routes as proj_routes
        from modules.pii import routes as pii_routes
        import inspect

        src_proj = inspect.getsource(proj_routes)
        src_pii  = inspect.getsource(pii_routes)

        for endpoint in ["generate_artifact", "generate_docs"]:
            idx = src_proj.find(f"def {endpoint}(")
            self.assertGreater(idx, 0, f"{endpoint} not found in project_docs routes")
            window = src_proj[max(0, idx - 400):idx]
            self.assertIn("rate_limit", window,
                          f"{endpoint} missing @rate_limit decorator")

        idx = src_pii.find("def extract_text_api(")
        self.assertGreater(idx, 0, "extract_text_api not found in pii routes")
        window = src_pii[max(0, idx - 400):idx]
        self.assertIn("rate_limit", window,
                      "extract_text_api missing @rate_limit decorator")


if __name__ == "__main__":
    unittest.main(verbosity=2)
