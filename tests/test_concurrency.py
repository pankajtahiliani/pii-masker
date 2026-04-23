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
        import app

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
        with patch.object(app, '_LLM_SEM', threading.BoundedSemaphore(3)), \
             patch.object(app, 'USE_CLOUD', False), \
             patch.object(app, '_call_llama_server', side_effect=slow_llama):
            threads = [threading.Thread(target=app._call_llm,
                                         args=("m", "p"),
                                         kwargs={"options_override": {"max_tokens": 100}})
                       for _ in range(10)]
            for t in threads: t.start()
            for t in threads: t.join()

        self.assertLessEqual(max_seen[0], 3, f"Semaphore breach: saw {max_seen[0]} concurrent")
        self.assertGreaterEqual(max_seen[0], 2, "Suspicious — no parallelism observed")

    def test_02_semaphore_releases_on_exception(self):
        """Exception in _call_llama_server must not leak semaphore permits."""
        import app

        sem = threading.BoundedSemaphore(2)
        with patch.object(app, '_LLM_SEM', sem), \
             patch.object(app, 'USE_CLOUD', False), \
             patch.object(app, '_call_llama_server', side_effect=RuntimeError("boom")):
            for _ in range(5):
                with self.assertRaises(RuntimeError):
                    app._call_llm("m", "p")

        # All 2 permits must be free
        for _ in range(2):
            self.assertTrue(sem.acquire(blocking=False), "Permit leaked")

    def test_03_semaphore_bound_from_env(self):
        """LLM_MAX_CONCURRENT env var sized from import-time os.environ."""
        import app
        # Default or env-set; must be ≥1 BoundedSemaphore-ish
        self.assertTrue(hasattr(app, '_LLM_SEM'))
        self.assertGreaterEqual(app._LLM_MAX_CONCURRENT, 1)


class RateLimitHelperTests(unittest.TestCase):
    def test_04_rate_limit_noop_when_absent(self):
        """rate_limit returns identity decorator when _HAS_LIMITER=False."""
        import app
        with patch.object(app, '_HAS_LIMITER', False):
            deco = app.rate_limit("5 per minute")
            def sample(): return "hi"
            wrapped = deco(sample)
            self.assertIs(wrapped, sample, "Should pass through when limiter absent")

    def test_05_rate_limit_calls_limiter_when_present(self):
        """rate_limit proxies to limiter.limit when _HAS_LIMITER=True."""
        import app
        called_with = []
        class FakeLimiter:
            def limit(self, spec):
                called_with.append(spec)
                return lambda f: f
        with patch.object(app, '_HAS_LIMITER', True), \
             patch.object(app, 'limiter', FakeLimiter()):
            app.rate_limit("10 per minute")
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
        import app
        import inspect
        src = inspect.getsource(app)
        for endpoint in ["generate_artifact", "generate_docs", "extract_text_api"]:
            # rate_limit should appear near endpoint def
            idx = src.find(f"def {endpoint}(")
            self.assertGreater(idx, 0, f"{endpoint} not found")
            window = src[max(0, idx-400):idx]
            self.assertIn("rate_limit", window,
                          f"{endpoint} missing @rate_limit decorator")


if __name__ == "__main__":
    unittest.main(verbosity=2)
