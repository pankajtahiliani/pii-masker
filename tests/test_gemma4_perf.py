"""
Gemma 4 performance + correctness tests.

Run: python -m tests.test_gemma4_perf
or:  pytest tests/test_gemma4_perf.py -v -s

Validates:
  1. Raw Ollama tok/s benchmark for gemma4:e4b
  2. Each artifact completes in target time
  3. Output is valid JSON (not _raw fallback)
  4. Output meets minimum content thresholds
"""
import json
import time
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import requests

OLLAMA_CHAT = "http://localhost:11434/api/chat"
FLASK_URL   = "http://localhost:5000"
MODEL       = "gemma4:e4b"

# Hard time budgets per artifact — cloud (OpenRouter) should hit these easily
# If any exceeds these, test fails so user knows to switch model or check network
USE_CLOUD = bool(os.environ.get("OPENROUTER_API_KEY", "").strip())
TIME_BUDGET_SECS = ({
    'backlog':       30, 'sprint_plan':   20, 'sprint_review': 25,
    'retrospective': 20, 'risk_register': 25, 'test_cases':    30,
} if USE_CLOUD else {
    'backlog':      180, 'sprint_plan':  120, 'sprint_review': 150,
    'retrospective':120, 'risk_register':150, 'test_cases':   180,
})

# Minimum content thresholds to ensure output is not stubbed/empty
MIN_ITEMS = {
    'backlog':       {'path': 'epics',       'min': 3},
    'sprint_plan':   {'path': 'sprints',     'min': 3},
    'sprint_review': {'path': 'reviews',     'min': 2},
    'risk_register': {'path': 'risks',       'min': 5},
    'test_cases':    {'path': 'test_cases',  'min': 8},
}

SAMPLE_SOURCE = """
Project: Customer Loyalty Platform
Goal: Build web + mobile app for retail chain (500 stores, 2M users).
Features: tiered rewards, point redemption, personalised offers, receipt scanning,
          referral program, stripe integration, push notifications.
Team: 1 PO, 1 Scrum Master, 3 backend devs, 2 mobile devs, 1 QA.
Timeline: 6 months. Deadline: Q3 launch.
Integrations: Stripe, Twilio, SendGrid, POS system (legacy SOAP API).
Compliance: PCI-DSS, GDPR, CCPA.
Risks: POS vendor delays, mobile app store review, stakeholder availability.
"""


def bench_raw_tokps():
    """Measure raw tok/s for gemma4:e4b via /api/chat."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Count from 1 to 50. Output only numbers comma-separated."}],
        "stream": False,
        "options": {"num_ctx": 2048, "num_predict": 300, "temperature": 0.1, "top_k": 40, "num_gpu": 99, "num_thread": 8},
        "keep_alive": "10m",
    }
    t0 = time.time()
    r = requests.post(OLLAMA_CHAT, json=payload, timeout=120)
    elapsed = time.time() - t0
    r.raise_for_status()
    j = r.json()
    eval_count = j.get("eval_count", 0)
    eval_dur_ns = j.get("eval_duration", 1)
    tokps = eval_count / (eval_dur_ns / 1e9) if eval_dur_ns else 0
    return tokps, elapsed, eval_count


class GemmaPerfTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Flask must be running
        try:
            r = requests.get(f"{FLASK_URL}/api/status", timeout=3)
            r.raise_for_status()
        except Exception as e:
            raise unittest.SkipTest(f"Flask server not running on {FLASK_URL}: {e}")

    def test_01_raw_tokps(self):
        """Warm model + measure baseline tok/s."""
        print("\n[bench] warming model + measuring tok/s...")
        tokps, wall, n = bench_raw_tokps()
        print(f"[bench] gemma4:e4b: {tokps:.1f} tok/s ({n} tokens in {wall:.1f}s wall)")
        self.assertGreater(tokps, 3, f"Too slow: {tokps:.1f} tok/s. Expect >3 on M-series.")
        self._tokps = tokps

    def _run_artifact(self, key):
        t0 = time.time()
        r = requests.post(
            f"{FLASK_URL}/api/generate-artifact",
            json={"source": SAMPLE_SOURCE, "artifact": key},
            timeout=TIME_BUDGET_SECS[key] + 30,
        )
        elapsed = time.time() - t0
        self.assertEqual(r.status_code, 200, f"{key}: HTTP {r.status_code} — {r.text[:200]}")
        body = r.json()
        data = body.get("data")
        self.assertIsNotNone(data, f"{key}: no data field")
        self.assertNotIn("_raw", data if isinstance(data, dict) else {},
                         f"{key}: JSON parse failed, got _raw fallback")

        # Content threshold check
        if key in MIN_ITEMS:
            cfg = MIN_ITEMS[key]
            items = data.get(cfg['path'], []) if isinstance(data, dict) else data
            if not isinstance(items, list):
                items = []
            self.assertGreaterEqual(
                len(items), cfg['min'],
                f"{key}: only {len(items)} items, expected ≥{cfg['min']}"
            )
        print(f"[{key}] OK in {elapsed:.1f}s (budget {TIME_BUDGET_SECS[key]}s)")
        self.assertLess(elapsed, TIME_BUDGET_SECS[key],
                        f"{key}: took {elapsed:.1f}s, budget {TIME_BUDGET_SECS[key]}s")
        return elapsed

    def test_02_backlog(self):      self._run_artifact('backlog')
    def test_03_sprint_plan(self):  self._run_artifact('sprint_plan')
    def test_04_retrospective(self): self._run_artifact('retrospective')
    def test_05_risk_register(self): self._run_artifact('risk_register')
    def test_06_test_cases(self):   self._run_artifact('test_cases')
    def test_07_sprint_review(self): self._run_artifact('sprint_review')


if __name__ == "__main__":
    unittest.main(verbosity=2)
