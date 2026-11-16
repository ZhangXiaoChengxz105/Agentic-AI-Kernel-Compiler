"""Tests for agent/lumen.py against a LOCAL fake Lumen server (no internet, no coins spent).

Run: envs/tirx/bin/python -m unittest tests.test_lumen -v
"""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.lumen import BudgetError, ConsentRequired, LumenClient, LumenError  # noqa: E402

MODELS = {"object": "list", "data": [
    {"id": "glm-5.3", "parent": None, "input_cost_per_million": 1.321413, "output_cost_per_million": 4.248318,
     "max_model_len": 524282, "tags": []},
    {"id": "needs-ack", "parent": None, "input_cost_per_million": 0.1, "output_cost_per_million": 0.1, "tags": ["needs_ack"]},
]}


class FakeLumen(BaseHTTPRequestHandler):
    state = {"coins": 10.0, "fail_next": 0, "calls": []}

    def log_message(self, *a):  # silence
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.headers.get("Authorization") != "Bearer sk_test":
            return self._send(401, {"error": "Invalid or revoked API key"})
        if self.path.startswith("/v1/models"):
            return self._send(200, MODELS)
        if self.path == "/v1/usage":
            return self._send(200, {"requests": len(self.state["calls"]), "coins_available": self.state["coins"]})
        self._send(404, {})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.state["calls"].append(body)
        if self.state["fail_next"] > 0:
            self.state["fail_next"] -= 1
            return self._send(503, {"error": "busy"})
        if body["model"] == "needs-ack":
            return self._send(403, {"error": {"code": "consent_required", "message": "acknowledge first"}})
        self._send(200, {"model": body["model"], "choices": [{"finish_reason": "stop", "message": {
            "role": "assistant", "content": "hello from fake lumen", "reasoning_content": "thinking..."}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 500}})


class LumenClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeLumen)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.srv.server_address[1]}/v1"
        cls.tmp = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.tmp.cleanup()

    def setUp(self):
        FakeLumen.state.update(coins=10.0, fail_next=0, calls=[])
        self.log = Path(self.tmp.name) / "calls.jsonl"
        self.c = LumenClient(base_url=self.base, api_key="sk_test", log_path=self.log)

    def test_models_and_price(self):
        self.assertEqual(self.c.price("glm-5.3"), (1.321413, 4.248318))

    def test_chat_parses_reply_usage_and_cost(self):
        r = self.c.chat([{"role": "user", "content": "hi"}], max_tokens=1000, tag="t")
        self.assertEqual((r.content, r.reasoning), ("hello from fake lumen", "thinking..."))
        self.assertEqual((r.input_tokens, r.output_tokens), (1000, 500))
        self.assertAlmostEqual(r.cost, 1000 / 1e6 * 1.321413 + 500 / 1e6 * 4.248318)
        rec = json.loads(self.log.read_text().splitlines()[-1])
        self.assertEqual((rec["tag"], rec["content"]), ("t", "hello from fake lumen"))

    def test_dry_run_sends_nothing(self):
        r = self.c.chat([{"role": "user", "content": "hi"}], max_tokens=100, dry_run=True)
        self.assertTrue(r.dry_run)
        self.assertEqual(FakeLumen.state["calls"], [])

    def test_retries_on_503(self):
        FakeLumen.state["fail_next"] = 2
        self.c.chat([{"role": "user", "content": "hi"}], max_tokens=100)
        self.assertEqual(len(FakeLumen.state["calls"]), 3)

    def test_per_call_limit(self):
        with self.assertRaises(BudgetError):
            self.c.chat([{"role": "user", "content": "x" * 400_000}], max_tokens=200_000)   # ~0.98 coins worst case
        self.assertEqual(FakeLumen.state["calls"], [])

    def test_reserve_protects_last_coins(self):
        FakeLumen.state["coins"] = 1.01
        with self.assertRaises(BudgetError):
            self.c.chat([{"role": "user", "content": "hi"}], max_tokens=8000)               # 0.034 worst case
        self.assertEqual(FakeLumen.state["calls"], [])

    def test_consent_required(self):
        with self.assertRaises(ConsentRequired):
            self.c.chat([{"role": "user", "content": "hi"}], model="needs-ack", max_tokens=10)

    def test_bad_key(self):
        bad = LumenClient(base_url=self.base, api_key="sk_wrong", log_path=None)
        with self.assertRaises(LumenError) as ctx:
            bad.usage()
        self.assertEqual(ctx.exception.status, 401)


if __name__ == "__main__":
    unittest.main()
