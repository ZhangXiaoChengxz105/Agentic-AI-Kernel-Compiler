"""Task 1.8: minimal Lumen (OpenAI-compatible) client with a budget guard and full call logging.

Lumen facts (verified 2026-11-15 from github.com/ncsa/lumen source + docs):
  base URL  https://lumen.ncsa.illinois.edu/v1, header "Authorization: Bearer $LUMEN_API_KEY"
  GET  /v1/models  -> ids, upstream "root", context, input/output_cost_per_million   (FREE)
  GET  /v1/usage   -> per-key tokens, cost, coins_available                          (FREE)
  POST /v1/chat/completions                                                         (COSTS COINS)
  403 code "consent_required" -> model needs a one-time acknowledgment (web UI)

LumenClient(model="glm-5.3")
  .models() / .usage() / .price(model)                    free
  .chat(messages, max_tokens=..., dry_run=...) -> ChatResult
      1. estimate input tokens (chars/4) and worst-case cost = in*price_in + max_tokens*price_out
      2. refuse (BudgetError) if worst case > per_call_limit, or coins_available - worst case < reserve
      3. dry_run: log and return without sending;  otherwise POST, retry 429/5xx (3x, backoff)
      4. append everything to logs/lumen/calls.jsonl (request, reply, reasoning, usage, cost, latency)
CLI:  python -m agent.lumen models | usage | chat [--dry-run] [--model M] [--max-tokens N] "prompt"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_BASE = "https://lumen.ncsa.illinois.edu/v1"
DEFAULT_MODEL = "glm-5.3"
LOG_PATH = WORKSPACE / "logs" / "lumen" / "calls.jsonl"


class LumenError(RuntimeError):
    def __init__(self, msg: str, status: int | None = None, body: str = ""):
        super().__init__(msg)
        self.status, self.body = status, body


class BudgetError(LumenError):
    pass


class ConsentRequired(LumenError):
    pass


@dataclass
class ChatResult:
    model: str
    content: str
    reasoning: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0                    # coins, computed from Lumen's published prices
    latency_s: float = 0.0
    finish_reason: str = ""
    dry_run: bool = False
    raw: dict = field(default_factory=dict)


class LumenClient:
    def __init__(self, model: str = DEFAULT_MODEL, base_url: str | None = None, api_key: str | None = None,
                 per_call_limit: float = 0.5, reserve: float = 1.0, timeout_s: float = 600.0,
                 log_path: Path | None = LOG_PATH, dry_run: bool = False):
        self.model = model
        self.base = (base_url or os.environ.get("LUMEN_BASE_URL") or DEFAULT_BASE).rstrip("/")
        self.key = api_key if api_key is not None else os.environ.get("LUMEN_API_KEY", "")
        self.per_call_limit, self.reserve, self.timeout_s = per_call_limit, reserve, timeout_s
        self.log_path, self.dry_run = log_path, dry_run
        self._models: dict | None = None

    # ---- HTTP ---------------------------------------------------------------------------------
    def _request(self, method: str, path: str, body: dict | None = None, retries: int = 3) -> dict:
        if not self.key:
            raise LumenError("LUMEN_API_KEY is not set (source env.sh)")
        data = json.dumps(body).encode() if body is not None else None
        for attempt in range(retries + 1):
            req = urllib.request.Request(self.base + path, data=data, method=method, headers={
                "Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as e:
                text = e.read().decode(errors="replace")
                if e.code == 403 and "consent_required" in text:
                    raise ConsentRequired(f"model needs acknowledgment on the Lumen web UI: {text[:300]}", 403, text)
                if e.code in (429, 500, 502, 503, 504) and attempt < retries:
                    time.sleep(2 ** attempt)
                    continue
                raise LumenError(f"HTTP {e.code} on {method} {path}: {text[:300]}", e.code, text) from None
            except urllib.error.URLError as e:
                if attempt < retries:
                    time.sleep(2 ** attempt)
                    continue
                raise LumenError(f"network error on {method} {path}: {e.reason}") from None
        raise LumenError("unreachable")

    # ---- free endpoints -----------------------------------------------------------------------
    def models(self, refresh: bool = False) -> dict:
        if self._models is None or refresh:
            data = self._request("GET", "/models?include_unacknowledged=true")["data"]
            self._models = {m["id"]: m for m in data}
        return self._models

    def usage(self) -> dict:
        return self._request("GET", "/usage")

    def price(self, model: str | None = None) -> tuple[float, float]:
        m = self.models()[model or self.model]
        return float(m["input_cost_per_million"]), float(m["output_cost_per_million"])

    # ---- chat ---------------------------------------------------------------------------------
    @staticmethod
    def estimate_tokens(messages: list[dict]) -> int:
        return sum(len(str(m.get("content", ""))) for m in messages) // 4 + 8 * len(messages)

    def check_budget(self, messages: list[dict], max_tokens: int, model: str) -> dict:
        p_in, p_out = self.price(model)
        est_in = self.estimate_tokens(messages)
        worst = est_in / 1e6 * p_in + max_tokens / 1e6 * p_out
        if worst > self.per_call_limit:
            raise BudgetError(f"worst-case cost {worst:.4f} coins > per-call limit {self.per_call_limit}")
        avail = self.usage().get("coins_available")
        if avail is not None and avail != -2 and avail - worst < self.reserve:
            raise BudgetError(f"only {avail:.3f} coins left; worst case {worst:.4f} would go below reserve {self.reserve}")
        return {"est_input_tokens": est_in, "worst_case_cost": worst, "coins_available": avail}

    def chat(self, messages: list[dict], *, model: str | None = None, max_tokens: int = 8192,
             temperature: float | None = None, dry_run: bool | None = None, tag: str = "", **extra) -> ChatResult:
        model = model or self.model
        dry = self.dry_run if dry_run is None else dry_run
        budget = self.check_budget(messages, max_tokens, model)
        body = {"model": model, "messages": messages, "max_tokens": max_tokens, "stream": False, **extra}
        if temperature is not None:
            body["temperature"] = temperature
        t0 = time.perf_counter()
        if dry:
            res = ChatResult(model, "[dry-run: request not sent]", input_tokens=budget["est_input_tokens"], dry_run=True)
        else:
            raw = self._request("POST", "/chat/completions", body)
            msg = raw["choices"][0]["message"]
            usage = raw.get("usage") or {}
            p_in, p_out = self.price(model)
            res = ChatResult(
                model=raw.get("model", model), content=msg.get("content") or "",
                reasoning=msg.get("reasoning_content") or msg.get("reasoning") or "",
                input_tokens=int(usage.get("prompt_tokens", 0)), output_tokens=int(usage.get("completion_tokens", 0)),
                finish_reason=raw["choices"][0].get("finish_reason") or "", raw=raw)
            res.cost = res.input_tokens / 1e6 * p_in + res.output_tokens / 1e6 * p_out
        res.latency_s = time.perf_counter() - t0
        self._log({"tag": tag, "request": body, "budget": budget, **{k: v for k, v in asdict(res).items() if k != "raw"}})
        return res

    def _log(self, rec: dict) -> None:
        if not self.log_path:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        rec = {"time": datetime.now(timezone.utc).isoformat(), **rec}
        with open(self.log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["models", "usage", "chat"])
    ap.add_argument("prompt", nargs="?")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    c = LumenClient(model=a.model)
    if a.cmd == "models":
        for mid, m in sorted(c.models().items()):
            if not m.get("parent"):
                print(f"{mid:30s} in {m['input_cost_per_million']:.3f}  out {m['output_cost_per_million']:.3f}  "
                      f"ctx {m.get('max_model_len')}  tags {m.get('tags')}")
    elif a.cmd == "usage":
        print(json.dumps(c.usage(), indent=2))
    else:
        r = c.chat([{"role": "user", "content": a.prompt or "Say hello in five words."}],
                   max_tokens=a.max_tokens, dry_run=a.dry_run, tag="cli")
        print(r.content)
        print(f"\n[{r.model}] in={r.input_tokens} out={r.output_tokens} cost={r.cost:.5f} coins "
              f"latency={r.latency_s:.1f}s dry_run={r.dry_run}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
