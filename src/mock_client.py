"""Offline mock of ``anthropic.AsyncAnthropic`` (the ``--mock`` flag).

It reads the structured-output schema and the slot briefs from each request and
fabricates a *schema-shaped placeholder* response, the way a (configurably
imperfect) model would. Values are gibberish on purpose: the mock exists to run
the full pipeline - planning, batching, validation, export, quality report -
without an API key, and to drive the test suite. It is not a data source.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import random
import re
import uuid
from types import SimpleNamespace
from typing import Any

WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november "
         "oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu").split()


def parse_briefs(prompt: str) -> dict[int, dict[str, Any]]:
    briefs: dict[int, dict[str, Any]] = {}
    for block in re.split(r"\n\s*\n", prompt):
        m = re.match(r"Slot (\d+):", block.strip())
        if not m:
            continue
        b: dict[str, Any] = {"fixed": {}, "targets": {}, "nulls": []}
        for line in block.splitlines()[1:]:
            line = line.strip()
            for key, dest in (("fixed values:", "fixed"), ("target values:", "targets")):
                if line.startswith(key):
                    for part in line[len(key):].split("; "):
                        k, v = part.split(" = ", 1)
                        b[dest][k.strip()] = json.loads(v)
            if line.startswith("null fields:"):
                b["nulls"] = [x.strip() for x in line[len("null fields:"):].split(",")]
        briefs[int(m.group(1))] = b
    return briefs


def parse_checks(system: str) -> list[str]:
    m = re.search(r"<hard_checks>\n.*?\n(.*?)\n</hard_checks>", system, re.S)
    if not m:
        return []
    return [line[2:] for line in m.group(1).splitlines() if line.startswith("- ")]


def _checks_pass(rec: dict[str, Any], checks: list[str]) -> bool:
    ns: dict[str, Any] = {}
    for k, v in rec.items():
        if isinstance(v, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", v):
            try:
                v = dt.date.fromisoformat(v)
            except ValueError:
                pass
        ns[k] = v
    ns["record"] = rec
    env = {"__builtins__": {"len": len, "set": set, "sum": sum, "abs": abs, "min": min, "max": max, "str": str,
                            "any": any, "all": all, "round": round, "int": int, "float": float},
           "date": dt.date, "datetime": dt.datetime, "timedelta": dt.timedelta, "re": re}
    for c in checks:
        if "parent" in c:
            continue
        try:
            if not eval(c, env, ns):  # noqa: S307 - checks come from the local schema file
                return False
        except Exception:  # noqa: BLE001
            return False
    return True


class MockModel:
    def __init__(self, seed: int = 0, invalid_rate: float = 0.0, disobey_rate: float = 0.0,
                 shuffle: bool = False, wrong_case_rate: float = 0.0, duplicate_rate: float = 0.0,
                 truncate_every: int = 0, refuse_every: int = 0, extra_record: bool = False,
                 value_hook: Any = None, honor_lengths: bool = True, obey_checks: bool = False):
        self.rng = random.Random(seed)
        self.invalid_rate = invalid_rate
        self.disobey_rate = disobey_rate
        self.shuffle = shuffle
        self.wrong_case_rate = wrong_case_rate
        self.duplicate_rate = duplicate_rate
        self.truncate_every = truncate_every
        self.refuse_every = refuse_every
        self.extra_record = extra_record
        self.value_hook = value_hook  # fn(field_name, value, record) -> value
        self.honor_lengths = honor_lengths
        self.obey_checks = obey_checks
        self.calls: list[dict[str, Any]] = []
        self._last: dict[str, Any] | None = None

    # ---- value synthesis ----------------------------------------------------------
    def _bounds(self, desc: str) -> tuple[str | None, str | None]:
        m = re.search(r"between (\S+) and (\S+) inclusive", desc or "")
        if m:
            return m.group(1), m.group(2)
        lo = re.search(r"at least (\S+?)[;.]", desc or "")
        hi = re.search(r"at most (\S+?)[;.]", desc or "")
        return (lo.group(1) if lo else None), (hi.group(1) if hi else None)

    def value(self, schema: dict[str, Any], name: str = "") -> Any:
        r = self.rng
        if "anyOf" in schema:
            if "null only when" not in schema.get("description", "") and r.random() < 0.2:
                return None
            inner = dict(schema["anyOf"][0])
            inner.setdefault("description", schema.get("description", ""))
            return self.value(inner, name)
        t = schema.get("type")
        desc = schema.get("description", "")
        if "enum" in schema:
            v = r.choice(schema["enum"])
            if isinstance(v, str) and r.random() < self.wrong_case_rate:
                v = v.upper()
            return v
        if t == "object":
            return {k: self.value(s, k) for k, s in schema["properties"].items() if k != "_slot"}
        if t == "array":
            m = re.search(r"(\d+)-(\d+) items", desc)
            lo, hi = (int(m.group(1)), int(m.group(2))) if m else (1, 3)
            return [self.value(schema["items"], name) for _ in range(r.randint(lo, hi))]
        if t == "boolean":
            return r.random() < 0.5
        if t in ("integer", "number"):
            lo, hi = self._bounds(desc)
            a = float(lo) if lo else 0.0
            b = float(hi) if hi else a + 100
            x = r.uniform(a, b)
            return int(round(x)) if t == "integer" else round(x, 2)
        fmt = schema.get("format")
        lo, hi = self._bounds(desc)
        if fmt == "date" and lo and hi:
            a, b = dt.date.fromisoformat(lo), dt.date.fromisoformat(hi)
            return (a + dt.timedelta(days=r.randint(0, (b - a).days))).isoformat()
        if fmt == "date":
            return (dt.date(2025, 1, 1) + dt.timedelta(days=r.randint(0, 300))).isoformat()
        if fmt == "date-time":
            return f"2025-0{r.randint(1, 9)}-1{r.randint(0, 9)}T10:00:00Z"
        if fmt == "email":
            return f"{r.choice(WORDS)}.{r.randint(1, 10**6)}@example.org"
        if fmt == "uri":
            return f"https://example.org/{r.randint(1, 10**6)}"
        if fmt == "uuid":
            return str(uuid.UUID(int=r.getrandbits(128), version=4))
        n = r.randint(3, 12) if "text" in desc else r.randint(2, 4)
        s = " ".join(r.choice(WORDS) for _ in range(n)) + f" {r.randint(1, 10**9)}"
        lo_len = re.search(r"(\d+)-\d+ characters|at least (\d+) characters", desc)
        if self.honor_lengths and lo_len:
            want = int(lo_len.group(1) or lo_len.group(2))
            while len(s) < want:
                s += " " + r.choice(WORDS)
        m = re.search(r"(?:at most |\d+-)(\d+) characters", desc)
        if m:
            s = s[: int(m.group(1))].strip()
        return s

    def make_record(self, item_schema: dict[str, Any], slot: int, brief: dict[str, Any],
                    checks: list[str] | None = None) -> dict[str, Any]:
        """Build a record; when ``obey_checks`` is on, resample until the schema's hard checks pass
        (a crude stand-in for a model that reads its instructions)."""
        rec = self._make_record(item_schema, slot, brief)
        if not (self.obey_checks and checks):
            return rec
        for _ in range(80):
            if _checks_pass(rec, checks):
                return rec
            rec = self._make_record(item_schema, slot, brief)
        return rec

    def _make_record(self, item_schema: dict[str, Any], slot: int, brief: dict[str, Any]) -> dict[str, Any]:
        rec = {"_slot": slot}
        for k, s in item_schema["properties"].items():
            if k == "_slot":
                continue
            v = self.value(s, k)
            if k in brief.get("nulls", []):
                v = None
            if k in brief.get("fixed", {}) and self.rng.random() >= self.disobey_rate:
                v = brief["fixed"][k]
            if k in brief.get("targets", {}) and self.rng.random() >= self.disobey_rate:
                v = brief["targets"][k]
            if self.value_hook:
                v = self.value_hook(k, v, rec)
            rec[k] = v
        if self.rng.random() < self.invalid_rate:
            k = self.rng.choice([k for k in rec if k != "_slot"])
            rec[k] = {"clearly": "wrong"}
        return rec

    # ---- API surface -----------------------------------------------------------------
    def respond(self, params: dict[str, Any]) -> SimpleNamespace:
        self.calls.append(params)
        n_call = len(self.calls)
        tool_mode = "tools" in params
        schema = params["tools"][0]["input_schema"] if tool_mode else params["output_config"]["format"]["schema"]
        prompt = params["messages"][-1]["content"]
        usage = SimpleNamespace(input_tokens=1000, output_tokens=0, cache_read_input_tokens=800,
                                cache_creation_input_tokens=0)

        if "scenarios" in schema["properties"]:
            m = re.search(r"Propose (\d+) distinct scenarios", prompt)
            n = int(m.group(1)) if m else 5
            text = json.dumps({"scenarios": [f"scenario {n_call}-{i}: {self.rng.choice(WORDS)}"
                                             for i in range(n)]})
            usage.output_tokens = 50 * n
            return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)],
                                   stop_reason="end_turn", usage=usage)

        if self.refuse_every and n_call % self.refuse_every == 0:
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="I can't help with that.")],
                                   stop_reason="refusal", usage=usage)

        briefs = parse_briefs(prompt)
        item_schema = schema["properties"]["records"]["items"]
        system = params.get("system") or ""
        system = system[0]["text"] if isinstance(system, list) else system
        checks = parse_checks(system)
        recs = [self.make_record(item_schema, i, b, checks) for i, b in sorted(briefs.items())]
        if self.duplicate_rate and self._last and self.rng.random() < self.duplicate_rate and recs:
            dup = dict(self._last)
            dup["_slot"] = recs[0]["_slot"]
            recs[0] = dup
        if recs:
            self._last = dict(recs[-1])
        if self.extra_record and recs:
            recs.append(dict(recs[0], _slot=999))
        if self.shuffle:
            self.rng.shuffle(recs)
        usage.output_tokens = max(1, len(json.dumps(recs)) // 4)  # ~4 characters per token
        truncate = bool(self.truncate_every and n_call % self.truncate_every == 0)
        if tool_mode:
            if truncate:
                recs = recs[: len(recs) // 2]
            block = SimpleNamespace(type="tool_use", id=f"toolu_{n_call}", name=params["tools"][0]["name"],
                                    input={"records": recs})
            return SimpleNamespace(content=[block], stop_reason="max_tokens" if truncate else "tool_use",
                                   usage=usage)
        text = json.dumps({"records": recs})
        stop = "end_turn"
        if truncate:
            text = text[: int(len(text) * 0.6)]
            stop = "max_tokens"
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], stop_reason=stop, usage=usage)


class _Messages:
    def __init__(self, model: MockModel, latency: float, tokens_per_second: float | None = None,
                 ttft: float = 0.0):
        self.model = model
        self.latency = latency
        self.tokens_per_second = tokens_per_second
        self.ttft = ttft
        self.batches = _Batches(model)

    async def create(self, **params: Any) -> SimpleNamespace:
        msg = self.model.respond(params)
        if self.tokens_per_second:  # latency model: time to first token + streaming time
            await asyncio.sleep(self.ttft + msg.usage.output_tokens / self.tokens_per_second)
        else:
            await asyncio.sleep(self.latency * self.model.rng.random())
        return msg


class _Batches:
    def __init__(self, model: MockModel):
        self.model = model
        self.store: dict[str, list[dict[str, Any]]] = {}
        self.polls = 0

    async def create(self, requests: list[dict[str, Any]]) -> SimpleNamespace:
        bid = f"msgbatch_{len(self.store) + 1}"
        self.store[bid] = requests
        return SimpleNamespace(id=bid, processing_status="in_progress")

    async def retrieve(self, batch_id: str) -> SimpleNamespace:
        self.polls += 1
        return SimpleNamespace(id=batch_id, processing_status="ended" if self.polls % 2 == 0 else "in_progress")

    async def results(self, batch_id: str) -> Any:
        reqs = self.store[batch_id]
        model = self.model

        async def gen():
            for r in reversed(reqs):  # out of order, like the real API
                msg = model.respond(r["params"])
                yield SimpleNamespace(custom_id=r["custom_id"],
                                      result=SimpleNamespace(type="succeeded", message=msg))
        return gen()


class MockAsyncAnthropic:
    def __init__(self, latency: float = 0.002, tokens_per_second: float | None = None, ttft: float = 0.0,
                 **model_kwargs: Any):
        """``latency``: max simulated seconds per request (uniform). Or give ``tokens_per_second`` (+ ``ttft``)
        to simulate latency proportional to output length, like a real model."""
        self.model = MockModel(**model_kwargs)
        self.messages = _Messages(self.model, latency, tokens_per_second, ttft)
