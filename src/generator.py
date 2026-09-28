"""Claude-powered data generation.

Per table the loop is:

1. **Plan** (optional): Claude proposes N distinct scenarios covering the
   dataset's space; each record slot is seeded with one.
2. **Allocate slots**: code decides each slot's fixed values (distributions),
   target labels (quota planner), nulls, parent rows and edge-case flag.
3. **Generate**: one request per batch of slots, many in flight at once, using
   structured outputs - JSON outputs (``output_config.format``) or strict tool
   use - so every response matches the batch schema. Alternatively all
   requests go out as one Message Batch (asynchronous, ~50% cheaper).
4. **Validate & repair**: salvage truncated JSON, match records to slots,
   enforce fixed values, run the Pydantic record model, checks, uniqueness
   and quotas. Rejected records are simply regenerated.
5. **Checkpoint**: accepted records are appended to disk immediately.

The loop self-corrects: quotas re-target under-filled labels, batch size adapts
to observed output length, avoid-lists steer Claude away from values already
used, and the most common rejection reasons are fed back into later prompts.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import math
import random
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Callable

from .prompts import SCENARIO_SCHEMA, scenario_prompt, system_prompt, user_prompt
from .sampling import Sampler, Slot
from .schema_builder import (SLOT_KEY, TOOL_NAME, DatasetSchema, Settings, TableSpec, batch_schema, complexity,
                             constraint_notes, output_config, to_jsonable, tool_definition)
from .storage import Store
from .validator import CheckRunner, RecordError, record_errors

ProgressFn = Callable[[str, int, int, "Stats"], None]
LogFn = Callable[[str], None]

_AVOID_SAMPLE = 40


class GenerationError(RuntimeError):
    pass


@dataclass
class Stats:
    requests: int = 0
    failed_requests: int = 0
    refusals: int = 0
    truncated: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    returned: int = 0
    accepted: int = 0
    fixed_repaired: int = 0
    target_mismatch: int = 0
    surplus: int = 0
    rejected: Counter = field(default_factory=Counter)
    started: float = field(default_factory=time.time)
    elapsed: float = 0.0
    finished: bool = False

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.returned if self.returned else 0.0

    @property
    def records_per_second(self) -> float:
        return self.accepted / self.elapsed if self.elapsed > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["rejected"] = dict(self.rejected.most_common())
        d["acceptance_rate"] = round(self.acceptance_rate, 4)
        d["elapsed"] = round(self.elapsed, 2)
        for k in ("started", "finished"):
            d.pop(k, None)
        return d

    def merge_saved(self, saved: dict[str, Any]) -> None:
        for k, v in saved.items():
            if k == "rejected":
                self.rejected.update(v)
            elif k != "accepted" and isinstance(getattr(self, k, None), int) and isinstance(v, int) \
                    and not isinstance(v, bool):
                setattr(self, k, getattr(self, k) + v)


def salvage_records(text: str) -> list[Any]:
    """Recover complete records from a truncated ``{"records": [ ...`` response."""
    start = text.find("[", max(0, text.find('"records"')))
    if start < 0:
        return []
    dec = json.JSONDecoder()
    i, out, n = start + 1, [], len(text)
    while i < n:
        while i < n and text[i] in " \t\r\n,":
            i += 1
        if i >= n or text[i] == "]":
            break
        try:
            obj, i = dec.raw_decode(text, i)
        except json.JSONDecodeError:
            break
        out.append(obj)
    return out


def extract_records(message: Any) -> list[Any]:
    """Records from a JSON-outputs text block or a strict tool_use block."""
    for b in message.content:
        if getattr(b, "type", None) == "tool_use" and getattr(b, "name", "") == TOOL_NAME:
            inp = getattr(b, "input", None) or {}
            recs = inp.get("records") if isinstance(inp, dict) else None
            return recs if isinstance(recs, list) else []
    text = "".join(getattr(b, "text", "") for b in message.content if getattr(b, "type", None) == "text")
    try:
        items = json.loads(text)["records"]
        if isinstance(items, list):
            return items
    except (json.JSONDecodeError, KeyError, TypeError):
        pass
    return salvage_records(text)


def _norm_key(v: Any) -> str:
    if isinstance(v, str):
        return " ".join(v.lower().split())
    return json.dumps(v, sort_keys=True, ensure_ascii=False, default=str).lower()


class TableGenerator:
    """Generates one table."""

    def __init__(self, table: TableSpec, settings: Settings, client: Any, store: Store | None = None,
                 parents: dict[str, list[dict[str, Any]]] | None = None, target: int | None = None,
                 on_progress: ProgressFn | None = None, log: LogFn | None = None,
                 scenarios: list[str] | None = None):
        self.table = table
        self.settings = settings
        self.client = client
        self.store = store or Store(None)
        self.target = target or table.count
        self.on_progress = on_progress
        self.log = log or (lambda msg: None)
        seed = f"{settings.seed}-{table.name}" if settings.seed is not None else None
        self.sampler = Sampler(table, self.target, seed, parents)
        self.preset_scenarios = scenarios
        self.checks = CheckRunner(table)
        self.stats = Stats()
        self.records: list[dict[str, Any]] = []
        self.batch_size = table.batch_size
        self._tpr: float | None = None
        self._seen: dict[str, set[str]] = {k: set() for k, f in table.unique_fields.items() if not f.generator}
        self._hashes: set[str] = set()
        avoid = list(dict.fromkeys(table.diversity.avoid_repeats + [
            k for k, f in table.unique_fields.items()
            if not f.code_generated and f.type in ("string", "text", "email", "url")]))
        self._recent: dict[str, deque[str]] = {k: deque(maxlen=400) for k in avoid}
        self.schema = batch_schema(table)
        self.system = [{"type": "text", "cache_control": {"type": "ephemeral"},
                        "text": system_prompt(table, privacy=settings.privacy_guardrails, mode=settings.mode)}]
        for w in complexity(self.schema)["warnings"]:
            self.log(f"[{table.name}] warning: {w}")

    # ------------------------------------------------------------------ requests
    def feedback(self) -> list[str]:
        """Top validation failures so far, fed back into prompts so Claude corrects course."""
        st = self.stats
        if st.returned < 10 or st.accepted / max(1, st.returned) > 0.9:
            return []
        out = []
        for cat, n in st.rejected.most_common(4):
            if n / st.returned < 0.03 or "duplicate" in cat or "quota" in cat:
                continue
            fname = cat.split(":", 1)[0].split(".", 1)[0].split("[", 1)[0]
            f = self.table.field_map.get(fname)
            notes = constraint_notes(f) if f is not None else []
            hint = f" (requirement: {'; '.join(notes)})" if notes else ""
            out.append(f"{cat}{hint} - {n / st.returned:.0%} of records so far")
        return out

    def request_params(self, slots: list[Slot]) -> dict[str, Any]:
        avoid = {}
        for k, dq in self._recent.items():
            if dq:
                pool = list(dq)
                avoid[k] = self.sampler.aux_rng.sample(pool, min(_AVOID_SAMPLE, len(pool)))
        params: dict[str, Any] = {
            "model": self.settings.model,
            "max_tokens": self.settings.max_tokens,
            "system": self.system,
            "messages": [{"role": "user", "content": user_prompt(slots, self.table, avoid, self.feedback())}],
        }
        if self.settings.mode == "tool":
            params["tools"] = [tool_definition(self.schema)]
            params["tool_choice"] = {"type": "tool", "name": TOOL_NAME}
        else:
            params["output_config"] = output_config(self.schema)
        if self.settings.temperature is not None:
            params["temperature"] = self.settings.temperature
        return params

    def _account(self, message: Any) -> None:
        u = getattr(message, "usage", None)
        if u is None:
            return
        self.stats.input_tokens += getattr(u, "input_tokens", 0) or 0
        self.stats.output_tokens += getattr(u, "output_tokens", 0) or 0
        self.stats.cache_read_tokens += getattr(u, "cache_read_input_tokens", 0) or 0
        self.stats.cache_write_tokens += getattr(u, "cache_creation_input_tokens", 0) or 0

    def effective_batch_size(self) -> int:
        bs = self.batch_size
        if self._tpr:
            bs = min(bs, max(1, int(0.75 * self.settings.max_tokens / self._tpr)))
        return max(1, bs)

    # ------------------------------------------------------------------ planning
    async def plan_scenarios(self) -> list[str]:
        if self.preset_scenarios:
            self.sampler.set_scenarios(self.preset_scenarios)
            self._save_state(scenarios=self.preset_scenarios)
            return self.preset_scenarios
        state = self.store.load_state(self.table.name)
        if state.get("scenarios"):
            self.sampler.set_scenarios(state["scenarios"])
            return state["scenarios"]
        n = self.table.diversity.scenarios
        if n <= 0:
            return []
        scenarios: list[str] = []
        attempts = 0
        while len(scenarios) < n and attempts < math.ceil(n / 40) + 2:
            attempts += 1
            prompt = scenario_prompt(self.table, min(40, n - len(scenarios)))
            if scenarios:
                prompt += "\n\nAlready planned (propose different ones):\n" + "\n".join(f"- {s}" for s in scenarios)
            try:
                msg = await self.client.messages.create(
                    model=self.settings.model, max_tokens=8000, system=self.system,
                    messages=[{"role": "user", "content": prompt}], output_config=output_config(SCENARIO_SCHEMA))
            except Exception as e:  # planning is best-effort
                self._raise_if_fatal(e)
                self.log(f"[{self.table.name}] scenario planning failed ({type(e).__name__}); continuing without")
                break
            self.stats.requests += 1
            self._account(msg)
            if msg.stop_reason == "refusal":
                break
            text = "".join(getattr(b, "text", "") for b in msg.content if getattr(b, "type", None) == "text")
            try:
                got = json.loads(text)["scenarios"]
            except (json.JSONDecodeError, KeyError, TypeError):
                got = salvage_records(text.replace('"scenarios"', '"records"'))
            seen = {s.lower() for s in scenarios}
            scenarios += [s.strip() for s in got if isinstance(s, str) and s.strip() and s.lower() not in seen]
        scenarios = scenarios[:n]
        if scenarios:
            self.sampler.set_scenarios(scenarios)
            self._save_state(scenarios=scenarios)
            self.log(f"[{self.table.name}] planned {len(scenarios)} scenarios")
        return scenarios

    @property
    def scenarios(self) -> list[str]:
        return list(self.sampler.scenarios)

    # ------------------------------------------------------------------ bookkeeping
    def _save_state(self, **extra: Any) -> None:
        state = self.store.load_state(self.table.name)
        state.update(extra)
        state["stats"] = self.stats.to_dict()
        state["target"] = self.target
        self.store.save_state(self.table.name, state)

    def _content_hash(self, rec: dict[str, Any]) -> str:
        return _norm_key({k: rec.get(k) for k, f in self.table.llm_fields.items() if not f.sampled})

    def _register(self, rec: dict[str, Any]) -> None:
        for k, q in self.sampler.quotas.items():
            if rec.get(k) is not None:
                q.accept(rec[k])
        for k in self._seen:
            if rec.get(k) is not None:
                self._seen[k].add(_norm_key(rec[k]))
        for k, dq in self._recent.items():
            if rec.get(k) is not None:
                dq.append(str(rec[k]))
        self._hashes.add(self._content_hash(rec))

    def load_existing(self) -> None:
        rows = self.store.load(self.table.name)
        for r in rows[: self.target]:
            self.records.append(r)
            self._register(r)
        if len(rows) != len(self.records) or self.table.name in self.store.torn:
            self.store.rewrite(self.table.name, self.records)
        for k, f in self.table.field_map.items():
            if f.generator == "sequence":
                self.sampler.sequence_next[k] = f.start + len(self.records)
        state = self.store.load_state(self.table.name)
        if "stats" in state:
            self.stats.merge_saved(state["stats"])
        self.stats.accepted = len(self.records)

    # ------------------------------------------------------------------ records
    def _assemble(self, slot: Slot, item: dict[str, Any]) -> dict[str, Any]:
        rec: dict[str, Any] = {}
        seq_peek = self.sampler.sequence_next
        for k, f in self.table.field_map.items():
            if f.generator == "uuid":
                rec[k] = self.sampler.generate_code_value(k, f)
            elif f.generator == "sequence":  # peek; committed only on acceptance (gapless)
                n = seq_peek[k]
                rec[k] = (f.format.format(n) if f.format else str(n)) if f.type == "string" else n
            elif f.ref:
                rec[k] = slot.refs.get(k)
            elif k in slot.nulls:
                rec[k] = None
            elif k in slot.fixed:
                want = to_jsonable(slot.fixed[k])
                if item.get(k) != want:
                    self.stats.fixed_repaired += 1
                rec[k] = want
            else:
                if k not in item:
                    raise RecordError(k, "Field required")
                rec[k] = item[k]
        return rec

    def _validate(self, slot: Slot, rec: dict[str, Any]) -> dict[str, Any]:
        out, errs = record_errors(self.table, rec)
        if errs:
            raise errs[0]
        assert out is not None
        for k, f in self.table.field_map.items():
            if out.get(k) is None and f.null_rate is not None and k not in slot.nulls:
                raise RecordError(k, "unexpected null (null_rate controls missingness)")
        for k, target in slot.targets.items():
            if out.get(k) is not None and str(out[k]).lower() != str(target).lower():
                self.stats.target_mismatch += 1
        if self.table.strict_distribution:
            for k, q in self.sampler.quotas.items():
                if out.get(k) is not None and q.is_full(out[k]):
                    raise RecordError(k, "quota already full (strict_distribution)")
        for k in self._seen:
            if out.get(k) is not None and _norm_key(out[k]) in self._seen[k]:
                raise RecordError(k, "duplicate value")
        if self._content_hash(out) in self._hashes:
            raise RecordError("record", "duplicate record")
        self.checks.run(out, parent=slot.parent_context)
        return out

    def process(self, slots: list[Slot], message: Any) -> int:
        """Validate one response; returns the number of accepted records."""
        for s in slots:
            self.sampler.release_slot(s)
        self._account(message)
        stop = getattr(message, "stop_reason", None)
        if stop == "refusal":
            self.stats.refusals += 1
            self.log(f"[{self.table.name}] a request was refused; its slots will be regenerated")
            return 0
        items = extract_records(message)
        if stop == "max_tokens":
            self.stats.truncated += 1
            self.batch_size = max(1, int(len(slots) * 0.6))
            self.log(f"[{self.table.name}] response hit max_tokens; salvaged {len(items)} records, "
                     f"batch size -> {self.batch_size}")
        elif items:
            tpr = (getattr(getattr(message, "usage", None), "output_tokens", 0) or 0) / len(items)
            if tpr:
                self._tpr = tpr if self._tpr is None else 0.7 * self._tpr + 0.3 * tpr

        # match records to slots by _slot, falling back to position
        by_index = {s.index: s for s in slots}
        used: set[int] = set()
        pairs: list[tuple[Slot, dict[str, Any]]] = []
        leftovers: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                self.stats.returned += 1
                self.stats.rejected["record: malformed"] += 1
                continue
            item = dict(item)
            sid = item.pop(SLOT_KEY, None)
            if isinstance(sid, int) and sid in by_index and sid not in used:
                used.add(sid)
                pairs.append((by_index[sid], item))
            else:
                leftovers.append(item)
        free = [s for s in slots if s.index not in used]
        for item in leftovers:
            if free:
                pairs.append((free.pop(0), item))
            else:
                self.stats.returned += 1
                self.stats.rejected["record: extra record"] += 1

        accepted_now: list[dict[str, Any]] = []
        for slot, item in pairs:
            self.stats.returned += 1
            if len(self.records) >= self.target:
                self.stats.surplus += 1
                continue
            try:
                rec = self._validate(slot, self._assemble(slot, item))
            except RecordError as e:
                self.stats.rejected[e.category] += 1
                continue
            for k, f in self.table.field_map.items():
                if f.generator == "sequence":
                    committed = self.sampler.generate_code_value(k, f)
                    rec[k] = str(committed) if f.type == "string" and not isinstance(committed, str) else committed
            self.records.append(rec)
            self._register(rec)
            accepted_now.append(rec)
        self.stats.accepted = len(self.records)
        self.store.append(self.table.name, accepted_now)
        return len(accepted_now)

    # ------------------------------------------------------------------ execution
    @staticmethod
    def _raise_if_fatal(e: Exception) -> None:
        try:
            import anthropic
        except ImportError:  # pragma: no cover
            return
        fatal = (anthropic.BadRequestError, anthropic.AuthenticationError, anthropic.PermissionDeniedError,
                 anthropic.NotFoundError)
        if isinstance(e, fatal):
            msg = getattr(e, "message", str(e))
            hint = ""
            if "too complex" in str(msg).lower():
                hint = " (flatten nested objects, reduce nullable fields, or split the table)"
            elif isinstance(e, anthropic.AuthenticationError):
                hint = " (check ANTHROPIC_API_KEY)"
            raise GenerationError(f"API rejected the request: {msg}{hint}") from e

    async def _call(self, slots: list[Slot]) -> tuple[list[Slot], Any, Exception | None]:
        try:
            return slots, await self.client.messages.create(**self.request_params(slots)), None
        except Exception as e:  # noqa: BLE001
            return slots, None, e

    def _progress(self) -> None:
        self.stats.elapsed = time.time() - self.stats.started
        if self.on_progress:
            self.on_progress(self.table.name, len(self.records), self.target, self.stats)

    def _stall_message(self, why: str) -> str:
        top = ", ".join(f"{k} x{v}" for k, v in self.stats.rejected.most_common(5)) or "none recorded"
        return (f"[{self.table.name}] {why}. Accepted {len(self.records)}/{self.target}. "
                f"Refusals: {self.stats.refusals}, failed requests: {self.stats.failed_requests}. "
                f"Top rejection reasons: {top}. Loosen the constraints/checks or clarify the rules.")

    def _yield(self) -> float:
        """Expected share of requested slots that end up accepted (for over-provisioning)."""
        st = self.stats
        if st.returned < 20:
            return 1.0
        return min(1.0, max(0.5, st.acceptance_rate))

    def _budget(self) -> int:
        """Cap on requested record slots: adapts to the observed acceptance rate (low-but-steady acceptance may continue,
        pathological schemas stop), bounded at 20x the target."""
        st = self.stats
        factor = 4.0 if st.returned < 20 else min(20.0, max(4.0, 1.5 / max(st.acceptance_rate, 1e-3)))
        return int(self.target * factor) + 2 * self.table.batch_size

    async def run(self, resume: bool = False) -> list[dict[str, Any]]:
        """Realtime mode: ``concurrency`` requests in flight."""
        if resume:
            self.load_existing()
        else:
            self.store.reset(self.table.name)
        await self.plan_scenarios()
        self._progress()
        if len(self.records) >= self.target:
            return self.records

        pending: set[asyncio.Task] = set()
        in_flight = zero_streak = slots_requested = 0
        try:
            while len(self.records) < self.target:
                while (len(pending) < self.settings.concurrency and slots_requested < self._budget()
                       and len(self.records) + in_flight * self._yield() < self.target - 1e-9):
                    # over-provision by the expected rejection rate so the tail doesn't need extra round trips
                    need = (self.target - len(self.records) - in_flight * self._yield()) / self._yield()
                    n = max(1, min(self.effective_batch_size(), math.ceil(need)))
                    slots = [self.sampler.make_slot(i + 1) for i in range(n)]
                    in_flight += n
                    slots_requested += n
                    pending.add(asyncio.create_task(self._call(slots)))
                if not pending:
                    break
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    slots, msg, err = task.result()
                    in_flight -= len(slots)
                    self.stats.requests += 1
                    if err is not None:
                        for s in slots:
                            self.sampler.release_slot(s)
                        self._raise_if_fatal(err)
                        self.stats.failed_requests += 1
                        self.log(f"[{self.table.name}] request failed: {type(err).__name__}: {err}")
                        got = 0
                    else:
                        got = self.process(slots, msg)
                    zero_streak = 0 if got else zero_streak + 1
                    self._progress()
                if zero_streak >= max(6, self.settings.concurrency * 2):
                    raise GenerationError(self._stall_message("consecutive requests produced no valid records"))
            if len(self.records) < self.target:
                self.log(f"[{self.table.name}] stopped at {len(self.records)}/{self.target}: request budget "
                         f"exhausted. Top rejection reasons: {self.stats.rejected.most_common(3)}")
        finally:
            for t in pending:
                t.cancel()
            self.stats.finished = True
            self._progress()
            self._save_state()
        return self.records

    async def run_batch_api(self, resume: bool = False, poll_interval: float = 30.0,
                            max_rounds: int = 10) -> list[dict[str, Any]]:
        """Message Batches mode: asynchronous, ~50% cheaper; tops up rejected records in later rounds."""
        if resume:
            self.load_existing()
        else:
            self.store.reset(self.table.name)
        await self.plan_scenarios()
        self._progress()
        state = self.store.load_state(self.table.name)
        pending = state.get("pending_batch") if resume else None
        rounds = idle_rounds = 0
        try:
            while len(self.records) < self.target and rounds < max_rounds and idle_rounds < 2:
                before = len(self.records)
                if pending:
                    batch_id = pending["id"]
                    mapping = {cid: [Slot(**s) for s in slots] for cid, slots in pending["slots"].items()}
                    for slots in mapping.values():  # re-reserve quota for in-flight slots
                        for s in slots:
                            for k, v in s.targets.items():
                                if k in self.sampler.quotas:
                                    q = self.sampler.quotas[k]
                                    q.reserved[q.key(v)] += 1
                    self.log(f"[{self.table.name}] resuming Message Batch {batch_id}")
                else:
                    rounds += 1
                    remaining = self.target - len(self.records)
                    mapping, requests, planned, i = {}, [], 0, 0
                    while planned < remaining:
                        n = min(self.effective_batch_size(), remaining - planned)
                        slots = [self.sampler.make_slot(j + 1) for j in range(n)]
                        cid = f"{self.table.name}-r{rounds}-{i}"
                        mapping[cid] = slots
                        requests.append({"custom_id": cid, "params": self.request_params(slots)})
                        planned += n
                        i += 1
                    batch = await self.client.messages.batches.create(requests=requests)
                    batch_id = batch.id
                    self._save_state(pending_batch={"id": batch_id, "slots": {
                        cid: [_slot_to_json(s) for s in slots] for cid, slots in mapping.items()}})
                    self.log(f"[{self.table.name}] submitted Message Batch {batch_id} "
                             f"({len(requests)} requests, {remaining} records)")
                pending = None
                while True:
                    b = await self.client.messages.batches.retrieve(batch_id)
                    if b.processing_status == "ended":
                        break
                    await asyncio.sleep(poll_interval)
                results = await self.client.messages.batches.results(batch_id)
                seen_ids = set()
                async for entry in results:
                    slots = mapping.get(entry.custom_id)
                    if slots is None:
                        continue
                    seen_ids.add(entry.custom_id)
                    self.stats.requests += 1
                    if entry.result.type == "succeeded":
                        self.process(slots, entry.result.message)
                    else:
                        self.stats.failed_requests += 1
                        for s in slots:
                            self.sampler.release_slot(s)
                    self._progress()
                for cid, slots in mapping.items():
                    if cid not in seen_ids:
                        for s in slots:
                            self.sampler.release_slot(s)
                self._save_state(pending_batch=None)
                idle_rounds = idle_rounds + 1 if len(self.records) == before else 0
            if len(self.records) < self.target:
                self.log(self._stall_message(f"stopped after {rounds} Message Batch round(s)"))
        finally:
            self.stats.finished = True
            self._progress()
            self._save_state()
        return self.records


def _slot_to_json(s: Slot) -> dict[str, Any]:
    d = dataclasses.asdict(s)
    d["fixed"] = {k: to_jsonable(v) for k, v in s.fixed.items()}
    return d


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def make_client(max_retries: int = 6) -> Any:
    """Real async client. The SDK retries 429/5xx with exponential backoff."""
    import anthropic
    return anthropic.AsyncAnthropic(max_retries=max_retries)


async def generate_dataset(
    schema: DatasetSchema,
    out_dir: str | None = None,
    client: Any = None,
    only: list[str] | None = None,
    counts: dict[str, int] | None = None,
    batch_api: bool = False,
    resume: bool = False,
    on_progress: ProgressFn | None = None,
    log: LogFn | None = None,
    poll_interval: float = 30.0,
    scenarios: dict[str, list[str]] | None = None,
) -> dict[str, TableGenerator]:
    """Generate every table, parents first. Returns the per-table generators."""
    if schema.settings.seed is None:  # always record a seed, so runs are reproducible
        schema.settings.seed = random.SystemRandom().randrange(1, 2**31)
    client = client or make_client()
    store = Store(out_dir)
    done: dict[str, TableGenerator] = {}
    for table in schema.ordered_tables():
        if only and table.name not in only:
            continue
        parents: dict[str, list[dict[str, Any]]] = {}
        for f in table.ref_fields.values():
            pt = f.ref_table or ""
            if pt in done:
                parents[pt] = done[pt].records
            else:
                rows = store.load(pt)
                if not rows:
                    raise GenerationError(f"table '{table.name}' references '{pt}'; generate '{pt}' first")
                parents[pt] = rows
        gen = TableGenerator(table, schema.settings, client, store, parents,
                             target=(counts or {}).get(table.name), on_progress=on_progress, log=log,
                             scenarios=(scenarios or {}).get(table.name))
        if batch_api:
            await gen.run_batch_api(resume=resume, poll_interval=poll_interval)
        else:
            await gen.run(resume=resume)
        done[table.name] = gen
    return done


def generate(schema: Any, count: int | None = None, out_dir: str | None = None,
             **kwargs: Any) -> dict[str, list[dict[str, Any]]]:
    """Synchronous convenience API.

    >>> from src.generator import generate
    >>> rows = generate("examples/job_postings.json", count=50)["job_postings"]
    """
    from pathlib import Path

    from .schema_builder import load_schema, schema_from_dict

    if isinstance(schema, DatasetSchema):
        ds = schema
    elif isinstance(schema, dict):
        ds = schema_from_dict(schema)
    else:
        ds = load_schema(Path(schema))
    counts = kwargs.pop("counts", None)
    if count is not None:
        if len(ds.tables) != 1:
            raise ValueError("count= applies to single-table schemas; use counts={table: n}")
        counts = {ds.tables[0].name: count}
    gens = asyncio.run(generate_dataset(ds, out_dir=out_dir, counts=counts, **kwargs))
    return {name: g.records for name, g in gens.items()}
