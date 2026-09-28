"""Code-side sampling: the statistical half of the generator.

LLMs are good at coherence and bad at calibrated distributions: ask for 1,000
ages and you get a lump around 34. So anything with a *target distribution*
is decided here, deterministically from the seed, and handed to Claude as a
fixed fact for each record "slot"; Claude then writes a record that is
coherent with those facts.

* ``distribution`` fields (numbers, dates) are sampled exactly.
* ``stratify`` enum fields are allocated by a self-correcting quota planner, so
  the final label balance matches the weights even when Claude drifts.
* ``null_rate`` decides missingness per slot.
* ``ref`` fields pick a parent row (optionally Zipf-skewed) and pass along
  that row's context so child records agree with their parent.
"""

from __future__ import annotations

import datetime as _dt
import math
import random
import uuid as _uuid
from dataclasses import dataclass, field
from typing import Any

from .schema_builder import FieldSpec, TableSpec, to_jsonable  # noqa: F401  (re-exported)


@dataclass
class Slot:
    """One record-to-be: everything decided before Claude writes it."""

    index: int                                                # 1-based within its request
    fixed: dict[str, Any] = field(default_factory=dict)       # must appear verbatim (sampled)
    targets: dict[str, Any] = field(default_factory=dict)     # stratified enum targets
    nulls: list[str] = field(default_factory=list)            # fields that must be null
    refs: dict[str, Any] = field(default_factory=dict)        # fk field -> parent key value
    parent_context: dict[str, dict[str, Any]] = field(default_factory=dict)
    scenario: str | None = None
    edge_case: bool = False


def apportion(total: int, weights: list[float]) -> list[int]:
    """Largest-remainder apportionment of ``total`` across ``weights``."""
    raw = [total * w for w in weights]
    counts = [math.floor(x) for x in raw]
    rest = total - sum(counts)
    order = sorted(range(len(raw)), key=lambda i: raw[i] - counts[i], reverse=True)
    for i in order[:rest]:
        counts[i] += 1
    return counts


class QuotaPlanner:
    """Tracks target vs. achieved counts for one stratified field."""

    def __init__(self, f: FieldSpec, total: int, rng: random.Random):
        assert f.values is not None and f.weights is not None
        self.field = f
        self.values = list(f.values)
        self.weights = list(f.weights)
        self.target = dict(zip(map(self.key, self.values), apportion(total, self.weights)))
        self.accepted = {self.key(v): 0 for v in self.values}
        self.reserved = {self.key(v): 0 for v in self.values}
        self.rng = rng

    @staticmethod
    def key(v: Any) -> str:
        return str(v).strip().lower()

    def deficit(self, v: Any) -> int:
        k = self.key(v)
        return self.target[k] - self.accepted[k] - self.reserved[k]

    def assign(self) -> Any:
        deficits = [(self.deficit(v), v) for v in self.values]
        open_ = [(d, v) for d, v in deficits if d > 0]
        if open_:
            # sample proportionally to remaining deficit: keeps batches mixed
            # rather than emitting one label at a time
            total = sum(d for d, _ in open_)
            r = self.rng.uniform(0, total)
            for d, v in open_:
                r -= d
                if r <= 0:
                    break
        else:  # quotas met (retries/overshoot): fall back to the weights
            v = self.rng.choices(self.values, weights=self.weights)[0]
        self.reserved[self.key(v)] += 1
        return v

    def release(self, v: Any) -> None:
        k = self.key(v)
        if k in self.reserved and self.reserved[k] > 0:
            self.reserved[k] -= 1

    def is_full(self, v: Any) -> bool:
        k = self.key(v)
        return k in self.target and self.accepted[k] >= self.target[k]

    def accept(self, v: Any) -> None:
        k = self.key(v)
        if k in self.accepted:
            self.accepted[k] += 1


class Sampler:
    """All randomness for a table flows through one seeded RNG."""

    def __init__(self, table: TableSpec, total: int, seed: int | None,
                 parents: dict[str, list[dict[str, Any]]] | None = None):
        self.table = table
        self.rng = random.Random(seed)  # slot planning only: same seed -> same sequence of sampled values
        # everything whose draw count depends on API timing (uuids, avoid-lists) uses a separate stream,
        # so it cannot shift the planning sequence
        self.aux_rng = random.Random(f"{seed}-aux" if seed is not None else None)
        self.quotas = {k: QuotaPlanner(f, total, self.rng) for k, f in table.stratified_fields.items()}
        self.parents = parents or {}
        self._ref_weights: dict[str, list[float]] = {}
        self.scenarios: list[str] = []
        self._scenario_queue: list[str] = []
        self.sequence_next: dict[str, int] = {k: f.start for k, f in table.field_map.items()
                                              if f.generator == "sequence"}

    # ---- numeric / temporal ----------------------------------------------------
    def sample_value(self, f: FieldSpec) -> Any:
        d = f.distribution.model_dump() if f.distribution is not None else {"type": "uniform"}
        dtype = d["type"]
        rng = self.rng
        if f.type in ("date", "datetime"):
            lo, hi = f.min, f.max
            if f.type == "date":
                span = (hi - lo).days
                return lo + _dt.timedelta(days=rng.randint(0, span))
            span_s = int((hi - lo).total_seconds())
            return (lo + _dt.timedelta(seconds=rng.randint(0, span_s))).replace(microsecond=0)

        for _ in range(1000):  # rejection sampling into [min, max]
            if dtype == "uniform":
                x = rng.uniform(f.min, f.max)
            elif dtype == "normal":
                x = rng.gauss(d["mean"], d["std"])
            elif dtype == "lognormal":
                x = rng.lognormvariate(math.log(d["median"]), d["sigma"])
            else:  # exponential
                x = rng.expovariate(1.0 / d["mean"])
            if f.type == "integer":
                x = int(round(x))
            elif f.decimals is not None:
                x = round(x, f.decimals)
            if (f.min is None or x >= f.min) and (f.max is None or x <= f.max):
                return x
        # pathological parameters: clamp
        x = min(max(x, f.min if f.min is not None else x), f.max if f.max is not None else x)
        return int(x) if f.type == "integer" else x

    # ---- references ---------------------------------------------------------------
    def _pick_parent(self, fname: str, f: FieldSpec) -> dict[str, Any]:
        rows = self.parents.get(f.ref_table or "", [])
        if not rows:
            raise RuntimeError(f"table {self.table.name!r} references {f.ref_table!r}, which has no rows")
        if f.ref_skew <= 0:
            return self.rng.choice(rows)
        if fname not in self._ref_weights:
            ranks = list(range(1, len(rows) + 1))
            self.rng.shuffle(ranks)  # which parents are "popular" is random
            self._ref_weights[fname] = [1.0 / (r ** f.ref_skew) for r in ranks]
        return self.rng.choices(rows, weights=self._ref_weights[fname])[0]

    # ---- scenarios -------------------------------------------------------------------
    def set_scenarios(self, scenarios: list[str]) -> None:
        self.scenarios = list(scenarios)
        self._scenario_queue = []

    def _next_scenario(self) -> str | None:
        if not self.scenarios:
            return None
        if not self._scenario_queue:  # shuffled round-robin: even coverage, no fixed order
            self._scenario_queue = list(self.scenarios)
            self.rng.shuffle(self._scenario_queue)
        return self._scenario_queue.pop()

    # ---- slots -----------------------------------------------------------------------
    def make_slot(self, index: int) -> Slot:
        t = self.table
        s = Slot(index=index)
        for k, f in t.field_map.items():
            if f.code_generated:
                continue
            if f.null_rate is not None and self.rng.random() < f.null_rate:
                s.nulls.append(k)
                continue
            if f.sampled:
                s.fixed[k] = self.sample_value(f)
            elif k in self.quotas:
                s.targets[k] = self.quotas[k].assign()
        for k, f in t.ref_fields.items():
            parent = self._pick_parent(k, f)
            s.refs[k] = parent[f.ref_field]
            ctx_fields = f.ref_context or []
            if ctx_fields:
                s.parent_context[k] = {c: parent.get(c) for c in ctx_fields}
        s.scenario = self._next_scenario()
        s.edge_case = t.diversity.edge_case_ratio > 0 and self.rng.random() < t.diversity.edge_case_ratio
        return s

    def release_slot(self, s: Slot) -> None:
        for k, v in s.targets.items():
            self.quotas[k].release(v)

    # ---- code generators --------------------------------------------------------------
    def generate_code_value(self, fname: str, f: FieldSpec) -> Any:
        if f.generator == "uuid":
            return str(_uuid.UUID(int=self.aux_rng.getrandbits(128), version=4))
        if f.generator == "sequence":
            n = self.sequence_next[fname]
            self.sequence_next[fname] = n + 1
            if f.type == "string":
                return f.format.format(n) if f.format else str(n)
            return n
        raise ValueError(f"no generator for {fname}")
