"""Storage & checkpointing: accepted records are appended to ``<out>/<table>.jsonl`` as they
arrive, and per-table state (scenario plan, stats, in-flight Message Batch)
lives in ``<out>/.synthgen/<table>.json``. A crashed or interrupted run
resumes from exactly where it stopped."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class Store:
    def __init__(self, out_dir: str | Path | None):
        self.out_dir = Path(out_dir) if out_dir else None
        self._mem: dict[str, list[dict[str, Any]]] = {}
        self._state_mem: dict[str, dict[str, Any]] = {}
        self.torn: set[str] = set()  # tables whose data file had an incomplete final line
        if self.out_dir:
            (self.out_dir / ".synthgen").mkdir(parents=True, exist_ok=True)

    def data_path(self, table: str) -> Path | None:
        return self.out_dir / f"{table}.jsonl" if self.out_dir else None

    def _state_path(self, table: str) -> Path | None:
        return self.out_dir / ".synthgen" / f"{table}.json" if self.out_dir else None

    def load(self, table: str) -> list[dict[str, Any]]:
        p = self.data_path(table)
        if p is None:
            return list(self._mem.get(table, []))
        if not p.exists():
            return []
        rows = []
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    self.torn.add(table)  # torn final line from a crash: drop it and anything after
                    break
        return rows

    def reset(self, table: str) -> None:
        self._mem.pop(table, None)
        self._state_mem.pop(table, None)
        for p in (self.data_path(table), self._state_path(table)):
            if p and p.exists():
                p.unlink()

    def rewrite(self, table: str, rows: list[dict[str, Any]]) -> None:
        """Rewrite the data file (used on resume to drop a torn tail)."""
        p = self.data_path(table)
        if p is None:
            self._mem[table] = list(rows)
            return
        tmp = p.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, p)

    def append(self, table: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        p = self.data_path(table)
        if p is None:
            self._mem.setdefault(table, []).extend(rows)
            return
        with p.open("a", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()

    def load_state(self, table: str) -> dict[str, Any]:
        p = self._state_path(table)
        if p is None:
            return dict(self._state_mem.get(table, {}))
        if not p.exists():
            return {}
        return json.loads(p.read_text(encoding="utf-8"))

    def save_state(self, table: str, state: dict[str, Any]) -> None:
        p = self._state_path(table)
        if p is None:
            self._state_mem[table] = dict(state)
            return
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, p)


# ---------------------------------------------------------------------------
# "Last run" pointer, so `--export` / `--validate` can default to the latest output
# ---------------------------------------------------------------------------

LAST_RUN = Path(".synthgen") / "last_run.json"


def write_last_run(info: dict[str, Any], base: str | Path = ".") -> None:
    p = Path(base) / LAST_RUN
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")


def read_last_run(base: str | Path = ".") -> dict[str, Any] | None:
    p = Path(base) / LAST_RUN
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))
