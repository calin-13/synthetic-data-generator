"""Export & storage: CSV, JSON, JSONL, Parquet and SQLite (via pandas), plus readers
that turn any of those back into records for validation."""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from .schema_builder import TableSpec

FORMATS = ("csv", "json", "jsonl", "parquet", "sqlite")
_EXT = {".csv": "csv", ".json": "json", ".jsonl": "jsonl", ".parquet": "parquet", ".sqlite": "sqlite",
        ".db": "sqlite", ".tsv": "csv"}


def format_from_path(path: str | Path) -> str:
    fmt = _EXT.get(Path(path).suffix.lower())
    if not fmt:
        raise ValueError(f"cannot infer format from '{path}'; use one of {', '.join(FORMATS)}")
    return fmt


def flatten(rec: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Nested objects become dotted columns; arrays become JSON strings (tabular formats)."""
    out: dict[str, Any] = {}
    for k, v in rec.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        elif isinstance(v, list):
            out[key] = json.dumps(v, ensure_ascii=False)
        else:
            out[key] = v
    return out


def to_dataframe(rows: list[dict[str, Any]], flat: bool = True) -> pd.DataFrame:
    if flat:
        return pd.DataFrame([flatten(r) for r in rows])
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_table(rows: list[dict[str, Any]], path: str | Path, fmt: str | None = None,
                table: TableSpec | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = fmt or format_from_path(path)
    if fmt == "jsonl":
        with path.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    elif fmt == "json":
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    elif fmt == "csv":
        to_dataframe(rows).to_csv(path, index=False, sep="\t" if path.suffix == ".tsv" else ",")
    elif fmt == "parquet":
        try:
            import pyarrow  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("Parquet export needs pyarrow: pip install pyarrow") from e
        # nested objects and arrays are preserved as Parquet structs/lists
        pd.DataFrame(rows).to_parquet(path, index=False)
    elif fmt == "sqlite":
        name = table.name if table else path.stem
        write_sqlite({name: rows}, {name: table} if table else {}, path)
    else:
        raise ValueError(f"unknown format '{fmt}'")
    return path


_SQL_TYPES = {"integer": "INTEGER", "number": "REAL", "boolean": "INTEGER"}


def write_sqlite(tables: dict[str, list[dict[str, Any]]], specs: dict[str, TableSpec], path: str | Path) -> Path:
    """One SQL table per dataset table, with primary and foreign keys where the schema declares them."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    con = sqlite3.connect(path)
    try:
        con.execute("PRAGMA foreign_keys = ON")
        order = list(tables)
        # parents first so FK constraints are satisfied on insert
        order.sort(key=lambda n: sum(1 for f in (specs[n].ref_fields.values() if specs.get(n) else [])))
        for name in order:
            df = to_dataframe(tables[name])
            spec = specs.get(name)
            if spec is not None:
                for f in spec.fields:
                    if f.name not in df.columns and f.type != "object":
                        df[f.name] = None
            pk = next((f.name for f in spec.fields if f.generator), None) if spec else None
            defs, fks = [], []
            for c in df.columns:
                f = spec.field_map.get(c) if spec else None
                d = f'"{c}" {_SQL_TYPES.get(f.type, "TEXT") if f else "TEXT"}'
                if c == pk:
                    d += " PRIMARY KEY"
                defs.append(d)
                if f is not None and f.ref:
                    fks.append(f'FOREIGN KEY ("{c}") REFERENCES "{f.ref_table}"("{f.ref_field}")')
            con.execute(f'CREATE TABLE "{name}" ({", ".join(defs + fks)})')
            df = df.astype(object).where(pd.notna(df), None)
            df.to_sql(name, con, if_exists="append", index=False)
        con.commit()
    finally:
        con.close()
    return path


def export_dataset(tables: dict[str, list[dict[str, Any]]], specs: dict[str, TableSpec], out_dir: str | Path,
                   formats: list[str]) -> list[Path]:
    out = Path(out_dir)
    written: list[Path] = []
    for fmt in formats:
        if fmt not in FORMATS:
            raise ValueError(f"unknown format '{fmt}'; choose from {', '.join(FORMATS)}")
        if fmt == "sqlite":
            written.append(write_sqlite(tables, specs, out / "dataset.sqlite"))
            continue
        for name, rows in tables.items():
            written.append(write_table(rows, out / f"{name}.{fmt}", fmt, specs.get(name)))
    return written


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------

def _clean(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if hasattr(v, "isoformat") and not isinstance(v, str):
        try:
            return v.isoformat()
        except Exception:  # noqa: BLE001
            return str(v)
    if hasattr(v, "tolist") and not isinstance(v, (str, bytes)):  # numpy scalars/arrays
        return _clean(v.tolist())
    if isinstance(v, list):
        return [_clean(x) for x in v]
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    return v


def unflatten(rec: dict[str, Any]) -> dict[str, Any]:
    """Dotted columns back into nested objects; JSON-array strings back into lists."""
    out: dict[str, Any] = {}
    for key, v in rec.items():
        if isinstance(v, str) and v.startswith("[") and v.endswith("]"):
            try:
                v = json.loads(v)
            except json.JSONDecodeError:
                pass
        parts = str(key).split(".")
        cur = out
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
            if not isinstance(cur, dict):
                break
        else:
            cur[parts[-1]] = v
    return out


def read_table(path: str | Path, limit: int | None = None, sqlite_table: str | None = None) -> list[dict[str, Any]]:
    """Read CSV/TSV, JSON, JSONL, Parquet or a SQLite table into records."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"data file not found: {path}")
    fmt = format_from_path(path)
    if fmt == "jsonl":
        rows = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
        return rows
    if fmt == "json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = next((v for v in data.values() if isinstance(v, list)), [data])
        return data[:limit] if limit else data
    if fmt == "csv":
        df = pd.read_csv(path, dtype=str, keep_default_na=False, nrows=limit,
                         sep="\t" if path.suffix == ".tsv" else ",")
        return [unflatten({k: (None if v == "" else v) for k, v in r.items()}) for r in df.to_dict("records")]
    if fmt == "parquet":
        df = pd.read_parquet(path)
        if limit:
            df = df.head(limit)
        return [_clean(r) for r in df.to_dict("records")]
    con = sqlite3.connect(path)
    try:
        names = [r[0] for r in con.execute("select name from sqlite_master where type='table'")]
        name = sqlite_table or (names[0] if len(names) == 1 else None)
        if name is None:
            raise ValueError(f"{path} has tables {names}; pass --table")
        df = pd.read_sql_query(f'select * from "{name}"' + (f" limit {int(limit)}" if limit else ""), con)
    finally:
        con.close()
    return [unflatten({k: _clean(v) for k, v in r.items()}) for r in df.to_dict("records")]
