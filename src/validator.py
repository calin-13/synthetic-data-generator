"""Data quality validation.

Two levels:

* **Record level** (used during generation and on any dataset): every record is
  validated with the Pydantic model compiled from the schema — types, ranges,
  lengths, regexes, date bounds, enums, nested objects, array sizes — plus the
  schema's cross-field ``checks``.
* **Dataset level** (:func:`quality_report`): a pandas-based report on what a
  schema can't express — duplicates and near-duplicates, label balance vs.
  target, numeric outliers, generation artefacts ("obvious patterns" such as
  values trending with row order, round-number heaping, templated text
  openings, low lexical diversity) — rolled into a 0-100 quality score.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import re
from collections import Counter
from typing import Any

import pandas as pd
from pydantic import ValidationError

from .schema_builder import FieldSpec, TableSpec, parse_temporal, record_model

_WORD = re.compile(r"[\w']+", re.UNICODE)


# ---------------------------------------------------------------------------
# Record level
# ---------------------------------------------------------------------------

class RecordError(ValueError):
    def __init__(self, path: str, reason: str):
        super().__init__(f"{path}: {reason}")
        self.path = path
        self.reason = reason

    @property
    def category(self) -> str:
        """Stable key for aggregating rejections, e.g. ``'salary: Input should be ...'``."""
        base = re.sub(r"\.\d+(?=\.|$)", "[]", self.path)
        return f"{base}: {self.reason}"


_SAFE_BUILTINS = {
    "len": len, "min": min, "max": max, "abs": abs, "sum": sum, "round": round, "any": any, "all": all,
    "sorted": sorted, "set": set, "str": str, "int": int, "float": float, "bool": bool,
    "isinstance": isinstance, "list": list, "dict": dict, "None": None, "True": True, "False": False,
}


class CheckRunner:
    """Evaluates the schema's cross-field ``checks`` (Python expressions over a record).

    Expressions come from the schema author (trusted, like code); records are data. Date and
    datetime fields are exposed as ``date``/``datetime`` objects so comparisons work. In child
    tables, ``parent['fk_field']`` holds the referenced row's ``ref_context``.
    """

    def __init__(self, table: TableSpec):
        self.table = table
        self.compiled = [(c, compile(c, "<check>", "eval")) for c in table.checks]
        self.needs_parent = {c for c, code in self.compiled if "parent" in code.co_names}
        self.env = {"__builtins__": _SAFE_BUILTINS, "date": _dt.date, "datetime": _dt.datetime,
                    "timedelta": _dt.timedelta, "re": re}

    def _namespace(self, rec: dict[str, Any]) -> dict[str, Any]:
        ns = dict(rec)
        for f in self.table.fields:
            if f.type in ("date", "datetime", "time") and isinstance(rec.get(f.name), str):
                try:
                    ns[f.name] = parse_temporal(f.type, rec[f.name])
                except ValueError:
                    pass
        ns["record"] = rec
        return ns

    def failures(self, rec: dict[str, Any], parent: dict[str, dict[str, Any]] | None = None) -> list[str]:
        out = []
        if not self.compiled:
            return out
        ns = self._namespace(rec)
        if parent is not None:
            ns["parent"] = parent
        for src, code in self.compiled:
            if parent is None and src in self.needs_parent:
                continue  # needs generation-time parent context
            try:
                ok = eval(code, self.env, ns)  # noqa: S307 - author-supplied expression
            except Exception as e:  # a check that errors counts as failed
                out.append(f"{src} (raised {type(e).__name__})")
                continue
            if not ok:
                out.append(src)
        return out

    def run(self, rec: dict[str, Any], parent: dict[str, dict[str, Any]] | None = None) -> None:
        failed = self.failures(rec, parent)
        if failed:
            raise RecordError("check", f"failed: {failed[0]}")


def record_errors(table: TableSpec, rec: Any) -> tuple[dict[str, Any] | None, list[RecordError]]:
    """Validate against the Pydantic record model. Returns (normalized record | None, errors)."""
    if not isinstance(rec, dict):
        return None, [RecordError("record", "not an object")]
    try:
        m = record_model(table).model_validate(rec)
    except ValidationError as e:
        errs = []
        for err in e.errors():
            loc = ".".join(str(x) for x in err["loc"]) or "record"
            msg = err["msg"].removeprefix("Value error, ")
            errs.append(RecordError(loc, msg))
        return None, errs
    return m.model_dump(by_alias=True, mode="json"), []


def validate_record(table: TableSpec, rec: Any, checks: CheckRunner | None = None,
                    parent: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """Validate one complete record; returns the normalized record or raises RecordError."""
    out, errs = record_errors(table, rec)
    if errs:
        raise errs[0]
    assert out is not None
    (checks or CheckRunner(table)).run(out, parent)
    return out


# ---------------------------------------------------------------------------
# Dataset level
# ---------------------------------------------------------------------------

def _flatten(rec: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in rec.items():
        if isinstance(v, dict):
            out.update(_flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def _spec_for(table: TableSpec | None, col: str) -> FieldSpec | None:
    if table is None:
        return None
    parts = col.split(".")
    f = table.field_map.get(parts[0])
    for p in parts[1:]:
        if f is None or f.type != "object":
            return None
        f = f.prop_map.get(p)
    return f


def _text_stats(values: list[str]) -> dict[str, Any]:
    words = [_WORD.findall(v.lower()) for v in values]
    bigrams = [tuple(w[i:i + 2]) for w in words for i in range(len(w) - 1)]
    openings = Counter(" ".join(w[:3]) for w in words if len(w) >= 3)
    sentences = Counter(s.strip().lower() for v in values for s in re.split(r"(?<=[.!?])\s+", v)
                        if len(s.split()) >= 5)
    repeated = sum(c for c in sentences.values() if c > 1)
    total_sent = sum(sentences.values()) or 1
    norm = Counter(re.sub(r"\W+", " ", v.lower()).strip() for v in values)
    return {
        "distinct_2": round(len(set(bigrams)) / len(bigrams), 3) if bigrams else None,
        "top_openings": [{"text": t, "share": round(c / len(values), 3)} for t, c in openings.most_common(3)],
        "repeated_sentence_share": round(repeated / total_sent, 3),
        "near_duplicates": sum(c - 1 for c in norm.values() if c > 1),
    }


def _numeric_stats(s: pd.Series) -> dict[str, Any]:
    q = s.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    iqr = q[0.75] - q[0.25]
    lo, hi = q[0.25] - 3 * iqr, q[0.75] + 3 * iqr
    outliers = s[(s < lo) | (s > hi)] if iqr > 0 else s.iloc[0:0]
    info: dict[str, Any] = {
        "min": float(s.min()), "p5": float(q[0.05]), "p50": float(q[0.5]), "p95": float(q[0.95]),
        "max": float(s.max()), "mean": round(float(s.mean()), 4), "std": round(float(s.std(ddof=0)), 4),
        "outliers": int(len(outliers)),
        "outlier_examples": [float(x) for x in outliers.head(5)],
        "top_value_share": round(float(s.value_counts(normalize=True).iloc[0]), 3),
    }
    if len(s) >= 20 and s.nunique() > 5:
        order = pd.Series(range(len(s)), index=s.index, dtype=float)
        rho = s.rank().corr(order.rank())  # Spearman = Pearson on ranks (no scipy needed)
        info["row_order_correlation"] = None if pd.isna(rho) else round(float(rho), 3)
        magnitude = s.abs().median()
        if magnitude >= 100:
            step = 10 ** max(1, int(math.log10(magnitude)) - 1)
            info["round_number_share"] = round(float((s % step == 0).mean()), 3)
            info["round_number_step"] = step
    return info


def quality_report(rows: list[dict[str, Any]], table: TableSpec | None = None) -> dict[str, Any]:  # noqa: C901
    n = len(rows)
    report: dict[str, Any] = {"rows": n, "table": table.name if table else None, "columns": {}, "issues": []}
    issues: list[dict[str, str]] = report["issues"]

    def issue(sev: str, col: str, msg: str) -> None:
        issues.append({"severity": sev, "column": col, "message": msg})

    if n == 0:
        issue("high", "*", "dataset is empty")
        report["score"], report["grade"] = 0, "F"
        return report

    # ---- schema compliance --------------------------------------------------------
    if table is not None:
        checks = CheckRunner(table)
        violations: Counter = Counter()
        examples = []
        valid = 0
        normalized = []
        for i, r in enumerate(rows):
            out, errs = record_errors(table, r)
            normalized.append(out if out is not None else r)  # typed values (e.g. from CSV strings)
            if out is not None:
                errs = [RecordError("check", f"failed: {c}") for c in checks.failures(out)]
            if errs:
                for e in errs:
                    violations[e.category] += 1
                if len(examples) < 5:
                    examples.append({"row": i, "error": str(errs[0])})
            else:
                valid += 1
        report["schema"] = {"valid_rows": valid, "pass_rate": round(valid / n, 4),
                            "violations": dict(violations.most_common(15)), "examples": examples}
        if valid < n:
            issue("high", "*", f"{n - valid} rows ({(n - valid) / n:.1%}) violate the schema")
        rows = normalized

    # ---- duplicates ---------------------------------------------------------------------
    keys = Counter(json.dumps(r, sort_keys=True, default=str) for r in rows)
    dup_rows = sum(c - 1 for c in keys.values() if c > 1)
    report["duplicate_rows"] = dup_rows
    if dup_rows:
        issue("high", "*", f"{dup_rows} exact duplicate rows")

    df = pd.DataFrame([_flatten(r) for r in rows])
    for col in df.columns:
        spec = _spec_for(table, col)
        s = df[col]
        present = s[s.map(lambda v: v is not None and not (isinstance(v, float) and math.isnan(v)) and v != "")]
        info: dict[str, Any] = {"null_rate": round(1 - len(present) / n, 3)}
        report["columns"][col] = info
        if spec is not None and spec.null_rate is not None and n >= 30 \
                and abs(info["null_rate"] - spec.null_rate) > 0.1:
            issue("low", col, f"null rate {info['null_rate']:.0%} vs target {spec.null_rate:.0%}")
        if present.empty:
            if spec is None or not spec.nullable:
                issue("medium", col, "column is entirely empty")
            continue
        vals = present.tolist()
        hashable = [json.dumps(v, sort_keys=True, default=str) if isinstance(v, (list, dict)) else v for v in vals]
        info["distinct"] = len(set(hashable))
        if info["distinct"] == 1 and n >= 10 and not (spec and spec.type == "boolean"):
            issue("low", col, f"constant column (always {vals[0]!r})")
        if spec is not None and spec.unique and info["distinct"] < len(vals):
            issue("high", col, f"{len(vals) - info['distinct']} duplicate values in a unique field")

        is_enum = spec is not None and spec.type == "enum"
        is_bool = all(isinstance(v, bool) for v in vals)
        if is_enum or is_bool or (spec is None and all(isinstance(v, str) for v in vals)
                                   and info["distinct"] <= 20 and info["distinct"] / len(vals) < 0.5):
            dist = pd.Series([str(v) for v in vals]).value_counts(normalize=True)
            info["distribution"] = {k: round(float(v), 3) for k, v in dist.items()}
            probs = dist.to_numpy()
            k = len(spec.values) if is_enum and spec.values else len(probs)
            info["entropy"] = round(float(-(probs * [math.log(p) for p in probs]).sum() / math.log(k)), 3) \
                if k > 1 else 1.0
            if is_enum and spec.weights:
                target = {str(v): w for v, w in zip(spec.values or [], spec.weights)}
                tvd = 0.5 * sum(abs(info["distribution"].get(key, 0) - w) for key, w in target.items())
                info["target"] = {k2: round(w, 3) for k2, w in target.items()}
                info["tvd_from_target"] = round(tvd, 3)
                if tvd > 0.1:
                    issue("high", col, f"label distribution is {tvd:.0%} (TVD) away from target")
                elif tvd > 0.05:
                    issue("medium", col, f"label distribution is {tvd:.0%} (TVD) away from target")
            elif is_enum and len(vals) >= 50:
                unused = set(map(str, spec.values or [])) - set(info["distribution"])
                if unused:
                    issue("low", col, f"allowed values never used: {sorted(unused)}")
            continue

        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
            num = _numeric_stats(pd.Series(vals, dtype=float))
            info.update(num)
            if num["outliers"] / len(vals) > 0.05:
                issue("medium", col, f"{num['outliers']} outliers beyond 3x IQR ({num['outliers'] / len(vals):.0%})")
            elif num["outliers"]:
                issue("low", col, f"{num['outliers']} outliers beyond 3x IQR, e.g. {num['outlier_examples'][:3]}")
            rho = num.get("row_order_correlation")
            if rho is not None and abs(rho) > 0.8:
                issue("medium", col, f"values trend with row order (Spearman {rho:+.2f}) - generation artefact")
            if num.get("round_number_share", 0) > 0.6 and not (spec and spec.sampled):
                issue("low", col, f"{num['round_number_share']:.0%} of values are multiples of "
                                  f"{num['round_number_step']} (round-number heaping)")
            if num["top_value_share"] > 0.5 and info["distinct"] > 2:
                issue("medium", col, f"one value accounts for {num['top_value_share']:.0%} of rows")
            continue

        if all(isinstance(v, list) for v in vals):
            lens = pd.Series([len(v) for v in vals])
            elems = Counter(json.dumps(e, sort_keys=True, default=str) for v in vals for e in v)
            info["items"] = {"min": int(lens.min()), "mean": round(float(lens.mean()), 2), "max": int(lens.max()),
                             "distinct_elements": len(elems),
                             "top_elements": [json.loads(e) for e, _ in elems.most_common(5)]}
            continue

        if all(isinstance(v, str) for v in vals):
            lens = pd.Series([len(v) for v in vals])
            info["length"] = {"min": int(lens.min()), "mean": round(float(lens.mean()), 1), "max": int(lens.max())}
            if spec is not None and spec.type in ("date", "datetime"):
                continue
            words = pd.Series([len(v.split()) for v in vals])
            if (spec is not None and spec.type == "text") or words.mean() >= 5:
                t = _text_stats(vals)
                info["text"] = t
                if n >= 20 and t["top_openings"] and t["top_openings"][0]["share"] > 0.15:
                    o = t["top_openings"][0]
                    issue("medium", col, f"{o['share']:.0%} of values open with \"{o['text']}\" (templated text)")
                if n >= 20 and t["distinct_2"] is not None and t["distinct_2"] < 0.3:
                    issue("medium", col, f"low lexical diversity (distinct-2 = {t['distinct_2']})")
                if t["repeated_sentence_share"] > 0.1:
                    issue("medium", col, f"{t['repeated_sentence_share']:.0%} of sentences are repeated verbatim "
                                         "across rows")
                if t["near_duplicates"]:
                    issue("medium" if t["near_duplicates"] / n > 0.02 else "low", col,
                          f"{t['near_duplicates']} near-duplicate values")
            elif info["distinct"] / len(vals) < 0.5 and len(vals) >= 20 and spec is not None \
                    and spec.type in ("string", "email") and info["distinct"] > 20:
                issue("low", col, f"only {info['distinct']} distinct values in {len(vals)} rows")

    # ---- score ------------------------------------------------------------------------
    penalty = {"high": 12, "medium": 5, "low": 1}
    score = 100.0 - sum(penalty[i["severity"]] for i in issues)
    if "schema" in report:
        score -= (1 - report["schema"]["pass_rate"]) * 40
    score -= min(20, dup_rows / n * 100)
    score = max(0, round(score))
    report["score"] = score
    report["grade"] = "A" if score >= 90 else "B" if score >= 80 else "C" if score >= 65 else "D" if score >= 50 else "F"
    issues.sort(key=lambda i: ["high", "medium", "low"].index(i["severity"]))
    return report


def format_report(report: dict[str, Any], title: str = "") -> str:
    lines = [f"== Quality report: {title or report.get('table') or 'dataset'} ({report['rows']} rows) ==",
             f"score {report['score']}/100  grade {report['grade']}"]
    sch = report.get("schema")
    if sch:
        lines.append(f"schema pass rate {sch['pass_rate']:.1%} ({sch['valid_rows']}/{report['rows']})")
        for k, v in list(sch["violations"].items())[:8]:
            lines.append(f"  x {v:>5}  {k}")
    lines.append(f"duplicate rows {report.get('duplicate_rows', 0)}")
    lines.append("columns:")
    for c, info in report["columns"].items():
        parts = [f"null {info['null_rate']:.0%}"]
        if "distinct" in info:
            parts.append(f"distinct {info['distinct']}")
        if "distribution" in info:
            tgt = info.get("target", {})
            items = [f"{k} {v:.0%}" + (f" (target {tgt[k]:.0%})" if k in tgt else "")
                     for k, v in list(info["distribution"].items())[:6]]
            parts.append(", ".join(items))
        if "mean" in info:
            parts.append(f"{info['min']:g}..{info['max']:g} mean {info['mean']:g} p50 {info['p50']:g}")
        if "items" in info:
            it = info["items"]
            parts.append(f"items {it['min']}-{it['max']} (mean {it['mean']}), {it['distinct_elements']} distinct")
        if "length" in info:
            L = info["length"]
            parts.append(f"len {L['min']}-{L['max']} (mean {L['mean']})")
        if "text" in info:
            parts.append(f"distinct-2 {info['text']['distinct_2']}")
        lines.append(f"  {c}: " + " | ".join(parts))
    if report["issues"]:
        lines.append("issues:")
        for i in report["issues"]:
            lines.append(f"  [{i['severity']:<6}] {i['column']}: {i['message']}")
    else:
        lines.append("no issues found")
    return "\n".join(lines)
