"""Schema definition helpers (the ``--define`` step).

Three ways to get from "I need X" to a validated schema:

* :func:`interactive_define` - a guided terminal wizard (Click prompts).
* :func:`schema_from_description` - Claude designs the schema from a plain-English
  description, using structured outputs; the result is validated with the real
  schema models and Claude repairs it if needed.
* :func:`schema_from_sample` - profiles an existing dataset **locally** and writes a
  schema reproducing its shape (types, category balance, numeric distributions,
  null rates, date ranges) without copying free-text or identifying values.
  Real rows never leave the machine.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import re
import statistics
import uuid as _uuid
from collections import Counter
from typing import Any


from .schema_builder import FIELD_TYPES, SchemaError, output_config, schema_from_dict

# ---------------------------------------------------------------------------
# From a description (Claude-designed)
# ---------------------------------------------------------------------------

_SCALAR = ["string", "text", "integer", "number", "boolean", "date", "datetime", "email", "url", "uuid", "enum"]

_FIELD_DEF = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "snake_case field name"},
        "type": {"type": "string", "enum": _SCALAR + ["array"]},
        "description": {"type": "string",
                        "description": "What the field holds and how it relates to other fields; guides generation."},
        "nullable": {"type": "boolean"},
        "null_rate": {"type": "number", "description": "Share of records where the value is null; 0 if not nullable."},
        "unique": {"type": "boolean"},
        "values": {"type": "array", "items": {"type": "string"},
                   "description": "Allowed values when type is enum (or items_type is enum); else empty."},
        "weights": {"type": "array", "items": {"type": "number"},
                    "description": "Realistic relative frequency per enum value, same order; empty for uniform."},
        "min": {"type": "string", "description": "Lower bound for numbers/dates as a literal, or empty."},
        "max": {"type": "string", "description": "Upper bound for numbers/dates as a literal, or empty."},
        "max_length": {"type": "integer", "description": "Max characters for strings; 0 for none."},
        "items_type": {"type": "string", "enum": ["", "string", "integer", "number", "enum"],
                       "description": "Element type when type is array; else empty."},
        "min_items": {"type": "integer"},
        "max_items": {"type": "integer"},
        "generator": {"type": "string", "enum": ["", "uuid", "sequence"],
                      "description": "Use for identifiers that code should assign; else empty."},
        "sequence_format": {"type": "string",
                            "description": "Python format for string sequences, e.g. 'ORD-{:06d}'; else empty."},
        "distribution": {"type": "string", "enum": ["", "uniform", "normal", "lognormal", "exponential"],
                         "description": "Sample numeric/date values from this distribution in code (gives "
                                        "realistic spreads). Dates support only uniform. Empty to let the model "
                                        "choose values."},
        "dist_a": {"type": "number", "description": "normal: mean; lognormal: median; exponential: mean; else 0"},
        "dist_b": {"type": "number", "description": "normal: std; lognormal: sigma (e.g. 0.5-1.2); else 0"},
    },
    "required": ["name", "type", "description", "nullable", "null_rate", "unique", "values", "weights", "min",
                 "max", "max_length", "items_type", "min_items", "max_items", "generator", "sequence_format",
                 "distribution", "dist_a", "dist_b"],
    "additionalProperties": False,
}

SPEC_DESIGN_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "snake_case dataset name"},
        "description": {"type": "string"},
        "context": {"type": "string",
                    "description": "Domain knowledge that makes records realistic: real-world norms, typical "
                                   "ranges, jargon, correlations between fields, common edge cases."},
        "rules": {"type": "array", "items": {"type": "string"},
                  "description": "Cross-field consistency rules in plain English."},
        "scenario_guidance": {"type": "string",
                              "description": "What kinds of situations the dataset should cover."},
        "edge_case_ratio": {"type": "number"},
        "fields": {"type": "array", "items": _FIELD_DEF},
    },
    "required": ["name", "description", "context", "rules", "scenario_guidance", "edge_case_ratio", "fields"],
    "additionalProperties": False,
}

_DESIGN_PROMPT = """\
Design a synthetic dataset specification for this request:

<request>
{request}
</request>

Think like a domain expert and a data engineer:
- Choose fields a real dataset of this kind would have, with precise descriptions.
- Use enum + realistic weights for categorical fields (label columns for ML datasets usually deserve \
deliberate balance - say so in the weights).
- Use distributions for numeric amounts and dates so values follow realistic spreads; leave the \
distribution empty for values that must be derived from other fields.
- Use generator uuid/sequence for identifiers.
- Write the domain context and cross-field rules that keep records coherent.
- Keep it flat (arrays of scalars at most) and prefer at most ~20 fields; at most 12 nullable fields.
Target size: {count} records."""


def _num(s: str) -> float | int | None:
    s = str(s).strip()
    if not s:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return int(v) if v.is_integer() and "." not in s else v


def design_to_schema(d: dict[str, Any], count: int) -> dict[str, Any]:
    """Convert Claude's flat design into a spec dict."""
    fields: list[dict[str, Any]] = []
    for fd in d.get("fields", []):
        name = re.sub(r"\W+", "_", fd["name"].strip()).strip("_").lower() or "field"
        t = fd["type"]
        f: dict[str, Any] = {"name": name, "type": t}
        if fd.get("description"):
            f["description"] = fd["description"]
        if fd.get("null_rate", 0) and 0 < fd["null_rate"] < 1:
            f["null_rate"] = round(fd["null_rate"], 3)
        elif fd.get("nullable"):
            f["nullable"] = True
        if fd.get("unique") and not fd.get("generator"):
            f["unique"] = True
        if t == "enum":
            f["values"] = fd.get("values") or ["unknown"]
            w = fd.get("weights") or []
            if len(w) == len(f["values"]) and sum(w) > 0 and all(x >= 0 for x in w):
                f["weights"] = [round(x / sum(w), 4) for x in w]
        if t in ("integer", "number"):
            lo, hi = _num(fd.get("min", "")), _num(fd.get("max", ""))
            if lo is not None:
                f["min"] = lo
            if hi is not None:
                f["max"] = hi
        if t in ("date", "datetime"):
            if fd.get("min"):
                f["min"] = fd["min"]
            if fd.get("max"):
                f["max"] = fd["max"]
        if t in ("string", "text") and fd.get("max_length", 0) > 0:
            f["max_length"] = fd["max_length"]
        if t == "array":
            it = fd.get("items_type") or "string"
            f["items"] = {"type": "enum", "values": fd.get("values") or ["unknown"]} if it == "enum" else {"type": it}
            if fd.get("min_items", 0) > 0:
                f["min_items"] = fd["min_items"]
            if fd.get("max_items", 0) > 0:
                f["max_items"] = fd["max_items"]
        gen = fd.get("generator")
        if gen:
            f.pop("unique", None)
            f["generator"] = gen
            if gen == "uuid":
                f["type"] = "uuid"
            elif gen == "sequence":
                fmt = fd.get("sequence_format", "")
                f["type"] = "string" if fmt else "integer"
                if fmt:
                    f["format"] = fmt
            for k in ("min", "max", "null_rate", "nullable", "max_length"):
                f.pop(k, None)
        dist = fd.get("distribution")
        if dist and not gen and t in ("integer", "number", "date", "datetime"):
            if t in ("date", "datetime"):
                if "min" in f and "max" in f:
                    f["distribution"] = {"type": "uniform"}
            elif dist == "uniform" and "min" in f and "max" in f:
                f["distribution"] = {"type": "uniform"}
            elif dist == "normal" and fd.get("dist_b", 0) > 0:
                f["distribution"] = {"type": "normal", "mean": fd["dist_a"], "std": fd["dist_b"]}
            elif dist == "lognormal" and fd.get("dist_a", 0) > 0 and fd.get("dist_b", 0) > 0:
                f["distribution"] = {"type": "lognormal", "median": fd["dist_a"], "sigma": fd["dist_b"]}
            elif dist == "exponential" and fd.get("dist_a", 0) > 0:
                f["distribution"] = {"type": "exponential", "mean": fd["dist_a"]}
        fields.append(f)

    spec: dict[str, Any] = {
        "name": re.sub(r"\W+", "_", d.get("name", "dataset")).strip("_").lower() or "dataset",
        "description": d.get("description", ""),
        "context": d.get("context", ""),
        "count": count,
        "batch_size": 10,
    }
    if d.get("rules"):
        spec["rules"] = d["rules"]
    div: dict[str, Any] = {"scenarios": max(10, min(60, count // 10))}
    if d.get("scenario_guidance"):
        div["scenario_guidance"] = d["scenario_guidance"]
    if 0 < d.get("edge_case_ratio", 0) <= 0.3:
        div["edge_case_ratio"] = round(d["edge_case_ratio"], 3)
    spec["diversity"] = div
    spec["fields"] = fields
    return spec


async def schema_from_description(client: Any, request: str, count: int = 200,
                                  model: str = "claude-opus-5-5", max_repairs: int = 2) -> dict[str, Any]:
    """Have Claude design a schema; validate it and let Claude fix any errors."""
    messages: list[dict[str, Any]] = [{"role": "user",
                                       "content": _DESIGN_PROMPT.format(request=request, count=count)}]
    last_err = ""
    for _ in range(max_repairs + 1):
        msg = await client.messages.create(model=model, max_tokens=16000, messages=messages,
                                           output_config=output_config(SPEC_DESIGN_SCHEMA))
        if msg.stop_reason == "refusal":
            raise SchemaError("Claude declined to design this dataset")
        text = "".join(getattr(b, "text", "") for b in msg.content if getattr(b, "type", "") == "text")
        try:
            design = json.loads(text)
        except json.JSONDecodeError as e:
            raise SchemaError(f"could not parse the designed spec (stop_reason={msg.stop_reason})") from e
        spec = design_to_schema(design, count)
        try:
            schema_from_dict(spec)
            return spec
        except SchemaError as e:
            last_err = str(e)
            messages += [{"role": "assistant", "content": text},
                         {"role": "user", "content": f"That design failed validation: {e}. Return a corrected "
                                                     "design addressing the error."}]
    raise SchemaError(f"designed spec failed validation: {last_err}")


# ---------------------------------------------------------------------------
# From a sample (local, privacy-preserving)
# ---------------------------------------------------------------------------

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")
_URL = re.compile(r"^https?://\S+$", re.I)
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DATETIME = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?$")


def _parse_scalar(v: Any) -> Any:
    """CSV cells arrive as strings: recover bools and numbers."""
    if not isinstance(v, str):
        return v
    s = v.strip()
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    if re.fullmatch(r"[+-]?\d+", s) and not (len(s) > 1 and s.lstrip("+-").startswith("0")):
        return int(s)
    if re.fullmatch(r"[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?", s):
        return float(s)
    if s.startswith(("{", "[")):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
    return s


def _is_uuid(s: str) -> bool:
    try:
        _uuid.UUID(s)
        return len(s) >= 32
    except ValueError:
        return False


def _round_sig(x: float, sig: int = 4) -> float:
    if x == 0 or not math.isfinite(x):
        return x
    return round(x, sig - int(math.floor(math.log10(abs(x)))) - 1)


def infer_field(values: list[Any], top_level: bool = True, max_enum: int = 20) -> dict[str, Any]:
    n_total = len(values)
    vals = [v for v in values if v is not None and v != ""]
    f: dict[str, Any] = {}
    null_rate = 1 - len(vals) / n_total if n_total else 0
    if null_rate > 0:
        if top_level:
            f["null_rate"] = round(null_rate, 3)
        else:
            f["nullable"] = True
    if not vals:
        return {"type": "string", "nullable": True}
    n = len(vals)

    if all(isinstance(v, dict) for v in vals):
        keys: list[str] = []
        for v in vals:
            keys += [k for k in v if k not in keys]
        return {**f, "type": "object",
                "properties": [{"name": k, **infer_field([v.get(k) for v in vals], top_level=False)} for k in keys]}
    if all(isinstance(v, list) for v in vals):
        elems = [e for v in vals for e in v]
        lens = [len(v) for v in vals]
        items = infer_field(elems, top_level=False) if elems else {"type": "string"}
        items.pop("nullable", None)
        return {**f, "type": "array", "items": items, "min_items": min(lens), "max_items": max(lens)}
    if all(isinstance(v, bool) for v in vals):
        return {**f, "type": "boolean"}

    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
        is_int = all(isinstance(v, int) or float(v).is_integer() for v in vals)
        xs = [float(v) for v in vals]
        distinct = len(set(xs))
        if top_level and is_int and distinct == n and n >= 10 and sorted(xs) == xs and \
                all(b - a == 1 for a, b in zip(xs, xs[1:])):
            return {"type": "integer", "generator": "sequence", "start": int(xs[0])}
        if is_int and distinct <= min(max_enum // 2, 8) and n >= 20 and distinct / n < 0.2:
            c = Counter(int(x) for x in xs)
            vs = sorted(c)
            return {**f, "type": "enum", "values": vs, "weights": [round(c[v] / n, 3) for v in vs]}
        f.update({"type": "integer" if is_int else "number",
                  "min": int(min(xs)) if is_int else _round_sig(min(xs)),
                  "max": int(max(xs)) if is_int else _round_sig(max(xs))})
        if not is_int:
            decimals = max((len(str(v).split(".")[1]) if "." in str(v) else 0) for v in vals[:200])
            if decimals <= 4:
                f["decimals"] = decimals
        if top_level and n >= 10 and distinct > 5:
            mean, med = statistics.fmean(xs), statistics.median(xs)
            std = statistics.pstdev(xs)
            skew = (statistics.fmean((x - mean) ** 3 for x in xs) / std ** 3) if std > 0 else 0.0
            if min(xs) > 0 and med > 0 and skew > 1.0:
                logs = [math.log(x) for x in xs]
                f["distribution"] = {"type": "lognormal", "median": _round_sig(med),
                                     "sigma": round(statistics.pstdev(logs), 3)}
            elif std > 0:
                f["distribution"] = {"type": "normal", "mean": _round_sig(mean), "std": _round_sig(std)}
        return f

    if all(isinstance(v, str) for v in vals):
        strs = [v.strip() for v in vals]
        if all(_DATE.match(s) for s in strs):
            try:
                ds = sorted(_dt.date.fromisoformat(s) for s in strs)
                f.update({"type": "date", "min": ds[0].isoformat(), "max": ds[-1].isoformat()})
                if top_level and ds[0] < ds[-1]:
                    f["distribution"] = "uniform"
                return f
            except ValueError:
                pass
        if all(_DATETIME.match(s) for s in strs):
            try:
                dts = sorted(_dt.datetime.fromisoformat(s.replace("Z", "+00:00").replace(" ", "T")) for s in strs)
                fmt = lambda d: (d if d.tzinfo else d.replace(tzinfo=_dt.timezone.utc)).astimezone(  # noqa: E731
                    _dt.timezone.utc).isoformat().replace("+00:00", "Z")
                f.update({"type": "datetime", "min": fmt(dts[0]), "max": fmt(dts[-1])})
                if top_level and dts[0] < dts[-1]:
                    f["distribution"] = "uniform"
                return f
            except ValueError:
                pass
        distinct = len(set(s.lower() for s in strs))
        if all(_is_uuid(s) for s in strs):
            return {"type": "uuid", "generator": "uuid"} if top_level and distinct == n else {**f, "type": "uuid"}
        if all(_EMAIL.match(s) for s in strs):
            return {**f, "type": "email", **({"unique": True} if distinct == n and n >= 10 else {})}
        if all(_URL.match(s) for s in strs):
            return {**f, "type": "url"}
        mean_len = statistics.fmean(len(s) for s in strs)
        if distinct <= max_enum and n >= 10 and distinct / n <= 0.5 and mean_len < 60:
            c = Counter(strs)
            vs = [v for v, _ in c.most_common()]
            return {**f, "type": "enum", "values": vs, "weights": [round(c[v] / n, 3) for v in vs]}
        f["type"] = "text" if mean_len > 80 else "string"
        f["max_length"] = int(max(len(s) for s in strs) * 1.2) + 1
        if distinct == n and n >= 10 and f["type"] == "string":
            f["unique"] = True
        return f

    return {**f, "type": "string"}


def schema_from_sample(rows: list[dict[str, Any]], name: str = "dataset", about: str = "",
                     count: int | None = None, max_enum: int = 20) -> dict[str, Any]:
    """Infer a schema from sample rows. Only aggregate statistics and category labels are kept."""
    if not rows:
        raise SchemaError("sample is empty")
    rows = [{k: _parse_scalar(v) for k, v in r.items()} for r in rows]
    cols: list[str] = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    fields = []
    for c in cols:
        key = re.sub(r"\W+", "_", c).strip("_") or "field"
        if key[0].isdigit():
            key = f"f_{key}"
        fd = infer_field([r.get(c) for r in rows], top_level=True, max_enum=max_enum)
        fd.setdefault("description", f"TODO: describe '{c}'" if c == key else f"TODO: describe '{c}' (source column {c!r})")
        fields.append({"name": key, **fd})
    spec: dict[str, Any] = {
        "name": re.sub(r"\W+", "_", name).strip("_").lower() or "dataset",
        "description": about or "TODO: describe what this dataset represents and who/what each record is.",
        "context": "TODO: add domain knowledge that makes records realistic (norms, jargon, correlations).",
        "count": count or len(rows),
        "batch_size": 10,
        "diversity": {"scenarios": 20},
        "fields": fields,
    }
    schema_from_dict(spec)  # guarantee it loads
    return spec


# ---------------------------------------------------------------------------
# Interactive wizard
# ---------------------------------------------------------------------------

_WIZARD_TYPES = ["string", "text", "integer", "number", "boolean", "enum", "date", "datetime", "email", "url",
                 "uuid", "array", "object"]


def _csv(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def _num_or_none(text: str) -> float | int | None:
    return _num(text) if text.strip() else None


def _prompt_field(click: Any, indent: str = "", top_level: bool = True) -> dict[str, Any] | None:
    name = click.prompt(f"{indent}Field name (empty to finish)", default="", show_default=False).strip()
    if not name:
        return None
    name = re.sub(r"\W+", "_", name).strip("_")
    t = click.prompt(f"{indent}  type", type=click.Choice(_WIZARD_TYPES), default="string")
    f: dict[str, Any] = {"name": name, "type": t}
    desc = click.prompt(f"{indent}  description (what it holds, how it relates to other fields)",
                        default="", show_default=False)
    if desc:
        f["description"] = desc
    if t == "enum":
        while True:
            vals = _csv(click.prompt(f"{indent}  allowed values (comma-separated)"))
            if vals:
                break
        f["values"] = vals
        w = click.prompt(f"{indent}  target weights, same order (optional, e.g. 0.5,0.3,0.2)",
                         default="", show_default=False)
        if w and top_level:
            try:
                ws = [float(x) for x in _csv(w)]
                if len(ws) == len(vals):
                    f["weights"] = ws
            except ValueError:
                click.echo(f"{indent}  (ignored weights: not numbers)")
    elif t in ("integer", "number"):
        lo = _num_or_none(click.prompt(f"{indent}  min (optional)", default="", show_default=False))
        hi = _num_or_none(click.prompt(f"{indent}  max (optional)", default="", show_default=False))
        if lo is not None:
            f["min"] = lo
        if hi is not None:
            f["max"] = hi
        if top_level:
            dist = click.prompt(f"{indent}  sample values in code from a distribution?",
                                type=click.Choice(["no", "uniform", "normal", "lognormal"]), default="no")
            if dist == "uniform" and lo is not None and hi is not None:
                f["distribution"] = {"type": "uniform"}
            elif dist == "normal":
                f["distribution"] = {"type": "normal", "mean": click.prompt(f"{indent}  mean", type=float),
                                     "std": click.prompt(f"{indent}  std", type=float)}
            elif dist == "lognormal":
                f["distribution"] = {"type": "lognormal", "median": click.prompt(f"{indent}  median", type=float),
                                     "sigma": click.prompt(f"{indent}  sigma (0.3-1.5)", type=float, default=0.7)}
    elif t in ("date", "datetime"):
        lo = click.prompt(f"{indent}  earliest (ISO, optional)", default="", show_default=False)
        hi = click.prompt(f"{indent}  latest (ISO, optional)", default="", show_default=False)
        if lo:
            f["min"] = lo
        if hi:
            f["max"] = hi
    elif t in ("string", "text"):
        ml = click.prompt(f"{indent}  max length (optional)", default="", show_default=False)
        if ml.strip().isdigit():
            f["max_length"] = int(ml)
    elif t == "uuid" and top_level:
        if click.confirm(f"{indent}  generate in code (no LLM tokens)?", default=True):
            f["generator"] = "uuid"
    if t == "array":
        item_t = click.prompt(f"{indent}  item type", type=click.Choice(["string", "integer", "number", "enum",
                                                                          "object"]), default="string")
        if item_t == "enum":
            f["items"] = {"type": "enum", "values": _csv(click.prompt(f"{indent}  item values (comma-separated)"))}
        elif item_t == "object":
            click.echo(f"{indent}  item properties:")
            props = []
            while (p := _prompt_field(click, indent + "    ", top_level=False)) is not None:
                props.append(p)
            f["items"] = {"type": "object", "properties": props}
        else:
            f["items"] = item_t
        lo = click.prompt(f"{indent}  min items (optional)", default="", show_default=False)
        hi = click.prompt(f"{indent}  max items (optional)", default="", show_default=False)
        if lo.strip().isdigit():
            f["min_items"] = int(lo)
        if hi.strip().isdigit() and int(hi) > 0:
            f["max_items"] = int(hi)
    if t == "object":
        click.echo(f"{indent}  properties of '{name}':")
        props = []
        while (p := _prompt_field(click, indent + "    ", top_level=False)) is not None:
            props.append(p)
        if not props:
            click.echo(f"{indent}  (no properties given; storing as a string instead)")
            f["type"] = "string"
        else:
            f["properties"] = props
    if not f.get("generator"):
        if click.confirm(f"{indent}  can it be null?", default=False):
            f["nullable"] = True
        if top_level and t in ("string", "email", "url") and click.confirm(f"{indent}  must be unique?",
                                                                            default=False):
            f["unique"] = True
    return f


def interactive_define(default_name: str = "dataset") -> dict[str, Any]:
    """Guided schema definition in the terminal. Returns a validated schema dict."""
    import click

    click.secho("Define a dataset schema", bold=True)
    click.echo("Descriptions matter most: they are what makes generated data realistic.\n")
    name = re.sub(r"\W+", "_", click.prompt("Dataset name", default=default_name)).strip("_").lower() or "dataset"
    description = click.prompt("What does one record represent?")
    context = click.prompt("Domain knowledge for realism (norms, typical ranges, correlations; optional)",
                           default="", show_default=False)
    count = click.prompt("How many records", type=int, default=500)
    click.echo("\nFields:")
    fields: list[dict[str, Any]] = []
    while True:
        f = _prompt_field(click)
        if f is None:
            if fields:
                break
            click.echo("  at least one field is required")
            continue
        fields.append(f)
    click.echo("\nConsistency rules in plain English (e.g. 'senior roles pay more'); empty line to finish:")
    rules = []
    while r := click.prompt("  rule", default="", show_default=False).strip():
        rules.append(r)
    scenarios = click.prompt("Scenarios for Claude to plan up front (diversity; 0 = off)", type=int,
                             default=max(10, min(60, count // 20)))
    schema: dict[str, Any] = {"name": name, "description": description}
    if context:
        schema["context"] = context
    schema.update({"count": count, "batch_size": 10, "fields": fields})
    if rules:
        schema["rules"] = rules
    if scenarios:
        schema["diversity"] = {"scenarios": scenarios}
    schema_from_dict(schema)  # raises SchemaError with a precise message if anything is off
    return schema
