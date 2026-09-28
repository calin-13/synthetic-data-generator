"""Schema definition & validation.

A *dataset schema* describes one table (``fields`` at the top level) or several
related tables (``tables``). It is written as JSON (or YAML) and validated with
Pydantic, so mistakes are caught with a precise message before any API call.

The schema is deliberately richer than JSON Schema: besides types and
constraints it carries *intent* (descriptions, domain context, rules,
target distributions, diversity settings) that drives prompting, sampling
and validation. This module also compiles it to:

* the **API JSON Schema** sent to Claude's structured outputs (only keywords
  the constrained-decoding grammar supports; every other constraint is
  written into the field description), and
* a **Pydantic record model** that enforces *every* constraint locally.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import re
import uuid as _uuid
from pathlib import Path
from typing import Annotated, Any, Literal, Optional

from pydantic import (AfterValidator, BaseModel, BeforeValidator, ConfigDict, Field, PrivateAttr,
                      ValidationError, create_model, field_validator, model_validator)


FieldType = Literal["string", "text", "integer", "number", "boolean", "date", "datetime", "time",
                    "email", "url", "uuid", "enum", "array", "object"]
FIELD_TYPES: tuple[str, ...] = FieldType.__args__  # type: ignore[attr-defined]
NUMERIC_TYPES = {"integer", "number"}
TEMPORAL_TYPES = {"date", "datetime", "time"}
STRINGISH_TYPES = {"string", "text", "email", "url", "uuid"}

SLOT_KEY = "_slot"
MAX_UNION_PARAMS = 16      # documented structured-outputs limit on union-typed parameters
MAX_OPTIONAL_PARAMS = 24   # documented structured-outputs limit on optional parameters

_FORMATS = {"date": "date", "datetime": "date-time", "time": "time", "email": "email", "url": "uri",
            "uuid": "uuid"}
_UNSAFE_PATTERN_TOKENS = ("(?", "\\b", "\\B", "\\1", "\\2", "\\3", "\\k", "\\p", "\\P")
_EMAIL = re.compile(r"^[A-Za-z0-9._%+'-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
_URL = re.compile(r"^https?://[^\s/$.?#][^\s]*$", re.I)
_IDENT = re.compile(r"^[A-Za-z_]\w*$")


class SchemaError(ValueError):
    """A malformed schema. The message points at the offending location."""


def parse_temporal(ftype: str, value: Any) -> Any:
    """Parse a date/datetime/time value. Datetimes are normalized to UTC-aware."""
    if value is None:
        return None
    if isinstance(value, _dt.datetime):
        v: Any = value
    elif isinstance(value, _dt.date):
        v = _dt.datetime.combine(value, _dt.time()) if ftype == "datetime" else value
    elif isinstance(value, _dt.time):
        v = value
    elif ftype == "date":
        v = _dt.date.fromisoformat(str(value).strip()[:10] if "T" in str(value) else str(value).strip())
    elif ftype == "datetime":
        v = _dt.datetime.fromisoformat(str(value).strip().replace("Z", "+00:00").replace(" ", "T", 1))
    else:
        v = _dt.time.fromisoformat(str(value).strip())
    if ftype == "date" and isinstance(v, _dt.datetime):
        v = v.date()
    if ftype == "datetime" and v.tzinfo is None:
        v = v.replace(tzinfo=_dt.timezone.utc)
    return v


def to_jsonable(v: Any) -> Any:
    if isinstance(v, _dt.datetime):
        return v.astimezone(_dt.timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(v, (_dt.date, _dt.time)):
        return v.isoformat()
    return v


# ---------------------------------------------------------------------------
# Schema models
# ---------------------------------------------------------------------------

class Distribution(BaseModel):
    """Sample this field in code (Claude receives the value as a fixed fact)."""

    model_config = ConfigDict(extra="forbid")
    type: Literal["uniform", "normal", "lognormal", "exponential"]
    mean: Optional[float] = None
    std: Optional[float] = Field(None, gt=0)
    median: Optional[float] = Field(None, gt=0)
    sigma: Optional[float] = Field(None, gt=0)

    @model_validator(mode="after")
    def _params(self) -> "Distribution":
        need = {"uniform": [], "normal": ["mean", "std"], "lognormal": ["median", "sigma"],
                "exponential": ["mean"]}[self.type]
        missing = [k for k in need if getattr(self, k) is None]
        if missing:
            raise ValueError(f"distribution '{self.type}' needs {missing}")
        if self.type == "exponential" and self.mean <= 0:  # type: ignore[operator]
            raise ValueError("exponential mean must be positive")
        return self


def _shorthand(v: Any, name: str = "item") -> Any:
    if isinstance(v, str):
        return {"name": name, "type": v}
    if isinstance(v, dict) and "name" not in v:
        return {"name": name, **v}
    return v


def _field_list(v: Any) -> Any:
    """Accept fields as a list of {name, ...} or a mapping {name: def | "type"}."""
    if isinstance(v, dict):
        return [{**_shorthand(d, k), "name": k} if isinstance(d, dict) else _shorthand(d, k)
                for k, d in v.items()]
    return v


class FieldSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    type: FieldType = "string"
    description: str = ""
    nullable: bool = False
    null_rate: Optional[float] = Field(None, ge=0, le=1,
                                       description="Exact share of nulls, decided in code.")
    unique: bool = False
    values: Optional[list[Any]] = None
    weights: Optional[list[float] | dict[str, float]] = None
    stratify: Optional[bool] = None
    min: Any = None
    max: Any = None
    min_length: Optional[int] = Field(None, ge=0)
    max_length: Optional[int] = Field(None, ge=1)
    pattern: Optional[str] = None
    items: Optional["FieldSpec"] = None
    min_items: Optional[int] = Field(None, ge=0)
    max_items: Optional[int] = Field(None, ge=1)
    properties: Optional[list["FieldSpec"]] = None
    generator: Optional[Literal["uuid", "sequence"]] = None
    start: int = 1
    format: Optional[str] = None
    distribution: Optional[Distribution] = None
    decimals: Optional[int] = Field(None, ge=0, le=10)
    ref: Optional[str] = None
    ref_context: list[str] = Field(default_factory=list)
    ref_skew: float = Field(0.0, ge=0)
    examples: list[Any] = Field(default_factory=list)
    inherit_type: bool = Field(False, exclude=True, description="internal: take the type from the ref target")

    _props: dict[str, "FieldSpec"] = PrivateAttr(default_factory=dict)

    # ---- normalization ------------------------------------------------------------
    @model_validator(mode="before")
    @classmethod
    def _before(cls, raw: Any) -> Any:
        if not isinstance(raw, dict):
            return raw
        raw = dict(raw)
        if "values" in raw and "type" not in raw:
            raw["type"] = "enum"
        if "ref" in raw and "type" not in raw:
            raw["type"] = "string"
            raw["inherit_type"] = True
        if isinstance(raw.get("distribution"), str):
            raw["distribution"] = {"type": raw["distribution"]}
        if "items" in raw:
            raw["items"] = _shorthand(raw["items"], f"{raw.get('name', 'field')}[]")
        if "properties" in raw:
            raw["properties"] = _field_list(raw["properties"])
        return raw

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("field name must not be empty")
        return v

    @model_validator(mode="after")
    def _check(self) -> "FieldSpec":  # noqa: C901 - one place for all cross-key rules
        n, t = self.name, self.type

        def err(msg: str) -> ValueError:
            return ValueError(f"field '{n}': {msg}")

        if self.null_rate is not None:
            self.nullable = True
        # enum
        if t == "enum":
            if not self.values:
                raise err("enum fields need a non-empty 'values' list")
            if len({str(v).strip().lower() for v in self.values}) != len(self.values):
                raise err("enum values must be unique ignoring case")
            if isinstance(self.weights, dict):
                missing = {str(v) for v in self.values} - set(map(str, self.weights))
                if missing:
                    raise err(f"weights missing for {sorted(missing)}")
                self.weights = [float(self.weights[str(v)]) for v in self.values]
            if self.weights is not None:
                if len(self.weights) != len(self.values) or any(w < 0 for w in self.weights) \
                        or sum(self.weights) <= 0:
                    raise err("weights must be non-negative, one per value, not all zero")
                total = float(sum(self.weights))
                self.weights = [w / total for w in self.weights]
            if self.stratify is None:
                self.stratify = self.weights is not None
            if self.stratify and self.weights is None:
                self.weights = [1.0 / len(self.values)] * len(self.values)
        elif self.values is not None or self.weights is not None:
            raise err("'values'/'weights' are only valid for type enum")
        else:
            self.stratify = False
        # ranges
        if t in NUMERIC_TYPES:
            for b in ("min", "max"):
                v = getattr(self, b)
                if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float))):
                    raise err(f"{b} must be a number")
        elif t in TEMPORAL_TYPES:
            try:
                self.min = parse_temporal(t, self.min)
                self.max = parse_temporal(t, self.max)
            except (ValueError, TypeError) as e:
                raise err(f"cannot parse min/max as {t}: {e}") from None
        elif self.min is not None or self.max is not None:
            raise err("min/max apply to numeric and date/time fields "
                      "(use min_length/max_length or min_items/max_items)")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise err("min is greater than max")
        if self.decimals is not None and t != "number":
            raise err("decimals applies to number fields")
        # strings
        if (self.min_length is not None or self.max_length is not None or self.pattern) \
                and t not in STRINGISH_TYPES:
            raise err("min_length/max_length/pattern apply to string-like fields")
        if self.min_length is not None and self.max_length is not None and self.min_length > self.max_length:
            raise err("min_length is greater than max_length")
        if self.pattern:
            try:
                re.compile(self.pattern)
            except re.error as e:
                raise err(f"invalid pattern: {e}") from None
        # containers
        if t == "array":
            if self.items is None:
                raise err("array fields need 'items' (e.g. \"items\": \"string\")")
            if self.items.code_generated or self.items.sampled:
                raise err("generators/refs/distributions are only supported on top-level fields")
            if self.min_items is not None and self.max_items is not None and self.min_items > self.max_items:
                raise err("min_items is greater than max_items")
        elif self.items is not None or self.min_items is not None or self.max_items is not None:
            raise err("items/min_items/max_items are only valid for arrays")
        if t == "object":
            if not self.properties:
                raise err("object fields need non-empty 'properties'")
            names = [p.name for p in self.properties]
            if len(set(names)) != len(names):
                raise err(f"duplicate property names in {names}")
            for p in self.properties:
                if p.code_generated or p.sampled or p.stratify:
                    raise err(f"property '{p.name}': generators/refs/distributions/weights are only "
                              "supported on top-level fields")
            self._props = {p.name: p for p in self.properties}
        elif self.properties is not None:
            raise err("'properties' is only valid for objects")
        # code-side generation
        if self.generator == "uuid" and t not in ("uuid", "string"):
            raise err("generator uuid requires type uuid or string")
        if self.generator == "sequence":
            if t not in ("integer", "string"):
                raise err("generator sequence requires type integer or string")
            if self.format and "{" not in self.format:
                raise err("sequence format needs a placeholder, e.g. 'ORD-{:06d}'")
        if self.generator:
            self.unique = True
        if self.distribution is not None:
            if t not in NUMERIC_TYPES | {"date", "datetime"}:
                raise err("distributions apply to integer/number/date/datetime fields")
            if t in ("date", "datetime") and self.distribution.type != "uniform":
                raise err("date/datetime fields support only the uniform distribution")
            if self.distribution.type == "uniform" and (self.min is None or self.max is None):
                raise err("uniform distribution needs both min and max")
        if self.ref is not None:
            if not re.fullmatch(r"[A-Za-z_]\w*\.[A-Za-z_]\w*", self.ref):
                raise err("ref must look like 'table.field'")
        elif self.ref_context or self.ref_skew:
            raise err("ref_context/ref_skew require 'ref'")
        if sum(bool(x) for x in (self.generator, self.ref, self.distribution, self.stratify)) > 1:
            raise err("generator, ref, distribution and weights/stratify are mutually exclusive")
        return self

    # ---- derived ------------------------------------------------------------------------
    @property
    def code_generated(self) -> bool:
        return self.generator is not None or self.ref is not None

    @property
    def sampled(self) -> bool:
        return self.distribution is not None

    @property
    def ref_table(self) -> Optional[str]:
        return self.ref.split(".", 1)[0] if self.ref else None

    @property
    def ref_field(self) -> Optional[str]:
        return self.ref.split(".", 1)[1] if self.ref else None

    @property
    def prop_map(self) -> dict[str, "FieldSpec"]:
        return self._props


class Diversity(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scenarios: int = Field(0, ge=0, le=500, description="Distinct scenarios Claude plans up front.")
    scenario_guidance: str = ""
    edge_case_ratio: float = Field(0.0, ge=0, le=1)
    avoid_repeats: list[str] = Field(default_factory=list)


class TableSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "records"
    description: str = ""
    context: str = ""
    count: int = Field(100, ge=1)
    batch_size: int = Field(10, ge=1, le=100)
    fields: list[FieldSpec]
    rules: list[str] = Field(default_factory=list)
    checks: list[str] = Field(default_factory=list)
    examples: list[dict[str, Any]] = Field(default_factory=list)
    diversity: Diversity = Field(default_factory=Diversity)
    strict_distribution: bool = False
    language: str = "English"

    _map: dict[str, FieldSpec] = PrivateAttr(default_factory=dict)
    _record_model: Any = PrivateAttr(None)

    @field_validator("fields", mode="before")
    @classmethod
    def _fields(cls, v: Any) -> Any:
        return _field_list(v)

    @model_validator(mode="after")
    def _check(self) -> "TableSpec":
        if not _IDENT.match(self.name):
            raise ValueError(f"table name '{self.name}' must be an identifier (letters, digits, _)")
        if not self.fields:
            raise ValueError(f"table '{self.name}': 'fields' must not be empty")
        names = [f.name for f in self.fields]
        dup = {x for x in names if names.count(x) > 1}
        if dup:
            raise ValueError(f"table '{self.name}': duplicate field names {sorted(dup)}")
        self._map = {f.name: f for f in self.fields}
        if not self.llm_fields:
            raise ValueError(f"table '{self.name}': at least one field must be generated by Claude")
        for fname in self.diversity.avoid_repeats:
            if fname not in self._map:
                raise ValueError(f"table '{self.name}': avoid_repeats names unknown field '{fname}'")
        for c in self.checks:
            try:
                compile(c, "<check>", "eval")
            except SyntaxError as e:
                raise ValueError(f"table '{self.name}': invalid check {c!r}: {e.msg}") from None
        return self

    @property
    def field_map(self) -> dict[str, FieldSpec]:
        return self._map

    @property
    def llm_fields(self) -> dict[str, FieldSpec]:
        return {f.name: f for f in self.fields if not f.code_generated}

    @property
    def stratified_fields(self) -> dict[str, FieldSpec]:
        return {f.name: f for f in self.fields if f.stratify and not f.code_generated}

    @property
    def sampled_fields(self) -> dict[str, FieldSpec]:
        return {f.name: f for f in self.fields if f.sampled}

    @property
    def ref_fields(self) -> dict[str, FieldSpec]:
        return {f.name: f for f in self.fields if f.ref}

    @property
    def unique_fields(self) -> dict[str, FieldSpec]:
        return {f.name: f for f in self.fields if f.unique}


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = "claude-sonnet-5-5"
    concurrency: int = Field(16, ge=1, le=256)
    max_tokens: int = Field(16000, ge=256)
    temperature: Optional[float] = Field(None, ge=0, le=1)
    seed: Optional[int] = None
    privacy_guardrails: bool = True
    mode: Literal["json", "tool"] = Field(
        "json", description="json = output_config.format (JSON outputs); tool = strict tool use")


_SETTINGS_KEYS = set(Settings.model_fields)


class DatasetSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tables: list[TableSpec]
    settings: Settings = Field(default_factory=Settings)

    @model_validator(mode="before")
    @classmethod
    def _shape(cls, raw: Any) -> Any:
        if not isinstance(raw, dict):
            return raw
        raw = dict(raw)
        settings = dict(raw.pop("settings", None) or {})
        for k in list(raw):
            if k in _SETTINGS_KEYS:
                settings.setdefault(k, raw.pop(k))
        if "fields" in raw:  # single-table shorthand
            if "tables" in raw:
                raise ValueError("use either top-level 'fields' or 'tables', not both")
            raw = {"tables": [raw]}
        elif isinstance(raw.get("tables"), dict):
            raw["tables"] = [{"name": k, **v} for k, v in raw["tables"].items()]
        raw["settings"] = settings
        return raw

    @model_validator(mode="after")
    def _relations(self) -> "DatasetSchema":
        names = [t.name for t in self.tables]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate table names: {names}")
        by_name = {t.name: t for t in self.tables}
        for t in self.tables:
            for f in t.ref_fields.values():
                where = f"table '{t.name}', field '{f.name}'"
                parent = by_name.get(f.ref_table or "")
                if parent is None:
                    raise ValueError(f"{where}: ref points at unknown table '{f.ref_table}'")
                if parent is t:
                    raise ValueError(f"{where}: self-references are not supported")
                pf = parent.field_map.get(f.ref_field or "")
                if pf is None:
                    raise ValueError(f"{where}: table '{parent.name}' has no field '{f.ref_field}'")
                if pf.type in ("array", "object"):
                    raise ValueError(f"{where}: cannot reference a {pf.type} field")
                if not pf.unique:
                    raise ValueError(f"{where}: referenced field {f.ref} must be unique "
                                     "(add \"unique\": true or a generator)")
                if f.inherit_type:
                    f.type, f.values = pf.type, pf.values
                elif f.type != pf.type:
                    raise ValueError(f"{where}: type '{f.type}' does not match {f.ref} ('{pf.type}'); "
                                     "omit 'type' to inherit it")
                for c in f.ref_context:
                    if c not in parent.field_map:
                        raise ValueError(f"{where}: ref_context names unknown field '{parent.name}.{c}'")
        self.ordered_tables()
        return self

    def table(self, name: str) -> TableSpec:
        for t in self.tables:
            if t.name == name:
                return t
        raise SchemaError(f"no table named '{name}'; have {[t.name for t in self.tables]}")

    def ordered_tables(self) -> list[TableSpec]:
        """Tables in dependency order (parents before children)."""
        by_name = {t.name: t for t in self.tables}
        order: list[TableSpec] = []
        state: dict[str, int] = {}

        def visit(t: TableSpec, chain: list[str]) -> None:
            if state.get(t.name) == 2:
                return
            if state.get(t.name) == 1:
                raise ValueError(f"circular table references: {' -> '.join(chain + [t.name])}")
            state[t.name] = 1
            for f in t.ref_fields.values():
                if f.ref_table in by_name:
                    visit(by_name[f.ref_table], chain + [t.name])  # type: ignore[index]
            state[t.name] = 2
            order.append(t)

        for t in self.tables:
            visit(t, [])
        return order

    def fingerprint(self) -> str:
        """Stable hash of the schema (for reproducibility manifests)."""
        blob = json.dumps(self.model_dump(mode="json"), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


FieldSpec.model_rebuild()


def _format_validation_error(e: ValidationError) -> str:
    lines = []
    for err in e.errors():
        loc = ".".join(str(x) for x in err["loc"] if not str(x).startswith("function-"))
        msg = err["msg"].removeprefix("Value error, ")
        lines.append(f"{loc}: {msg}" if loc else msg)
    return "invalid schema:\n  " + "\n  ".join(dict.fromkeys(lines))


def schema_from_dict(raw: dict[str, Any], default_name: str | None = None) -> DatasetSchema:
    if not isinstance(raw, dict):
        raise SchemaError("schema must be a JSON object")
    raw = dict(raw)
    if "fields" in raw and "name" not in raw and default_name:
        name = re.sub(r"\W+", "_", default_name).strip("_") or "records"
        raw["name"] = name if _IDENT.match(name) else "records"
    try:
        return DatasetSchema.model_validate(raw)
    except ValidationError as e:
        raise SchemaError(_format_validation_error(e)) from None


def read_schema_file(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise SchemaError(f"schema file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        import yaml
        return yaml.safe_load(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise SchemaError(f"{path}: invalid JSON at line {e.lineno}: {e.msg}") from None


def load_schema(path: str | Path) -> DatasetSchema:
    return schema_from_dict(read_schema_file(path), default_name=Path(path).stem)


def save_schema(raw: dict[str, Any], path: str | Path) -> Path:
    """Validate and write a schema dict (JSON, or YAML by extension)."""
    schema_from_dict(raw, default_name=Path(path).stem)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix in (".yaml", ".yml"):
        import yaml
        path.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8")
    else:
        path.write_text(json.dumps(raw, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Compilation to the API JSON Schema (structured outputs)
# ---------------------------------------------------------------------------

def _fmt_bound(v: Any) -> str:
    return str(to_jsonable(v))


def constraint_notes(f: FieldSpec) -> list[str]:
    """Human-readable versions of constraints the grammar can't enforce."""
    notes: list[str] = []
    if f.min is not None and f.max is not None:
        notes.append(f"between {_fmt_bound(f.min)} and {_fmt_bound(f.max)} inclusive")
    elif f.min is not None:
        notes.append(f"at least {_fmt_bound(f.min)}")
    elif f.max is not None:
        notes.append(f"at most {_fmt_bound(f.max)}")
    if f.type == "integer":
        notes.append("whole number")
    if f.type == "number" and f.decimals is not None:
        notes.append(f"round to {f.decimals} decimal places")
    if f.min_length is not None and f.max_length is not None:
        notes.append(f"{f.min_length}-{f.max_length} characters")
    elif f.max_length is not None:
        notes.append(f"at most {f.max_length} characters")
    elif f.min_length is not None:
        notes.append(f"at least {f.min_length} characters")
    if f.pattern:
        notes.append(f"must fully match regex {f.pattern}")
    if f.min_items is not None and f.max_items is not None:
        notes.append(f"{f.min_items}-{f.max_items} items")
    elif f.min_items is not None:
        notes.append(f"at least {f.min_items} items")
    elif f.max_items is not None:
        notes.append(f"at most {f.max_items} items")
    if f.null_rate is not None:
        notes.append("null only when the slot's brief lists it under null fields")
    if f.unique:
        notes.append("unique across the whole dataset")
    if f.type == "text":
        notes.append("free-form natural-language text")
    return notes


def describe(f: FieldSpec) -> str:
    parts = [f.description] if f.description else []
    notes = constraint_notes(f)
    if notes:
        parts.append("Constraints: " + "; ".join(notes) + ".")
    if f.examples:
        parts.append("Style examples (do not copy verbatim): " + ", ".join(map(repr, f.examples[:5])))
    return " ".join(parts)


def _base_schema(f: FieldSpec) -> dict[str, Any]:
    t = f.type
    if t == "enum":
        vals = f.values or []
        kinds = {type(v) for v in vals}
        if kinds <= {str}:
            return {"type": "string", "enum": list(vals)}
        if kinds <= {int}:
            return {"type": "integer", "enum": list(vals)}
        if kinds <= {int, float}:
            return {"type": "number", "enum": list(vals)}
        return {"type": "string", "enum": [str(v) for v in vals]}
    if t in ("string", "text"):
        s: dict[str, Any] = {"type": "string"}
        if f.pattern and not any(tok in f.pattern for tok in _UNSAFE_PATTERN_TOKENS):
            s["pattern"] = f.pattern
        return s
    if t in _FORMATS:
        return {"type": "string", "format": _FORMATS[t]}
    if t in ("integer", "number", "boolean"):
        return {"type": t}
    if t == "array":
        s = {"type": "array", "items": api_field_schema(f.items)}  # type: ignore[arg-type]
        if f.min_items is not None and f.min_items >= 1:
            s["minItems"] = 1  # the only non-zero value the grammar supports
        return s
    if t == "object":
        return object_schema(f.properties or [])
    raise ValueError(f"unhandled type {t}")


def api_field_schema(f: FieldSpec) -> dict[str, Any]:
    s = _base_schema(f)
    if f.nullable:
        s = {"anyOf": [s, {"type": "null"}]}
    desc = describe(f)
    if desc:
        s["description"] = desc
    return s


def object_schema(fields: list[FieldSpec], leading: dict[str, Any] | None = None) -> dict[str, Any]:
    props: dict[str, Any] = dict(leading or {})
    props.update({f.name: api_field_schema(f) for f in fields})
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def record_schema(table: TableSpec, with_slot: bool = True) -> dict[str, Any]:
    """One record as Claude produces it (code-generated fields excluded)."""
    leading = {SLOT_KEY: {"type": "integer", "description": "The slot number this record fulfils (1-based)."}} \
        if with_slot else None
    return object_schema(list(table.llm_fields.values()), leading)


def batch_schema(table: TableSpec) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"records": {"type": "array", "items": record_schema(table),
                                   "description": "One record per requested slot, in slot order."}},
        "required": ["records"],
        "additionalProperties": False,
    }


def output_config(schema: dict[str, Any]) -> dict[str, Any]:
    """``output_config`` for JSON outputs mode."""
    return {"format": {"type": "json_schema", "schema": schema}}


TOOL_NAME = "submit_records"


def tool_definition(schema: dict[str, Any]) -> dict[str, Any]:
    """A strict tool whose input is the batch; used in tool-calling mode."""
    return {"name": TOOL_NAME, "strict": True, "input_schema": schema,
            "description": "Submit the generated records, one per requested slot."}


def full_record_schema(table: TableSpec) -> dict[str, Any]:
    """JSON Schema (draft 2020-12) of a *final* record with every constraint, for downstream consumers."""
    def conv(f: FieldSpec) -> dict[str, Any]:
        if f.type == "array":
            s: dict[str, Any] = {"type": "array", "items": conv(f.items)}  # type: ignore[arg-type]
            if f.min_items is not None:
                s["minItems"] = f.min_items
            if f.max_items is not None:
                s["maxItems"] = f.max_items
        elif f.type == "object":
            s = {"type": "object", "properties": {p.name: conv(p) for p in f.properties or []},
                 "required": [p.name for p in f.properties or []], "additionalProperties": False}
        else:
            s = _base_schema(f)
        if f.type in NUMERIC_TYPES:
            if f.min is not None:
                s["minimum"] = f.min
            if f.max is not None:
                s["maximum"] = f.max
        if f.min_length is not None:
            s["minLength"] = f.min_length
        if f.max_length is not None:
            s["maxLength"] = f.max_length
        if f.pattern:
            s["pattern"] = f.pattern
        if f.description:
            s["description"] = f.description
        if f.nullable:
            s = {"anyOf": [s, {"type": "null"}]}
        return s

    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "title": table.name, "type": "object",
            "properties": {f.name: conv(f) for f in table.fields}, "required": [f.name for f in table.fields],
            "additionalProperties": False}


def complexity(schema: dict[str, Any]) -> dict[str, Any]:
    """Count what drives grammar size and warn before hitting documented API limits."""
    stats = {"unions": 0, "objects": 0, "properties": 0, "optional": 0, "max_depth": 0}

    def walk(s: Any, depth: int) -> None:
        if not isinstance(s, dict):
            return
        if "anyOf" in s or isinstance(s.get("type"), list):
            stats["unions"] += 1
        if s.get("type") == "object":
            stats["objects"] += 1
            stats["max_depth"] = max(stats["max_depth"], depth)
            props = s.get("properties", {})
            stats["properties"] += len(props)
            stats["optional"] += len(set(props) - set(s.get("required", [])))
            for v in props.values():
                walk(v, depth + 1)
            return
        items = s.get("items")
        if isinstance(items, dict):
            walk(items, depth)
        for v in s.get("anyOf", []) or []:
            walk(v, depth)

    walk(schema, 0)
    warnings = []
    if stats["unions"] > MAX_UNION_PARAMS:
        warnings.append(f"{stats['unions']} nullable fields exceed the API limit of {MAX_UNION_PARAMS} "
                        "union-typed parameters; make some non-nullable or code-generated.")
    if stats["optional"] > MAX_OPTIONAL_PARAMS:
        warnings.append(f"{stats['optional']} optional parameters exceed the API limit of {MAX_OPTIONAL_PARAMS}.")
    if stats["properties"] > 120 or stats["max_depth"] > 5:
        warnings.append("large or deeply nested schema; if the API reports 'Schema is too complex for "
                        "compilation', flatten nested objects or split the table.")
    stats["warnings"] = warnings
    return stats


# ---------------------------------------------------------------------------
# Pydantic record model: enforces every constraint on final records
# ---------------------------------------------------------------------------

def _maybe_json(kind: type) -> Any:
    def parse(v: Any) -> Any:
        if isinstance(v, str) and v.strip()[:1] in ("[", "{"):
            try:
                parsed = json.loads(v)
            except json.JSONDecodeError:
                return v
            return parsed if isinstance(parsed, kind) else v
        if kind is list and hasattr(v, "tolist") and not isinstance(v, (str, bytes)):
            return list(v.tolist())  # numpy arrays from parquet
        return v
    return BeforeValidator(parse)


def _annotation(f: FieldSpec) -> Any:  # noqa: C901
    t = f.type
    if t == "enum":
        allowed = list(f.values or [])
        lookup = {str(v).strip().lower(): v for v in allowed}

        def enum_check(v: Any) -> Any:
            # structured outputs don't guarantee enum capitalization: match case-insensitively
            if v in allowed and not isinstance(v, bool):
                return v
            key = str(v).strip().lower()
            if key in lookup:
                return lookup[key]
            raise ValueError("not one of the allowed values")
        return Annotated[Any, AfterValidator(enum_check)]

    if t in NUMERIC_TYPES:
        py = int if t == "integer" else float
        extra: list[Any] = [Field(ge=f.min, le=f.max)]
        if t == "number" and f.decimals is not None:
            d = f.decimals
            extra.append(AfterValidator(lambda v: round(v, d)))

        def finite(v: Any) -> Any:
            if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
                raise ValueError("not a finite number")
            return v
        return Annotated[(py, AfterValidator(finite), *extra)]  # type: ignore[valid-type]

    if t == "boolean":
        return bool

    if t in TEMPORAL_TYPES:
        def temporal(v: Any) -> str:
            try:
                parsed = parse_temporal(t, v)
            except (ValueError, TypeError):
                raise ValueError(f"not a valid {t} (ISO 8601)") from None
            if f.min is not None and parsed < f.min:
                raise ValueError(f"before min {_fmt_bound(f.min)}")
            if f.max is not None and parsed > f.max:
                raise ValueError(f"after max {_fmt_bound(f.max)}")
            return to_jsonable(parsed)
        return Annotated[Any, AfterValidator(temporal)]

    if t in STRINGISH_TYPES:
        def string_check(v: str) -> str:
            v = v.strip()
            if not v and f.min_length != 0:
                raise ValueError("empty string")
            if t == "email" and not _EMAIL.match(v):
                raise ValueError("not a valid email address")
            if t == "url" and not _URL.match(v):
                raise ValueError("not a valid http(s) URL")
            if t == "uuid":
                try:
                    v = str(_uuid.UUID(v))
                except ValueError:
                    raise ValueError("not a valid UUID") from None
            if f.min_length is not None and len(v) < f.min_length:
                raise ValueError(f"too short (min {f.min_length} characters)")
            if f.max_length is not None and len(v) > f.max_length:
                raise ValueError(f"too long (max {f.max_length} characters)")
            if f.pattern and not re.fullmatch(f.pattern, v):
                raise ValueError(f"does not match pattern {f.pattern}")
            return v

        def to_str(v: Any) -> Any:
            if isinstance(v, (int, float)) and not isinstance(v, bool) and f.generator == "sequence":
                return str(v)
            return v
        return Annotated[str, BeforeValidator(to_str), AfterValidator(string_check)]

    if t == "array":
        item = _annotation(f.items)  # type: ignore[arg-type]
        if f.items is not None and f.items.nullable:
            item = Optional[item]
        return Annotated[list[item], _maybe_json(list),  # type: ignore[valid-type]
                         Field(min_length=f.min_items, max_length=f.max_items)]

    if t == "object":
        return Annotated[build_model(f"obj_{f.name}", f.properties or []), _maybe_json(dict)]

    raise ValueError(f"unhandled type {t}")


def build_model(name: str, fields: list[FieldSpec]) -> type[BaseModel]:
    defs: dict[str, Any] = {}
    for i, f in enumerate(fields):
        ann = _annotation(f)
        if f.nullable:
            defs[f"f{i}"] = (Optional[ann], Field(default=None, alias=f.name))
        else:
            defs[f"f{i}"] = (ann, Field(alias=f.name))
    safe = re.sub(r"\W", "_", name)
    return create_model(safe, __config__=ConfigDict(extra="forbid", populate_by_name=False), **defs)


def record_model(table: TableSpec) -> type[BaseModel]:
    """Pydantic model of a final record (cached per table)."""
    if table._record_model is None:
        table._record_model = build_model(f"{table.name}_record", table.fields)
    return table._record_model
