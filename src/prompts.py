"""Prompt construction.

The system prompt holds everything that is constant for a table (dataset
description, domain context, field guide, rules, examples, quality bar) and is
marked for prompt caching, so repeated batches pay for it once. The user
message holds only what changes per request: the slot briefs and the list of
values to avoid repeating.
"""

from __future__ import annotations

import json
from typing import Any

from .sampling import Slot
from .schema_builder import SLOT_KEY, TOOL_NAME, FieldSpec, TableSpec, describe, to_jsonable

QUALITY_BAR = """\
- Coherence: every field in a record must agree with every other field and with the slot's brief. \
Numbers, dates, categories and free text must tell the same story.
- Realism: a domain expert skimming the data should not be able to tell it is synthetic. Use plausible, \
specific details rather than generic filler. Reflect real-world messiness where it would naturally occur \
(uneven lengths, informal tone in user-written text, occasional typos in customer-written text, etc.).
- Diversity: vary names, entities, phrasing, sentence structure, length and tone across records. Never \
start multiple records' text fields the same way. Avoid stock phrases and template-like output.
- Fidelity to the brief: values listed under "fixed values" and "target values" must be used exactly as \
given. Fields listed as null must be null.
- Output exactly one record per slot, set "{slot}" to the slot number, and return the slots in order."""

PRIVACY_BAR = """\
- Privacy: all people, companies, addresses, emails, phone numbers and identifiers must be invented. \
Never reproduce personal data of real private individuals. Public-facing brand or place names are fine \
when the context calls for them. For emails and URLs prefer plausible fictional domains."""


def _field_line(name: str, f: FieldSpec, indent: str = "") -> list[str]:
    kind = f.type
    if f.type == "enum":
        kind = "one of " + ", ".join(map(repr, f.values or []))
    elif f.type == "array" and f.items is not None:
        kind = f"list of {f.items.type}"
    if f.nullable:
        kind += ", nullable"
    desc = describe(f)
    lines = [f"{indent}- {name} ({kind})" + (f": {desc}" if desc else "")]
    if f.type == "object" and f.properties:
        for p in f.properties:
            lines += _field_line(p.name, p, indent + "  ")
    if f.type == "array" and f.items is not None and f.items.type == "object" and f.items.properties:
        for p in f.items.properties:
            lines += _field_line(p.name, p, indent + "  ")
    return lines


def system_prompt(table: TableSpec, privacy: bool = True, mode: str = "json") -> str:
    parts = [
        f'You are a senior data engineer and domain expert generating a high-quality synthetic dataset '
        f'called "{table.name}". The data will be used where realism, internal consistency and variety matter '
        f'(ML training, application testing, research, product demos). Write all natural-language content '
        f'in {table.language}.'
    ]
    if table.description:
        parts.append(f"<dataset_description>\n{table.description}\n</dataset_description>")
    if table.context:
        parts.append(f"<domain_context>\n{table.context}\n</domain_context>")

    lines: list[str] = []
    for k, f in table.llm_fields.items():
        lines += _field_line(k, f)
    parts.append("<fields>\n" + "\n".join(lines) + "\n</fields>")

    code_side = {f.name: f for f in table.fields if f.code_generated}
    if code_side:
        notes = []
        for k, f in code_side.items():
            if f.ref:
                notes.append(f"- {k}: references {f.ref}; the related record is described in each slot's brief.")
            else:
                notes.append(f"- {k}: assigned automatically ({f.generator}); do not output it.")
        parts.append("<assigned_elsewhere>\n" + "\n".join(notes) + "\n</assigned_elsewhere>")

    if table.rules:
        parts.append("<rules>\n" + "\n".join(f"- {r}" for r in table.rules) + "\n</rules>")
    if table.checks:
        parts.append("<hard_checks>\nEvery record is validated with these Python expressions over its fields "
                     "(dates as date objects; `parent` is the related record from the brief). Records where any "
                     "expression is false are discarded, so make sure all of them hold:\n"
                     + "\n".join(f"- {c}" for c in table.checks) + "\n</hard_checks>")
    if table.examples:
        ex = "\n".join(json.dumps(e, ensure_ascii=False, default=str) for e in table.examples[:5])
        parts.append("<example_records>\nThese show the expected style and level of detail. "
                     f"Do not copy them or their entities.\n{ex}\n</example_records>")
    bar = QUALITY_BAR.format(slot=SLOT_KEY)
    if privacy:
        bar += "\n" + PRIVACY_BAR
    parts.append("<quality_bar>\n" + bar + "\n</quality_bar>")
    if mode == "tool":
        parts.append(f"Deliver the records by calling the {TOOL_NAME} tool exactly once.")
    return "\n\n".join(parts)


def _fmt(v: Any) -> str:
    return json.dumps(to_jsonable(v), ensure_ascii=False, default=str)


def slot_brief(s: Slot, table: TableSpec) -> str:
    lines = [f"Slot {s.index}:"]
    if s.scenario:
        lines.append(f"  scenario: {s.scenario}")
    if s.fixed:
        lines.append("  fixed values: " + "; ".join(f"{k} = {_fmt(v)}" for k, v in s.fixed.items()))
    if s.targets:
        lines.append("  target values: " + "; ".join(f"{k} = {_fmt(v)}" for k, v in s.targets.items()))
    if s.nulls:
        lines.append("  null fields: " + ", ".join(s.nulls))
    for k, ctx in s.parent_context.items():
        f = table.field_map[k]
        lines.append(f"  related {f.ref_table} record ({k} = {_fmt(s.refs[k])}): {_fmt(ctx)}")
    if s.edge_case:
        lines.append("  edge case: make this an unusual but entirely valid record: a rare combination, "
                     "boundary value or atypical situation that real data contains and naive "
                     "systems mishandle.")
    return "\n".join(lines)


def user_prompt(slots: list[Slot], table: TableSpec, avoid: dict[str, list[str]] | None = None,
                feedback: list[str] | None = None) -> str:
    parts = [f"Generate {len(slots)} record{'s' if len(slots) != 1 else ''} for \"{table.name}\", "
             "one per slot below."]
    parts.append("\n\n".join(slot_brief(s, table) for s in slots))
    avoid = {k: v for k, v in (avoid or {}).items() if v}
    if avoid:
        lines = ["Already used elsewhere in the dataset (do not reuse these or close variants):"]
        for k, vals in avoid.items():
            lines.append(f"- {k}: " + "; ".join(v[:80] for v in vals))
        parts.append("\n".join(lines))
    if feedback:
        parts.append("Earlier records were rejected by validation for these reasons; make sure every record "
                     "satisfies them:\n" + "\n".join(f"- {x}" for x in feedback))
    return "\n\n".join(parts)


def scenario_prompt(table: TableSpec, n: int) -> str:
    guidance = table.diversity.scenario_guidance or (
        "Cover the realistic range of situations, user types and contexts this dataset should represent, "
        "from the common to the less common, so that records seeded from them are meaningfully different.")
    fields = ", ".join(table.llm_fields)
    return (
        f"Before generating the dataset, plan its coverage. Propose {n} distinct scenarios. Each scenario is a "
        f"one- or two-sentence seed describing a specific situation that one or more records could be built "
        f"around (records have these fields: {fields}).\n\n{guidance}\n\n"
        "Make scenarios concrete and mutually distinct. Do not fix values for fields that the per-record briefs "
        "will control (categories with target distributions, dates, numeric amounts) - describe situations, "
        "not field values."
    )


SCENARIO_SCHEMA = {
    "type": "object",
    "properties": {"scenarios": {"type": "array", "items": {"type": "string"}}},
    "required": ["scenarios"],
    "additionalProperties": False,
}
