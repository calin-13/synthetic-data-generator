from __future__ import annotations

import json

import pytest

from src.schema_builder import (SchemaError, batch_schema, complexity, full_record_schema, load_schema,
                                record_model, save_schema, schema_from_dict, tool_definition)

from .conftest import EXAMPLES


def test_readme_example_schema():
    """The exact list-of-fields format from the project brief."""
    s = schema_from_dict({
        "fields": [
            {"name": "job_title", "type": "string", "description": "Job position title"},
            {"name": "salary", "type": "number", "min": 30000, "max": 200000},
            {"name": "skills", "type": "array", "items": "string"},
        ],
        "count": 500,
    }, default_name="jobs")
    t = s.tables[0]
    assert (t.name, t.count) == ("jobs", 500)
    assert t.field_map["skills"].items.type == "string"


def test_dict_fields_and_shorthand_are_accepted():
    s = schema_from_dict({"name": "x", "fields": {"a": "integer", "b": {"values": ["p", "q"]}}})
    t = s.tables[0]
    assert t.field_map["a"].type == "integer" and t.field_map["b"].type == "enum"


def test_derived_properties(basic):
    t = schema_from_dict(basic()).tables[0]
    assert set(t.stratified_fields) == {"tier"}
    assert set(t.sampled_fields) == {"age", "joined"}
    assert "id" not in t.llm_fields and "uid" not in t.llm_fields
    assert t.field_map["score"].nullable
    assert abs(sum(t.field_map["tier"].weights) - 1) < 1e-9
    assert t.field_map["address"].prop_map["zip"].type == "string"


def _field(s, name):
    return next(f for f in s["fields"] if f["name"] == name)


@pytest.mark.parametrize("mutate, msg", [
    (lambda s: _field(s, "name").update(colour="red"), "Extra inputs are not permitted"),
    (lambda s: _field(s, "tier").update(weights=[1, 2]), "weights"),
    (lambda s: _field(s, "age").update(min=100), "min is greater than max"),
    (lambda s: _field(s, "name").update(min=3), "min/max apply"),
    (lambda s: _field(s, "bio").update(pattern="("), "invalid pattern"),
    (lambda s: s.update(checks=["age >"]), "invalid check"),
    (lambda s: _field(s, "age").update(distribution={"type": "normal"}), "needs"),
    (lambda s: _field(s, "tags").pop("items"), "need 'items'"),
    (lambda s: _field(s, "email").update(type="colour"), "Input should be"),
    (lambda s: s["fields"].append({"name": "id", "type": "string"}), "duplicate field names"),
    (lambda s: _field(s, "tier").update(values=["a", "A"], weights=None), "unique ignoring case"),
])
def test_schema_errors_are_precise(basic, mutate, msg):
    s = basic()
    mutate(s)
    with pytest.raises(SchemaError, match=msg):
        schema_from_dict(s)


def test_multi_table_refs():
    good = {"tables": [
        {"name": "a", "fields": [{"name": "id", "type": "integer", "generator": "sequence"}, {"name": "x"}]},
        {"name": "b", "fields": [{"name": "a_id", "ref": "a.id"}, {"name": "y"}]},
    ]}
    s = schema_from_dict(good)
    assert [t.name for t in s.ordered_tables()] == ["a", "b"]
    assert s.table("b").field_map["a_id"].type == "integer"  # inherited from the parent

    bad = json.loads(json.dumps(good))
    bad["tables"][1]["fields"][0] = {"name": "a_id", "ref": "zzz.id"}
    with pytest.raises(SchemaError, match="unknown table"):
        schema_from_dict(bad)
    bad = json.loads(json.dumps(good))
    bad["tables"][1]["fields"][0] = {"name": "a_id", "ref": "a.x"}
    with pytest.raises(SchemaError, match="must be unique"):
        schema_from_dict(bad)
    cyc = {"tables": [
        {"name": "a", "fields": [{"name": "id", "type": "integer", "generator": "sequence"},
                                 {"name": "b_id", "ref": "b.id"}, {"name": "x"}]},
        {"name": "b", "fields": [{"name": "id", "type": "integer", "generator": "sequence"},
                                 {"name": "a_id", "ref": "a.id"}, {"name": "y"}]},
    ]}
    with pytest.raises(SchemaError, match="circular"):
        schema_from_dict(cyc)


def test_examples_load_within_api_limits():
    files = sorted(EXAMPLES.glob("*.json"))
    assert {f.stem for f in files} >= {"job_postings", "product_catalog", "customer_profiles"}
    for f in files:
        for t in load_schema(f).tables:
            assert not complexity(batch_schema(t))["warnings"], f


UNSUPPORTED = {"minimum", "maximum", "minLength", "maxLength", "maxItems", "exclusiveMinimum",
               "exclusiveMaximum", "multipleOf"}


def _keys(o):
    if isinstance(o, dict):
        for k, v in o.items():
            yield k
            if k == "properties":
                for vv in v.values():
                    yield from _keys(vv)
            else:
                yield from _keys(v)
    elif isinstance(o, list):
        for x in o:
            yield from _keys(x)


def test_api_schema_uses_only_supported_keywords():
    import anthropic

    for f in EXAMPLES.glob("*.json"):
        for t in load_schema(f).tables:
            s = batch_schema(t)
            assert not (set(_keys(s)) & UNSUPPORTED)

            def walk(o):
                if isinstance(o, dict):
                    if o.get("type") == "object":
                        assert o["additionalProperties"] is False
                        assert set(o["required"]) == set(o["properties"])
                    for v in o.values():
                        walk(v)
                elif isinstance(o, list):
                    for v in o:
                        walk(v)
            walk(s)
            # the SDK's own transformer finds nothing unsupported to move into descriptions
            assert "{minimum" not in json.dumps(anthropic.transform_schema(s))
            assert tool_definition(s)["strict"] is True


def test_constraints_move_into_descriptions(basic):
    t = schema_from_dict(basic()).tables[0]
    props = batch_schema(t)["properties"]["records"]["items"]["properties"]
    assert "between 18 and 90" in props["age"]["description"]
    assert "at most 400 characters" in props["bio"]["description"]
    assert props["score"]["anyOf"][1] == {"type": "null"}
    assert props["tags"]["minItems"] == 1
    assert list(props)[0] == "_slot"
    full = full_record_schema(t)
    assert full["properties"]["age"]["maximum"] == 90 and full["properties"]["tags"]["maxItems"] == 3


def test_record_model_enforces_everything(basic):
    t = schema_from_dict(basic()).tables[0]
    M = record_model(t)
    good = {"id": "P0001", "uid": "2f1e4b5a-3c7d-4e8f-9a0b-1c2d3e4f5a6b", "name": " Ann ", "email": "a@b.io",
            "tier": "GOLD", "age": 30.0, "joined": "2024-05-01", "score": None, "bio": "hi",
            "tags": '["x"]', "address": {"city": "Cluj", "zip": "400001"}}
    out = M.model_validate(good).model_dump(by_alias=True, mode="json")
    assert out["tier"] == "gold" and out["age"] == 30 and out["name"] == "Ann" and out["tags"] == ["x"]
    bad = dict(good, age=91, joined="2023-12-31", email="nope", tags=["a", "b", "c", "d"], extra=1)
    from pydantic import ValidationError
    with pytest.raises(ValidationError) as e:
        M.model_validate(bad)
    locs = {err["loc"][0] for err in e.value.errors()}
    assert locs >= {"age", "joined", "email", "tags", "extra"}


def test_save_and_reload_roundtrip(tmp_path, basic):
    p = save_schema(basic(), tmp_path / "people.json")
    s = load_schema(p)
    assert s.tables[0].name == "people"
    # the normalized dump (used in manifests) re-validates to the same fingerprint
    again = schema_from_dict(s.model_dump(mode="json", exclude_none=True))
    assert again.fingerprint() == s.fingerprint()
