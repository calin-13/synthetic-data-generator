from __future__ import annotations

import random

import pytest

from src.schema_builder import schema_from_dict
from src.validator import CheckRunner, RecordError, format_report, quality_report, validate_record

from .conftest import run


def _issues(rep, severity=None):
    return [i["message"] for i in rep["issues"] if severity is None or i["severity"] == severity]


def test_validate_record_normalizes_and_rejects(basic):
    t = schema_from_dict(basic()).tables[0]
    rec = {"id": "P0001", "uid": "2F1E4B5A-3C7D-4E8F-9A0B-1C2D3E4F5A6B", "name": "Ann", "email": "a@b.io",
           "tier": "Silver", "age": "33", "joined": "2024-02-29", "score": "0.5", "bio": "x",
           "tags": ["a"], "address": {"city": "Iasi", "zip": "700001"}}
    out = validate_record(t, rec)
    assert out["tier"] == "silver" and out["age"] == 33 and out["score"] == 0.5
    assert out["uid"] == "2f1e4b5a-3c7d-4e8f-9a0b-1c2d3e4f5a6b"
    with pytest.raises(RecordError) as e:
        validate_record(t, dict(rec, tier="platinum"))
    assert e.value.category == "tier: not one of the allowed values"
    with pytest.raises(RecordError) as e:
        validate_record(t, dict(rec, tags=["a", "b", "c", "d"]))
    assert e.value.path == "tags"
    with pytest.raises(RecordError) as e:
        validate_record(t, dict(rec, address={"city": "Iasi"}))
    assert e.value.category.startswith("address.zip")


def test_checks_runner_dates_and_parent():
    s = schema_from_dict({"name": "t", "fields": [
        {"name": "start", "type": "date"}, {"name": "end", "type": "date"}, {"name": "x"}],
        "checks": ["end >= start", "parent['p']['k'] == 1"]})
    c = CheckRunner(s.tables[0])
    rec = {"start": "2025-01-02", "end": "2025-01-01", "x": "a"}
    assert c.failures(rec) == ["end >= start"]  # parent check skipped without context
    assert c.failures({**rec, "end": "2025-02-01"}, parent={"p": {"k": 2}}) == ["parent['p']['k'] == 1"]


def test_clean_generated_data_scores_high(basic, mock):
    s = schema_from_dict(basic(count=80))
    rows = run(s, mock(seed=1))["people"].records
    rep = quality_report(rows, s.tables[0])
    assert rep["schema"]["pass_rate"] == 1.0
    assert rep["duplicate_rows"] == 0
    assert rep["columns"]["tier"]["tvd_from_target"] < 0.05
    assert rep["score"] >= 80, _issues(rep)
    assert "score" in format_report(rep)


def test_detects_violations_duplicates_and_label_skew():
    s = schema_from_dict({"name": "t", "fields": [
        {"name": "label", "values": ["a", "b"], "weights": [0.5, 0.5]},
        {"name": "n", "type": "integer", "min": 0, "max": 10}]})
    rows = [{"label": "a", "n": i % 10} for i in range(45)] + [{"label": "b", "n": 3}] * 5 + [{"label": "a", "n": 99}]
    rep = quality_report(rows, s.tables[0])
    assert rep["schema"]["pass_rate"] < 1 and "n: Input should be less than or equal to 10" in rep["schema"]["violations"]
    assert rep["duplicate_rows"] > 0
    assert any("TVD" in m for m in _issues(rep, "high"))
    assert rep["grade"] in ("C", "D", "F")


def test_detects_generation_artefacts():
    r = random.Random(0)
    rows = []
    for i in range(60):
        rows.append({
            "salary": 40000 + i * 1000,                                   # trends with row order + round numbers
            "text": f"Dear team, I am writing about issue {r.randint(1, 10**6)} today.",   # templated opening
            "latency_ms": r.gauss(100, 5) if i != 7 else 5000,           # one extreme outlier
            "constant": "same",
        })
    rep = quality_report(rows)
    msgs = " | ".join(_issues(rep))
    assert "trend with row order" in msgs
    assert "round-number" in msgs
    assert "templated text" in msgs
    assert "outlier" in msgs
    assert "constant column" in msgs


def test_csv_strings_are_typed_before_profiling(basic, mock, tmp_path):
    from src.exporters import read_table, write_table
    s = schema_from_dict(basic(count=30))
    rows = run(s, mock(seed=2))["people"].records
    p = write_table(rows, tmp_path / "people.csv")
    back = read_table(p)
    rep = quality_report(back, s.tables[0])
    assert rep["schema"]["pass_rate"] == 1.0
    assert "mean" in rep["columns"]["age"]  # numeric stats, not string lengths
    assert rep["columns"]["tags"]["items"]["max"] <= 3
