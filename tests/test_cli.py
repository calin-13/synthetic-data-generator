from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from src.cli import main
from src.designer import design_to_schema, schema_from_sample
from src.schema_builder import load_schema, schema_from_dict

from .conftest import EXAMPLES


@pytest.fixture
def cli(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    runner = CliRunner()

    def invoke(*args, input=None):
        return runner.invoke(main, [str(a) for a in args], input=input, catch_exceptions=False)
    return invoke


def test_full_workflow_with_mock(cli, tmp_path):
    schema = EXAMPLES / "job_postings.json"
    r = cli("--generate", "--schema", schema, "--count", 60, "--format", "csv,parquet", "--mock", "--seed", 7)
    assert r.exit_code == 0, r.output
    out = tmp_path / "output" / "job_postings"
    assert {"job_postings.jsonl", "job_postings.csv", "job_postings.parquet", "manifest.json",
            "quality_report.json"} <= {p.name for p in out.iterdir()}
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["settings"]["seed"] == 7 and manifest["tables"]["job_postings"]["records"] == 60
    assert len(manifest["tables"]["job_postings"]["scenarios"]) == 40

    r = cli("--validate", "--data", out / "job_postings.csv", "--schema", schema, "--report", "report.json")
    assert r.exit_code == 0, r.output
    assert "schema pass rate 100.0%" in r.output
    assert json.loads((tmp_path / "report.json").read_text())["schema"]["pass_rate"] == 1.0

    r = cli("--export", "--format", "parquet", "--output", "data.parquet")  # defaults to the last run
    assert r.exit_code == 0, r.output
    assert (tmp_path / "data.parquet").exists()
    r = cli("--validate", "--data", "data.parquet")  # schema taken from the last run's manifest
    assert r.exit_code == 0 and "schema pass rate 100.0%" in r.output

    # regenerate from the manifest: same schema, seed and scenario plan, no planning call needed
    r = cli("--generate", "--config", out / "manifest.json", "--mock", "--format", "json")
    assert r.exit_code == 0, r.output
    regen = tmp_path / "output" / "job_postings_regen"
    m2 = json.loads((regen / "manifest.json").read_text())
    assert m2["schema_fingerprint"] == manifest["schema_fingerprint"] and m2["settings"]["seed"] == 7
    assert m2["tables"]["job_postings"]["scenarios"] == manifest["tables"]["job_postings"]["scenarios"]


def test_generate_refuses_to_clobber_and_resumes(cli, tmp_path):
    schema = EXAMPLES / "banking_intents.json"
    assert cli("--schema", schema, "-n", 30, "--mock").exit_code == 0
    r = cli("--schema", schema, "-n", 30, "--mock")
    assert r.exit_code == 2 and "--resume" in r.output
    assert cli("--schema", schema, "-n", 50, "--mock", "--resume").exit_code == 0
    lines = (tmp_path / "output" / "banking_intents" / "banking_intents.jsonl").read_text().splitlines()
    assert len(lines) == 50


def test_missing_api_key_is_explained(cli):
    r = cli("--generate", "--schema", EXAMPLES / "job_postings.json", "-n", 5)
    assert r.exit_code == 2 and "ANTHROPIC_API_KEY" in r.output and "--mock" in r.output


def test_check_shows_prompt_and_schema(cli):
    r = cli("--check", "--schema", EXAMPLES / "ecommerce_store.json", "--show-prompt", "--show-schema")
    assert r.exit_code == 0, r.output
    assert "related customers record" in r.output and '"additionalProperties": false' in r.output


def test_bad_schema_error_is_readable(cli, tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"fields": [{"name": "salary", "type": "number", "min": 10, "max": 1}]}))
    r = cli("--check", "--schema", p)
    assert r.exit_code == 2 and "field 'salary': min is greater than max" in r.output


def test_define_wizard(cli, tmp_path):
    answers = "\n".join([
        "jobs", "A job posting", "", "100",            # name, description, context, count
        "title", "string", "Job title", "80", "n", "n",  # field 1: type, desc, max len, nullable, unique
        "salary", "number", "Annual EUR", "30000", "200000", "normal", "70000", "20000", "n",
        "skills", "array", "Required skills", "string", "2", "6", "n",
        "level", "enum", "", "junior,mid,senior", "0.3,0.4,0.3", "n",
        "",                                              # finish fields
        "senior roles pay more", "",                     # rules
        "10",                                            # scenarios
    ]) + "\n"
    r = cli("--define", "--schema", "jobs.json", input=answers)
    assert r.exit_code == 0, r.output
    s = load_schema(tmp_path / "jobs.json")
    t = s.tables[0]
    assert [f.name for f in t.fields] == ["title", "salary", "skills", "level"]
    assert t.field_map["salary"].distribution.type == "normal"
    assert t.field_map["level"].stratify and t.rules == ["senior roles pay more"]


def test_define_from_sample(cli, tmp_path):
    import random
    r = random.Random(0)
    rows = [{"id": i + 1, "plan": r.choice(["free", "pro", "pro", "team"]), "amount": round(r.lognormvariate(3, 0.8), 2),
             "comment": f"private note {i} " + "lorem ipsum " * r.randint(5, 15)} for i in range(120)]
    (tmp_path / "sample.jsonl").write_text("\n".join(json.dumps(x) for x in rows))
    res = cli("--define", "--schema", "accounts.json", "--from-sample", "sample.jsonl")
    assert res.exit_code == 0, res.output
    text = (tmp_path / "accounts.json").read_text()
    assert "private note" not in text
    t = load_schema(tmp_path / "accounts.json").tables[0]
    assert t.field_map["id"].generator == "sequence" and t.field_map["plan"].type == "enum"
    assert t.field_map["amount"].distribution.type == "lognormal"


def test_schema_from_sample_privacy_and_types():
    import random
    r = random.Random(0)
    rows = [{"email": f"user{i}@corp.example", "created": f"2024-0{r.randint(1, 9)}-1{r.randint(0, 9)}",
             "active": r.choice(["true", "false"]), "maybe": "" if i % 4 == 0 else str(r.randint(1, 50))}
            for i in range(200)]
    spec = schema_from_sample(rows, name="accounts")
    f = {x["name"]: x for x in spec["fields"]}
    assert f["created"]["type"] == "date" and f["active"]["type"] == "boolean"
    assert 0.2 < f["maybe"]["null_rate"] < 0.3
    assert "user1@" not in json.dumps(spec)


def test_design_to_schema_roundtrip():
    base = {"description": "", "nullable": False, "null_rate": 0, "unique": False, "values": [], "weights": [],
            "min": "", "max": "", "max_length": 0, "items_type": "", "min_items": 0, "max_items": 0,
            "generator": "", "sequence_format": "", "distribution": "", "dist_a": 0, "dist_b": 0}
    design = {"name": "Loan Applications", "description": "d", "context": "c", "rules": ["r"],
              "scenario_guidance": "g", "edge_case_ratio": 0.1, "fields": [
                  {**base, "name": "app_id", "type": "string", "generator": "sequence", "sequence_format": "APP-{:05d}"},
                  {**base, "name": "income", "type": "number", "min": "0", "max": "500000",
                   "distribution": "lognormal", "dist_a": 55000, "dist_b": 0.6},
                  {**base, "name": "decision", "type": "enum", "values": ["approved", "denied"], "weights": [3, 1]},
                  {**base, "name": "applied", "type": "date", "min": "2024-01-01", "max": "2024-06-30",
                   "distribution": "uniform"}]}
    t = schema_from_dict(design_to_schema(design, 100)).tables[0]
    assert t.name == "loan_applications" and t.field_map["app_id"].format == "APP-{:05d}"
    assert t.field_map["income"].sampled and t.field_map["decision"].stratify


def test_root_entry_script_exists():
    root = Path(__file__).resolve().parent.parent
    assert "from src.cli import main" in (root / "generator.py").read_text()
