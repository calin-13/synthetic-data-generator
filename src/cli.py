"""Command-line interface (Click).

    python generator.py --define   --schema job_postings.json
    python generator.py --generate --schema job_postings.json --count 1000 --format csv
    python generator.py --validate --data output/job_postings/job_postings.csv --schema job_postings.json
    python generator.py --export   --format parquet --output data.parquet
    python generator.py --check    --schema job_postings.json --show-prompt

``--generate`` is the default action when only ``--schema`` is given.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import hashlib
import json
import os
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any

import click

from . import __version__
from .designer import interactive_define, schema_from_description, schema_from_sample
from .exporters import FORMATS, export_dataset, format_from_path, read_table, write_table
from .generator import GenerationError, Stats, TableGenerator, generate_dataset, make_client
from .prompts import user_prompt
from .schema_builder import (DatasetSchema, SchemaError, batch_schema, complexity, full_record_schema,
                             load_schema, save_schema, schema_from_dict)
from .storage import Store, read_last_run, write_last_run
from .validator import format_report, quality_report


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_dotenv(path: str | Path = ".env") -> None:
    """Minimal .env support (KEY=VALUE lines); never overrides real environment variables."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip().removeprefix("export ").strip(), v.strip().strip('"').strip("'")
        if k and v and k not in os.environ:
            os.environ[k] = v


def _fail(msg: str, code: int = 2) -> None:
    click.secho(f"error: {msg}", fg="red", err=True)
    sys.exit(code)


def _info(msg: str, quiet: bool = False, **style: Any) -> None:
    if not quiet:
        click.secho(msg, err=True, **style)


class Progress:
    def __init__(self, quiet: bool = False):
        self.quiet = quiet
        self.tty = sys.stderr.isatty()
        self._last = 0.0
        self._table = ""

    def __call__(self, table: str, done: int, target: int, st: Stats) -> None:
        if self.quiet:
            return
        now = time.time()
        final = done >= target or st.finished
        if not final and table == self._table and now - self._last < (0.3 if self.tty else 5.0):
            return
        self._last, self._table = now, table
        width = 30
        filled = int(width * done / target) if target else width
        line = (f"{table:<20} [{'#' * filled}{'-' * (width - filled)}] {done}/{target}  "
                f"{st.records_per_second:5.1f} rec/s  accept {st.acceptance_rate:4.0%}  req {st.requests}")
        if self.tty:
            click.echo("\r" + line, nl=final, err=True)
        else:
            click.echo(line, err=True)

    def log(self, msg: str) -> None:
        if not self.quiet:
            click.echo(("\n" if self.tty else "") + msg, err=True)


def _parse_counts(schema: DatasetSchema, count: tuple[str, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in count:
        if "=" in item:
            k, v = item.split("=", 1)
            schema.table(k)
            counts[k] = int(v)
        elif len(schema.tables) == 1:
            counts[schema.tables[0].name] = int(item)
        else:
            counts.update({t.name: int(item) for t in schema.tables})
    for k, v in counts.items():
        if v < 1:
            raise click.BadParameter(f"count for '{k}' must be positive")
    return counts


def _formats(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return default
    out = [f.strip().lower() for f in value.split(",") if f.strip()]
    bad = [f for f in out if f not in FORMATS]
    if bad:
        raise click.BadParameter(f"unknown format(s) {bad}; choose from {', '.join(FORMATS)}", param_hint="--format")
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sdk_version() -> str:
    try:
        return metadata.version("anthropic")
    except metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


# ---------------------------------------------------------------------------
# actions
# ---------------------------------------------------------------------------

def do_define(schema_path: str | None, from_description: str | None, from_sample: str | None,
              about: str | None, count: tuple[str, ...], model: str | None, mock: bool) -> None:
    if not schema_path:
        _fail("--define needs --schema PATH (where to save the schema), e.g. --schema job_postings.json")
    path = Path(schema_path)
    if path.exists() and not click.confirm(f"{path} exists. Overwrite?", default=False, err=True):
        sys.exit(1)
    n = int(count[0]) if count and count[0].isdigit() else None
    if from_sample:
        rows = read_table(from_sample, limit=5000)
        raw = schema_from_sample(rows, name=path.stem, about=about or "", count=n)
        _info(f"inferred {len(raw['fields'])} fields locally from {len(rows)} rows of {from_sample} "
              "(no data left this machine; no free-text values were copied)")
        _info("fill in the TODO descriptions: they are what makes generated data realistic", fg="yellow")
    elif from_description:
        if mock:
            _fail("--from-description needs the real API (drop --mock)")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            _fail("ANTHROPIC_API_KEY is not set (export it or put it in .env)")
        _info("asking Claude to design the schema...")
        raw = asyncio.run(schema_from_description(make_client(), from_description, count=n or 500,
                                                  model=model or "claude-opus-5-5"))
    else:
        raw = interactive_define(default_name=path.stem)
    save_schema(raw, path)
    ds = schema_from_dict(raw, default_name=path.stem)
    t = ds.tables[0]
    _info(f"saved {path}: table '{t.name}', {len(t.fields)} fields, {t.count} records", fg="green")
    _info(f"next: python generator.py --generate --schema {path} --count {t.count} --format csv")


def do_check(schema: DatasetSchema, show_prompt: bool, show_schema: bool) -> None:
    order = [t.name for t in schema.ordered_tables()]
    click.echo(f"schema OK: {len(schema.tables)} table(s), order {order}, model {schema.settings.model}, "
               f"mode {schema.settings.mode}")
    for t in schema.tables:
        cx = complexity(batch_schema(t))
        click.echo(f"\n[{t.name}] {t.count} records, batch {t.batch_size} (~{-(-t.count // t.batch_size)} requests)")
        click.echo(f"  fields: {len(t.fields)} ({len(t.llm_fields)} by Claude, {len(t.fields) - len(t.llm_fields)} by code)")
        if t.stratified_fields:
            click.echo(f"  target distributions: {', '.join(t.stratified_fields)}")
        if t.sampled_fields:
            click.echo(f"  sampled in code: {', '.join(t.sampled_fields)}")
        if t.ref_fields:
            click.echo("  references: " + ", ".join(f"{k} -> {f.ref}" for k, f in t.ref_fields.items()))
        click.echo(f"  API schema: {cx['properties']} properties, {cx['unions']} nullable, depth {cx['max_depth']}")
        for w in cx["warnings"]:
            click.secho(f"  ! {w}", fg="yellow")
        if show_schema:
            click.echo("\n--- structured-output schema ---\n" + json.dumps(batch_schema(t), indent=2))
            click.echo("\n--- full record JSON Schema ---\n" + json.dumps(full_record_schema(t), indent=2, default=str))
        if show_prompt:
            parents = {}
            for f in t.ref_fields.values():
                pt = schema.table(f.ref_table or "")
                parents[pt.name] = [{k: f"<{pt.name}.{k}>" for k in pt.field_map}]
            gen = TableGenerator(t, schema.settings, client=None, parents=parents)
            if t.diversity.scenarios:
                gen.sampler.set_scenarios(["<scenario planned by Claude>"])
            slots = [gen.sampler.make_slot(i + 1) for i in range(min(3, t.batch_size))]
            click.echo("\n--- system prompt (cached) ---\n" + gen.system[0]["text"])
            click.echo("\n--- user prompt (3 slots) ---\n" + user_prompt(slots, t))


def do_generate(schema: DatasetSchema, schema_file: str | None, counts: dict[str, int], out_dir: str,
                formats: list[str], only: list[str], batch_api: bool, poll_interval: float, resume: bool,
                overwrite: bool, mock: bool, quiet: bool, scenarios: dict[str, list[str]] | None) -> int:
    store = Store(out_dir)
    if not resume and not overwrite:
        existing = [t.name for t in schema.tables if (not only or t.name in only) and store.load(t.name)]
        if existing:
            _fail(f"{out_dir} already has data for {existing}; use --resume to continue or --overwrite")
    if mock:
        from .mock_client import MockAsyncAnthropic
        client: Any = MockAsyncAnthropic(latency=0.05, obey_checks=True)
        _info("MOCK MODE: offline placeholder values, for trying the pipeline only", quiet, fg="yellow")
    else:
        if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("ANTHROPIC_AUTH_TOKEN"):
            _fail("ANTHROPIC_API_KEY is not set (export it, put it in .env, or try --mock)")
        client = make_client()

    prog = Progress(quiet)
    started = time.time()
    try:
        gens = asyncio.run(generate_dataset(
            schema, out_dir=out_dir, client=client, only=only or None, counts=counts, batch_api=batch_api,
            resume=resume, on_progress=prog, log=prog.log, poll_interval=poll_interval, scenarios=scenarios))
    except GenerationError as e:
        hint = ("\n(the offline mock writes random placeholders, so it can't satisfy arithmetic checks or regex "
                "patterns; run without --mock for real data)") if mock else ""
        _fail(f"{e}{hint}\npartial results kept in {out_dir}; rerun with --resume", code=1)
    except KeyboardInterrupt:
        _fail(f"interrupted; progress saved in {out_dir}. Rerun with --resume.", code=130)
    elapsed = time.time() - started

    specs = {t.name: t for t in schema.tables}
    tables = {name: g.records for name, g in gens.items()}
    if "sqlite" in formats:  # include untouched parent tables so foreign keys resolve
        for t in schema.tables:
            if t.name not in tables and store.load(t.name):
                tables[t.name] = store.load(t.name)
    written = export_dataset(tables, specs, out_dir, [f for f in formats if f != "jsonl"])

    reports = {}
    for name, g in gens.items():
        rep = quality_report(g.records, specs[name])
        reports[name] = rep
        st = g.stats
        _info(f"\n{name}: {len(g.records)} records | score {rep['score']}/100 ({rep['grade']}) | "
              f"schema pass {rep['schema']['pass_rate']:.0%} | acceptance {st.acceptance_rate:.0%} | "
              f"tokens in {st.input_tokens:,} (cached {st.cache_read_tokens:,}) out {st.output_tokens:,}", quiet)
        if st.rejected:
            _info("  regenerated after: " + ", ".join(f"{k} x{v}" for k, v in st.rejected.most_common(4)), quiet)
        for i in rep["issues"][:5]:
            _info(f"  [{i['severity']}] {i['column']}: {i['message']}", quiet,
                  fg="red" if i["severity"] == "high" else "yellow")
    Path(out_dir, "quality_report.json").write_text(json.dumps(reports, indent=2, default=str), encoding="utf-8")

    # reproducibility manifest
    manifest = {
        "tool": "synthetic-data-generator", "version": __version__, "anthropic_sdk": _sdk_version(),
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "command": [Path(sys.argv[0]).name, *sys.argv[1:]], "mock": mock, "batch_api": batch_api,
        "schema_file": schema_file, "schema_fingerprint": schema.fingerprint(),
        "settings": schema.settings.model_dump(mode="json"),
        "counts": {n: g.target for n, g in gens.items()},
        "schema": schema.model_dump(mode="json", exclude_none=True),
        "tables": {n: {"records": len(g.records), "data_file": str(store.data_path(n)),
                       "sha256": _sha256(store.data_path(n)) if store.data_path(n).exists() else None,  # type: ignore[union-attr]
                       "scenarios": g.scenarios, "stats": g.stats.to_dict()} for n, g in gens.items()},
        "elapsed_seconds": round(elapsed, 2),
    }
    Path(out_dir, "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    write_last_run({"output": out_dir, "schema": schema_file, "manifest": str(Path(out_dir, "manifest.json")),
                    "tables": {n: str(store.data_path(n)) for n in gens}})

    total = sum(len(g.records) for g in gens.values())
    _info(f"\ngenerated {total} records in {elapsed:.1f}s ({total / elapsed if elapsed else 0:.1f} rec/s) -> {out_dir}",
          quiet, fg="green", bold=True)
    for p in written:
        _info(f"  wrote {p}", quiet)
    _info(f"  manifest (seed {schema.settings.seed}): {Path(out_dir, 'manifest.json')}", quiet)
    short = [n for n, g in gens.items() if len(g.records) < g.target]
    return 1 if short else 0


def _schema_for_data(schema_path: str | None) -> DatasetSchema | None:
    if schema_path:
        return load_schema(schema_path)
    last = read_last_run()
    if last and last.get("manifest") and Path(last["manifest"]).exists():
        m = json.loads(Path(last["manifest"]).read_text(encoding="utf-8"))
        return schema_from_dict(m["schema"])
    return None


def _pick_table(schema: DatasetSchema | None, data: Path, table: tuple[str, ...]) -> Any:
    if schema is None:
        return None
    if table:
        return schema.table(table[0])
    names = [t.name for t in schema.tables]
    if data.stem in names:
        return schema.table(data.stem)
    if len(names) == 1:
        return schema.tables[0]
    _fail(f"schema has tables {names}; pass --table")


def do_validate(data: str | None, schema_path: str | None, table: tuple[str, ...], report_path: str | None,
                as_json: bool) -> int:
    if not data:
        last = read_last_run()
        if not last:
            _fail("--validate needs --data FILE (no previous run found)")
        data = next(iter(last["tables"].values()))  # type: ignore[index]
        _info(f"validating last run: {data}")
    path = Path(data)  # type: ignore[arg-type]
    schema = _schema_for_data(schema_path)
    t = _pick_table(schema, path, table)
    rows = read_table(path, sqlite_table=table[0] if table else None)
    rep = quality_report(rows, t)
    if as_json:
        click.echo(json.dumps(rep, indent=2, default=str))
    else:
        click.echo(format_report(rep, path.name))
        if t is None:
            click.secho("(no schema given: structure/constraint checks skipped; pass --schema)", fg="yellow")
    if report_path:
        Path(report_path).write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
        _info(f"wrote {report_path}")
    failed = (rep.get("schema") and rep["schema"]["pass_rate"] < 1) or rep.get("duplicate_rows")
    return 1 if failed else 0


def do_export(data: str | None, schema_path: str | None, fmt_value: str | None, output: str | None,
              table: tuple[str, ...]) -> None:
    schema = _schema_for_data(schema_path)
    if data:
        sources = {Path(data).stem: Path(data)}
    else:
        last = read_last_run()
        if not last:
            _fail("--export needs --data FILE (no previous run found)")
        sources = {n: Path(p) for n, p in last["tables"].items()}  # type: ignore[index]
        if table:
            sources = {n: p for n, p in sources.items() if n in table}
    tables = {n: read_table(p) for n, p in sources.items()}
    specs = {t.name: t for t in schema.tables} if schema else {}
    single_file = output and Path(output).suffix and len(tables) == 1
    if single_file:
        fmt = _formats(fmt_value, [format_from_path(output)])[0]  # type: ignore[arg-type]
        (name, rows), = tables.items()
        p = write_table(rows, output, fmt, specs.get(name))  # type: ignore[arg-type]
        click.echo(f"wrote {p} ({len(rows)} rows)")
        return
    if not fmt_value:
        _fail("--export needs --format (or an --output file name with an extension)")
    out_dir = output or str(next(iter(sources.values())).parent)
    for p in export_dataset(tables, specs, out_dir, _formats(fmt_value, [])):
        click.echo(f"wrote {p}")


# ---------------------------------------------------------------------------
# command
# ---------------------------------------------------------------------------

@click.command(context_settings={"help_option_names": ["-h", "--help"], "max_content_width": 110})
@click.option("--define", "action", flag_value="define", help="Create a schema (wizard, from a description, or from a sample).")
@click.option("--generate", "action", flag_value="generate", help="Generate data from a schema (default action).")
@click.option("--validate", "action", flag_value="validate", help="Validate a dataset and print a quality report.")
@click.option("--export", "action", flag_value="export", help="Export generated data to another format.")
@click.option("--check", "action", flag_value="check", help="Validate a schema and preview prompts (no API calls).")
@click.option("--schema", "schema_path", type=click.Path(dir_okay=False), help="Schema file (.json or .yaml).")
@click.option("-n", "--count", multiple=True, help="Records to generate (or table=N for multi-table schemas).")
@click.option("--format", "fmt", help=f"Output format(s), comma-separated: {', '.join(FORMATS)}.")
@click.option("-o", "--output", help="Generate: output directory. Export: file or directory.")
@click.option("--data", type=click.Path(exists=True, dir_okay=False), help="Dataset file to validate/export.")
@click.option("--table", multiple=True, help="Restrict to this table (repeatable).")
@click.option("--model", help="Claude model, e.g. claude-sonnet-5-5, claude-haiku-4-5-20251001.")
@click.option("--concurrency", type=int, help="Requests in flight (default 16).")
@click.option("--batch-size", type=int, help="Records per request.")
@click.option("--max-tokens", type=int)
@click.option("--temperature", type=float)
@click.option("--seed", type=int, help="Seed for code-side sampling (recorded in the manifest).")
@click.option("--mode", type=click.Choice(["json", "tool"]), help="Structured outputs via JSON outputs or strict tool use.")
@click.option("--batch-api", is_flag=True, help="Use the Message Batches API (async, ~50% cheaper).")
@click.option("--poll-interval", type=float, default=30.0, show_default=True)
@click.option("--resume", is_flag=True, help="Continue an interrupted run.")
@click.option("--overwrite", is_flag=True, help="Replace existing output.")
@click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False),
              help="manifest.json of an earlier run: regenerate with its schema, seed, settings and scenario plan.")
@click.option("--mock", is_flag=True, help="Offline mock client: no API key, placeholder values (pipeline demo).")
@click.option("--from-description", help="Define: have Claude design the schema from this description.")
@click.option("--from-sample", type=click.Path(exists=True, dir_okay=False), help="Define: infer the schema locally from a sample file.")
@click.option("--about", help="Define: one-line description used with --from-sample.")
@click.option("--report", "report_path", type=click.Path(dir_okay=False), help="Validate: write the JSON report here.")
@click.option("--json", "as_json", is_flag=True, help="Validate: print the report as JSON.")
@click.option("--show-prompt", is_flag=True, help="Check: print the system prompt and a sample request.")
@click.option("--show-schema", is_flag=True, help="Check: print the structured-output and full JSON Schemas.")
@click.option("-q", "--quiet", is_flag=True)
@click.version_option(__version__, prog_name="synthetic-data-generator")
@click.pass_context
def main(ctx: click.Context, action: str | None, schema_path: str | None, count: tuple[str, ...], fmt: str | None,
         output: str | None, data: str | None, table: tuple[str, ...], model: str | None, concurrency: int | None,
         batch_size: int | None, max_tokens: int | None, temperature: float | None, seed: int | None,
         mode: str | None, batch_api: bool, poll_interval: float, resume: bool, overwrite: bool,
         config_path: str | None, mock: bool, from_description: str | None, from_sample: str | None,
         about: str | None, report_path: str | None, as_json: bool, show_prompt: bool, show_schema: bool,
         quiet: bool) -> None:
    """Synthetic Data Generator: realistic datasets from a schema, powered by Claude structured outputs.

    \b
    Workflow:
      python generator.py --define   --schema job_postings.json
      python generator.py --generate --schema examples/job_postings.json --count 1000 --format csv
      python generator.py --validate --data output/job_postings/job_postings.csv --schema examples/job_postings.json
      python generator.py --export   --format parquet --output data.parquet
    """
    load_dotenv()
    action = action or ("generate" if schema_path or config_path else None)
    if action is None:
        click.echo(ctx.get_help())
        ctx.exit(0)
    try:
        if action == "define":
            do_define(schema_path, from_description, from_sample, about, count, model, mock)
            return
        if action == "validate":
            ctx.exit(do_validate(data, schema_path, table, report_path, as_json))
        if action == "export":
            do_export(data, schema_path, fmt, output, table)
            return

        scenarios = None
        if config_path:
            m = json.loads(Path(config_path).read_text(encoding="utf-8"))
            schema = schema_from_dict(m["schema"])
            scenarios = {n: t.get("scenarios") or [] for n, t in m.get("tables", {}).items()}
            counts = dict(m.get("counts", {}))
            schema_file = m.get("schema_file")
            _info(f"regenerating from {config_path} (seed {schema.settings.seed}, schema {m.get('schema_fingerprint')})",
                  quiet)
        else:
            if not schema_path:
                _fail("--schema is required")
            schema = load_schema(schema_path)  # type: ignore[arg-type]
            counts, schema_file = {}, schema_path
        s = schema.settings
        for attr, val in (("model", model), ("concurrency", concurrency), ("max_tokens", max_tokens),
                          ("temperature", temperature), ("seed", seed), ("mode", mode)):
            if val is not None:
                setattr(s, attr, val)
        if batch_size:
            for t in schema.tables:
                t.batch_size = batch_size
        counts.update(_parse_counts(schema, count))

        if action == "check":
            do_check(schema, show_prompt, show_schema)
            return
        default_name = Path(schema_file).stem if schema_file else schema.tables[0].name
        out_dir = output or str(Path("output") / (default_name + ("_regen" if config_path else "")))
        ctx.exit(do_generate(schema, schema_file, counts, out_dir, _formats(fmt, ["csv"]), list(table),
                             batch_api, poll_interval, resume, overwrite, mock, quiet, scenarios))
    except (SchemaError, FileNotFoundError, ValueError) as e:
        _fail(str(e))


if __name__ == "__main__":  # pragma: no cover
    main()
