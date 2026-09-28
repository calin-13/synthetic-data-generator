# Synthetic Data Generator

Realistic, schema-valid synthetic datasets generated with Claude's **structured outputs**. You define
the fields, types and constraints of your data; the generator gives you production-ready CSV, JSON,
Parquet or SQLite files, plus a quality report. There's no manual labeling and no real personal data involved.

```bash
python generator.py --define   --schema job_postings.json                                   # 1. define
python generator.py --generate --schema examples/job_postings.json --count 1000 --format csv # 2. generate
python generator.py --validate --data output/job_postings/job_postings.csv --schema examples/job_postings.json
python generator.py --export   --format parquet --output data.parquet                       # 4. export
```

Typical uses:

- **ML teams:** bootstrap labelled training sets.
- **QA:** realistic test fixtures.
- **Product:** populated demo environments.
- **Research:** privacy-safe corpora.
- **Data science:** diverse validation scenarios.

---

## Open it in PyCharm

1. **File → Open…** and select the `synthetic-data-generator` folder.
2. **Settings → Project → Python Interpreter → Add Interpreter → Virtualenv** (Python 3.10+). When
   PyCharm offers to *install requirements from requirements.txt*, accept. Otherwise run
   `pip install -r requirements.txt` in the built-in terminal.
3. Copy `.env.example` to `.env` and paste your `ANTHROPIC_API_KEY`. The CLI loads `.env` automatically,
   and the file is git-ignored.
4. Pick a ready-made run configuration from the toolbar dropdown:

| Run configuration | What it does |
|---|---|
| `2 Generate job postings (mock, no API key)` | Runs the full pipeline offline and writes CSV, Parquet and SQLite in seconds |
| `2 Generate job postings (Claude API)` | Generates real data (needs `.env`) |
| `1 Define schema (wizard)` | Interactive schema builder |
| `3 Validate job postings` | Quality report plus a JSON report file |
| `4 Export last run to Parquet` | Re-exports the latest output |
| `Check schema and preview prompts` | Shows exactly what is sent to Claude, without calling the API |
| `Benchmark 1000 records (simulated)` | Throughput check |
| `Tests` | Runs the 62 pytest tests, which are all offline |

If PyCharm reports "module not specified" (this happens when the folder was renamed), open **Edit
Configurations** and select your module. That's all.

---

## 1. Schema definition

A schema is JSON (YAML also works). The format from the brief is accepted as-is:

```json
{
  "fields": [
    {"name": "job_title", "type": "string", "description": "Job position title"},
    {"name": "salary", "type": "number", "min": 30000, "max": 200000},
    {"name": "skills", "type": "array", "items": "string"}
  ],
  "count": 500
}
```

Pydantic validates it before any API call, and errors point at the exact problem, such as
`field 'salary': min is greater than max`. The format also carries the *intent* that makes the data realistic:

```json
{
  "name": "job_postings",
  "description": "Job postings from a European tech job board.",
  "context": "Salaries depend on seniority, then location and department ...",
  "count": 500,
  "fields": [
    {"name": "job_id", "type": "string", "generator": "sequence", "format": "JOB-{:06d}"},
    {"name": "seniority", "type": "enum", "values": ["junior", "mid", "senior"], "weights": [0.3, 0.4, 0.3]},
    {"name": "salary", "type": "number", "min": 30000, "max": 200000, "decimals": 0},
    {"name": "location", "type": "object", "properties": [
      {"name": "city", "type": "string"},
      {"name": "remote_policy", "type": "enum", "values": ["onsite", "hybrid", "remote"]}]},
    {"name": "applicants", "type": "integer", "min": 0, "max": 3000,
     "distribution": {"type": "lognormal", "median": 45, "sigma": 1.0}}
  ],
  "rules":  ["Salary bands by seniority: junior 35-55k, mid 50-80k, senior 70-115k ..."],
  "checks": ["salary >= {'junior': 30000, 'mid': 40000, 'senior': 55000}[seniority]"],
  "diversity": {"scenarios": 40, "edge_case_ratio": 0.05, "avoid_repeats": ["company"]}
}
```

**Types:** `string`, `text` (long prose), `integer`, `number`, `boolean`, `enum`, `date`, `datetime`,
`time`, `email`, `url`, `uuid`, `array` (of any type, including objects), and `object` (nested, to any depth).

| Field key | Meaning |
|---|---|
| `description` | What the field holds and how it relates to others. This matters most for realism. |
| `min` / `max`, `decimals` | Numeric and date bounds, and rounding |
| `min_length` / `max_length` / `pattern` | String limits and a full-match regex |
| `values` + `weights` | Enum values and their target distribution. A quota planner enforces the weights. |
| `items`, `min_items` / `max_items` | Array element type and length |
| `properties` | Fields of a nested object |
| `nullable` / `null_rate` | Allow null, or set an exact share of nulls decided in code |
| `unique` | No duplicates across the dataset |
| `distribution` | Sample in code: `uniform`, `normal {mean,std}`, `lognormal {median,sigma}`, `exponential {mean}` |
| `generator` | `uuid` or `sequence` (with `format` such as `"ORD-{:07d}"`). Assigned in code at zero token cost. |
| `ref`, `ref_context`, `ref_skew` | Foreign key to another table in a multi-table schema |

Table-level keys:

- `rules`: plain English, for Claude.
- `checks`: Python expressions that are enforced. In child tables, `parent[...]` gives the referenced row.
- `examples`: seed records.
- `diversity`: scenario planning, edge cases and avoid-lists.
- `strict_distribution`: exact label counts.
- `language`.

Settings, top-level or under `settings`:

- `model`
- `concurrency`
- `max_tokens`
- `temperature`
- `seed`
- `mode`: `json` for JSON outputs, or `tool` for strict tool use

**Three ways to create a schema** (`--define`):

- **Interactive wizard:** `python generator.py --define --schema my.json`
- **Claude designs it:** `--from-description "loan applications for a UK lender"`. The design
  comes back as structured output, is checked against the real schema models, and Claude repairs it if
  it fails.
- **Mirror real data you can't share:** `--from-sample customers.csv`. This runs **locally**. Only
  types, category labels and aggregate statistics go into the schema. No free text or identifiers are
  copied, and nothing is uploaded.

## 2. Intelligent data generator

```
schema ─► plan scenarios ─► allocate slots ─► structured-output requests ─► validate & repair ─► checkpoint
           (Claude)        (quotas, sampled     (concurrent, JSON outputs    (Pydantic model,     (JSONL,
                            values, nulls,       or strict tool use; or       checks, dedupe,      resumable)
                            FKs, edge cases)     one Message Batch)           salvage truncation)
                                 ▲                                                   │
                                 └── quotas re-target, batch size adapts, rejection reasons fed back ─┘
```

The work is split between Claude and ordinary code, and each does what it's good at:

| Concern | Who | How |
|---|---|---|
| **Shape** (valid JSON, types, required keys) | Claude structured outputs | Each request carries a JSON Schema compiled from your schema (`output_config.format`, or a `strict: true` tool with forced `tool_choice` in `--mode tool`). Constrained decoding means responses can't be malformed. |
| **Constraints the grammar can't express** (ranges, lengths, regex, date bounds) | Pydantic | Written into field descriptions so Claude sees them, then enforced on every record. Failures are regenerated. |
| **Distributions** (label balance, numeric spread, missingness) | Code (seeded) | LLMs produce poorly calibrated distributions. Sampled values are handed to Claude as fixed facts per record *slot*, and Claude writes a coherent record around them. |
| **Logical consistency and domain knowledge** (salary matches seniority, the text matches the numbers) | Claude | A cached system prompt with the description, domain context, field guide, rules, hard checks and a quality bar. |
| **Diversity** | Both | Claude first plans N distinct scenarios that seed the slots. Avoid-lists steer away from values already used, and a share of slots are flagged as edge cases. |
| **Relations** | Code + Claude | Foreign keys are assigned in code, optionally Zipf-skewed. Each child record gets its parent's context. |

The loop corrects itself as it runs:

- Under-filled labels get more slots.
- Batch size shrinks when output runs long or is truncated. Truncated JSON is salvaged.
- The top rejection reasons are added to later prompts, for example "salary: Input should be ≥ 30000"
  or a failing check.
- The last round is over-provisioned by the expected rejection rate, so the tail doesn't need extra round trips.
- A refusal or a failed request only costs that batch.

Structured outputs are generally available: no beta header is needed, and the tests confirm this
against the real SDK.

**Scale:**

- `--concurrency` sets how many requests are in flight. The SDK retries 429/5xx with backoff.
- `--batch-api` uses the Message Batches API: asynchronous, about 50% cheaper, with automatic top-up rounds.
- The shared system prompt uses prompt caching.
- `--resume` continues any interrupted run. Records are appended as they're accepted, and a torn last
  line is repaired.

## 3. Quality validator

`--validate` works on any CSV, JSON, JSONL, Parquet or SQLite file, whether it was generated or not. It
reports a 0-100 score (A-F) and a list of issues, each marked high, medium or low:

| Check | Detects |
|---|---|
| Schema validation (Pydantic model + checks) | Type errors, out-of-range values, bad formats, missing fields, failed cross-field checks. Reports the pass rate and the top violation reasons. |
| Duplicates | Exact duplicate rows, near-duplicate text, duplicates in `unique` fields |
| Distribution | Label distribution vs. target (total variation distance), normalized entropy, unused enum values, dominant values, constant columns, null rate vs. target |
| **Patterns** (the usual signs of generated data) | Values that trend with row order (Spearman), round-number heaping, templated text openings, low lexical diversity (distinct-2), sentences repeated across rows |
| Anomalies | Numeric outliers beyond 3×IQR, with examples |

Every generation run also writes `quality_report.json`. `--validate` exits with code 1 when there are
schema violations or duplicates, so you can use it as a CI gate.

## 4. Export, storage and reproducibility

- **Formats:**
  - `csv`: nested objects become dotted columns, and arrays become JSON strings.
  - `json`, `jsonl`
  - `parquet`: nested data is kept as structs and lists.
  - `sqlite`: one table per dataset table, with primary and **foreign keys**.
- `--export` defaults to the most recent run. The format comes from `--format` or from the `--output` file extension.
- **`manifest.json`** in every output folder records:
  - the full normalized schema and its fingerprint
  - the settings, including the **seed** (a random one is always chosen and recorded if you give none)
  - the model and SDK version
  - the scenario plan
  - per-table stats and a SHA-256 of each data file

  `--generate --config output/x/manifest.json` regenerates with the same schema, seed, settings and
  scenario plan. Everything decided in code (label quotas, sampled numbers and dates, nulls, foreign keys)
  follows the same seed. The LLM-written text is realistic but not byte-identical between runs.

## 5. CLI

`python generator.py --help` shows every option. The main ones:

| Option | Purpose |
|---|---|
| `--count N` or `-n table=N` | Records per table |
| `--format csv,parquet,sqlite` | Export formats |
| `-o DIR` | Output directory (default `output/<schema>`) |
| `--model` | e.g. `claude-sonnet-5-5` (default), `claude-haiku-4-5-20251001` (fastest and cheapest), `claude-opus-5-5` (hardest domains) |
| `--concurrency`, `--batch-size`, `--max-tokens`, `--temperature`, `--seed` | Tuning |
| `--mode json/tool` | JSON outputs or strict tool use |
| `--batch-api` | Message Batches mode |
| `--resume`, `--overwrite` | Continue or replace an existing output |
| `--config manifest.json` | Regenerate a previous run |
| `--mock` | Offline placeholder mode, for trying the pipeline without a key |
| `--check --show-prompt --show-schema` | Validate the schema and show exactly what is sent to Claude |

`pip install -e .` also installs the same CLI as a `synthgen` command.

**Python API:**

```python
from src import generate, load_schema, quality_report, format_report

rows = generate("examples/job_postings.json", count=200, out_dir="output/jobs")["job_postings"]
print(format_report(quality_report(rows, load_schema("examples/job_postings.json").tables[0])))
```

## Examples

| File | Use case | Shows |
|---|---|---|
| `job_postings.json` | QA / demo | Salary consistent with seniority, location and department (rules + checks), nested location, skills array |
| `product_catalog.json` | Demo / e-commerce | Lognormal prices, nested dimensions, array of variant objects, `null_rate` |
| `customer_profiles.json` | ML (churn) | Label distribution, justified labels, country-consistent addresses, code-sampled ages |
| `banking_intents.json` | ML (NLP) | Balanced intent labels, 15% edge cases, scenario planning |
| `support_tickets.json` | QA | Long-tail response times, conditional nulls, cross-field checks |
| `clinical_notes.json` | Research | Privacy-safe SOAP notes consistent with structured vitals, ICD-10 regex |
| `ecommerce_store.json` | Demo DB | 3 related tables, FK context, skewed popularity, checks against the parent row, SQLite export |

The mock can't satisfy arithmetic checks or regex patterns with random placeholders, so it can't run
`ecommerce_store` or `clinical_notes`. Those need Claude. That is rather the point of the project.

## Success metrics

| Target | Status |
|---|---|
| 1000+ records in < 30 s | It depends on record size and your rate-limit tier. Wall time ≈ ⌈requests ÷ concurrency⌉ × (time to first token + batch output ÷ tokens/s). Simulated at 150 tok/s per request with concurrency 100 and batch 5: 1000 customer profiles took **12.7 s** and 1000 job postings **27.8 s**. The engine overhead itself is tested (`test_throughput_1000_records`). To measure on your own account: `python scripts/benchmark.py --count 1000 --model claude-haiku-4-5-20251001 --concurrency 100 --batch-size 5 --no-scenarios`. Scenario planning adds one sequential call; reuse its plan with `--config`. |
| 100% schema validation pass rate | Every accepted record passes the full Pydantic model and the checks. Anything else is regenerated, never written. Tests assert this after deliberately corrupting responses. |
| 5+ data types and nested structures | 14 types, objects nested to any depth, arrays of objects |
| 3+ export formats | CSV, JSON, JSONL, Parquet, SQLite |
| Documented with examples | This README, 7 example schemas, `--check --show-prompt` |

## Project layout

```
synthetic-data-generator/
├── generator.py              # entry point: python generator.py ...
├── src/
│   ├── schema_builder.py     # Pydantic schema models, API JSON Schema compiler, record model
│   ├── designer.py           # --define: wizard, Claude-designed schemas, local inference from samples
│   ├── generator.py          # Claude-powered generation engine (async, batch API, resume)
│   ├── sampling.py           # quotas, distributions, record slots
│   ├── prompts.py            # system/user prompts (cached system prompt)
│   ├── validator.py          # record validation + dataset quality report (pandas)
│   ├── exporters.py          # CSV, JSON, JSONL, Parquet, SQLite (pandas) + readers
│   ├── storage.py            # checkpoints, last-run pointer
│   ├── mock_client.py        # offline mock of the API (--mock, tests)
│   └── cli.py                # Click CLI
├── examples/                 # 7 example schemas
├── tests/                    # test_schema / test_generator / test_validator / test_exporters / test_cli
├── scripts/benchmark.py
├── .run/                     # PyCharm run configurations
├── requirements.txt  pyproject.toml  .env.example
```

## Tests

```bash
pytest     # 62 tests, fully offline, ~5 s
```

- An offline mock model misbehaves on purpose: shuffled slots, wrong-case enums, invalid values,
  duplicates, truncation and refusals.
- The **real Anthropic SDK** runs over an in-process HTTP transport for JSON mode, tool mode and
  Message Batches.
- The CLI tests cover the full define, generate, validate, export and regenerate workflow.
- Verified on Python 3.10, 3.11 and 3.13.

## Limitations

- Synthetic data reflects your schema and Claude's knowledge, not the hidden correlations in your real
  data. Validate models trained on it against real held-out data.
- `--from-sample` reproduces each column's distribution but not the relationships between columns.
  Express the correlations that matter as `rules` or `checks`.
- `checks` are Python expressions evaluated with restricted builtins. Treat schema files like code.
- Enum values that differ only by capitalization are rejected, because structured outputs don't
  guarantee enum casing and the validator matches case-insensitively.
