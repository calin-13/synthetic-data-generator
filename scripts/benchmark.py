#!/usr/bin/env python3
"""Throughput benchmark.

Real API (needs ANTHROPIC_API_KEY; costs tokens):
    python scripts/benchmark.py --schema examples/customer_profiles.json --count 1000 \\
        --model claude-haiku-4-5-20251001 --concurrency 64 --batch-size 5

Offline simulation (latency = time-to-first-token + output_tokens / tokens_per_second per request):
    python scripts/benchmark.py --schema examples/customer_profiles.json --count 1000 --simulate \\
        --tps 150 --ttft 0.8 --concurrency 64 --batch-size 5

Wall time is roughly  ceil(requests / concurrency) x (ttft + batch_size x tokens_per_record / tps),
so throughput scales with concurrency until your account's rate limits (requests and output tokens per
minute) cap it. The simulation's tps/ttft are assumptions: measure your own with a small real run.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generator import generate_dataset, make_client  # noqa: E402
from src.mock_client import MockAsyncAnthropic  # noqa: E402
from src.schema_builder import load_schema  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--schema", default="examples/customer_profiles.json")
    ap.add_argument("--count", type=int, default=1000)
    ap.add_argument("--model")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=5)
    ap.add_argument("--simulate", action="store_true", help="offline, with a latency model")
    ap.add_argument("--tps", type=float, default=150.0, help="simulated output tokens/second per request")
    ap.add_argument("--ttft", type=float, default=0.8, help="simulated time to first token (s)")
    ap.add_argument("--no-scenarios", action="store_true", help="skip the scenario-planning call")
    args = ap.parse_args()

    schema = load_schema(args.schema)
    table = schema.tables[0]
    table.batch_size = args.batch_size
    if args.no_scenarios:
        table.diversity.scenarios = 0
    schema.settings.concurrency = args.concurrency
    if args.model:
        schema.settings.model = args.model
    if args.simulate:
        client = MockAsyncAnthropic(tokens_per_second=args.tps, ttft=args.ttft, obey_checks=True)
        label = f"SIMULATED (tps={args.tps}, ttft={args.ttft}s)"
    else:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("ANTHROPIC_API_KEY is not set (or use --simulate)", file=sys.stderr)
            return 2
        client = make_client()
        label = f"REAL API ({schema.settings.model})"

    t0 = time.time()
    gens = asyncio.run(generate_dataset(schema, client=client, counts={table.name: args.count}))
    elapsed = time.time() - t0
    g = gens[table.name]
    st = g.stats
    print(f"{label}: {len(g.records)} records in {elapsed:.1f}s = {len(g.records) / elapsed:.1f} rec/s")
    print(f"  requests {st.requests}, acceptance {st.acceptance_rate:.1%}, "
          f"output tokens {st.output_tokens:,} ({st.output_tokens / max(1, len(g.records)):.0f}/record), "
          f"cache reads {st.cache_read_tokens:,}")
    print(f"  target '1000+ records in < 30 s': {'MET' if len(g.records) >= 1000 and elapsed < 30 else 'not met'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
