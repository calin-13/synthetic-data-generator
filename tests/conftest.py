from __future__ import annotations

import asyncio
import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.generator import generate_dataset  # noqa: E402
from src.mock_client import MockAsyncAnthropic  # noqa: E402

EXAMPLES = ROOT / "examples"

BASIC = {
    "name": "people",
    "description": "test people",
    "count": 60,
    "batch_size": 7,
    "seed": 1,
    "concurrency": 4,
    "fields": [
        {"name": "id", "type": "string", "generator": "sequence", "format": "P{:04d}"},
        {"name": "uid", "type": "uuid", "generator": "uuid"},
        {"name": "name", "type": "string", "unique": True},
        {"name": "email", "type": "email"},
        {"name": "tier", "type": "enum", "values": ["gold", "silver", "bronze"], "weights": [0.2, 0.3, 0.5]},
        {"name": "age", "type": "integer", "min": 18, "max": 90,
         "distribution": {"type": "normal", "mean": 40, "std": 12}},
        {"name": "joined", "type": "date", "min": "2024-01-01", "max": "2024-12-31", "distribution": "uniform"},
        {"name": "score", "type": "number", "min": 0, "max": 1, "null_rate": 0.25},
        {"name": "bio", "type": "text", "max_length": 400},
        {"name": "tags", "type": "array", "items": "string", "min_items": 1, "max_items": 3},
        {"name": "address", "type": "object", "properties": [
            {"name": "city", "type": "string"}, {"name": "zip", "type": "string"}]},
    ],
}


@pytest.fixture
def basic():
    """A fresh copy of a schema exercising every major feature; tweak per test."""
    def make(**over):
        s = copy.deepcopy(BASIC)
        s.update(over)
        return s
    return make


def run(schema, client, **kw):
    return asyncio.run(generate_dataset(schema, client=client, **kw))


@pytest.fixture
def mock():
    return MockAsyncAnthropic
