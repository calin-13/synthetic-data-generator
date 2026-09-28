from __future__ import annotations

import sqlite3

import pytest

from src.exporters import export_dataset, flatten, read_table, unflatten, write_table
from src.schema_builder import schema_from_dict
from src.validator import validate_record

from .conftest import run


@pytest.fixture
def people(basic, mock):
    s = schema_from_dict(basic(count=15))
    return s.tables[0], run(s, mock(seed=1))["people"].records


@pytest.mark.parametrize("ext", ["csv", "json", "jsonl", "parquet", "sqlite"])
def test_roundtrip_every_format(tmp_path, people, ext):
    table, rows = people
    p = write_table(rows, tmp_path / f"people.{ext}", table=table)
    back = read_table(p)
    assert len(back) == 15
    for original, restored in zip(rows, back):
        assert validate_record(table, restored) == original


def test_flatten_unflatten():
    rec = {"a": {"b": 1, "c": {"d": 2}}, "e": [1, 2]}
    assert flatten(rec) == {"a.b": 1, "a.c.d": 2, "e": "[1, 2]"}
    assert unflatten(flatten(rec)) == rec


def test_sqlite_with_foreign_keys(tmp_path, mock):
    s = schema_from_dict({"seed": 3, "tables": [
        {"name": "authors", "count": 5, "fields": [
            {"name": "author_id", "type": "integer", "generator": "sequence"}, {"name": "name"}]},
        {"name": "books", "count": 12, "fields": [
            {"name": "book_id", "type": "string", "generator": "sequence", "format": "B{:03d}"},
            {"name": "author_id", "ref": "authors.author_id"}, {"name": "title"}]},
    ]})
    gens = run(s, mock(seed=3))
    paths = export_dataset({k: g.records for k, g in gens.items()}, {t.name: t for t in s.tables}, tmp_path,
                           ["sqlite", "csv"])
    assert {p.name for p in paths} == {"dataset.sqlite", "authors.csv", "books.csv"}
    con = sqlite3.connect(tmp_path / "dataset.sqlite")
    assert con.execute("select count(*) from books join authors using(author_id)").fetchone()[0] == 12
    ddl = con.execute("select sql from sqlite_master where name='books'").fetchone()[0]
    assert "FOREIGN KEY" in ddl and "PRIMARY KEY" in ddl
