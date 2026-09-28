from __future__ import annotations

import json
import re
from collections import Counter

import pytest

from src.generator import GenerationError, extract_records, salvage_records
from src.sampling import QuotaPlanner, apportion
from src.schema_builder import schema_from_dict
from src.storage import Store
from src.validator import validate_record

from .conftest import run


def test_apportion_and_quota(basic):
    import random
    assert apportion(10, [0.2, 0.3, 0.5]) == [2, 3, 5]
    assert sum(apportion(7, [1 / 3] * 3)) == 7
    f = schema_from_dict(basic()).tables[0].field_map["tier"]
    q = QuotaPlanner(f, 100, random.Random(0))
    got = Counter()
    for _ in range(100):
        v = q.assign()
        q.release(v)
        q.accept(v)
        got[v] += 1
    assert got == {"gold": 20, "silver": 30, "bronze": 50}


def test_salvage_truncated_json():
    text = '{"records": [{"a": 1}, {"a": 2, "b": [1,2]}, {"a": 3, "b": "unfinish'
    assert salvage_records(text) == [{"a": 1}, {"a": 2, "b": [1, 2]}]
    assert salvage_records("garbage") == []


def test_end_to_end_with_an_imperfect_model(tmp_path, basic, mock):
    s = schema_from_dict(basic(diversity={"scenarios": 8, "edge_case_ratio": 0.2}))
    client = mock(seed=3, shuffle=True, wrong_case_rate=0.2, invalid_rate=0.1, disobey_rate=0.05,
                  duplicate_rate=0.3)
    g = run(s, client, out_dir=str(tmp_path))["people"]
    recs = g.records
    assert len(recs) == 60
    assert [r["id"] for r in recs] == [f"P{i:04d}" for i in range(1, 61)]  # gapless sequence
    assert len({r["uid"] for r in recs}) == 60
    t = s.tables[0]
    for r in recs:
        assert validate_record(t, r) == r  # already normalized, 100% schema-valid
    assert g.stats.rejected and g.stats.returned > 60  # rejections happened and were recovered
    assert g.stats.fixed_repaired > 0
    assert Store(tmp_path).load("people") == recs  # checkpoint == memory
    c = Counter(r["tier"] for r in recs)
    assert abs(c["bronze"] - 30) <= 4 and abs(c["gold"] - 12) <= 4
    assert len({r["name"].lower() for r in recs}) == 60
    call = client.model.calls[-1]
    assert "scenario:" in call["messages"][0]["content"]
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert call["output_config"]["format"]["type"] == "json_schema"


def test_strict_tool_use_mode(basic, mock):
    s = schema_from_dict(basic(count=30, mode="tool"))
    client = mock(seed=4)
    g = run(s, client)["people"]
    assert len(g.records) == 30
    call = client.model.calls[-1]
    assert call["tools"][0]["strict"] is True
    assert call["tool_choice"] == {"type": "tool", "name": "submit_records"}
    assert "output_config" not in call


def test_extract_records_from_tool_block():
    from types import SimpleNamespace as NS
    msg = NS(content=[NS(type="tool_use", name="submit_records", input={"records": [{"a": 1}]})])
    assert extract_records(msg) == [{"a": 1}]


def test_sampled_values_and_null_rate_are_enforced(basic, mock):
    s = schema_from_dict(basic(count=40))
    client = mock(seed=5, disobey_rate=0.5)
    recs = run(s, client)["people"].records
    ages = [r["age"] for r in recs]
    briefs = " ".join(c["messages"][0]["content"] for c in client.model.calls)
    assert set(ages) <= {int(a) for a in re.findall(r"age = (\d+)", briefs)}  # code-sampled, not model-chosen
    assert 0.05 < sum(r["score"] is None for r in recs) / len(recs) < 0.5


def test_strict_distribution_gives_exact_counts(basic, mock):
    s = schema_from_dict(basic(count=50, strict_distribution=True))
    recs = run(s, mock(seed=9, disobey_rate=0.3))["people"].records
    assert Counter(r["tier"] for r in recs) == {"bronze": 25, "silver": 15, "gold": 10}


def test_truncation_refusal_and_extra_records(basic, mock):
    s = schema_from_dict(basic(count=45, batch_size=10))
    g = run(s, mock(seed=11, truncate_every=3, refuse_every=5, extra_record=True))["people"]
    assert len(g.records) == 45
    assert g.stats.truncated > 0 and g.stats.refusals > 0
    assert g.batch_size < 10  # adapted after truncation


def test_checks_filter_and_feedback_reaches_prompt(basic, mock):
    s = schema_from_dict(basic(count=30, checks=["age >= 30", "len(tags) <= 2"]))
    client = mock(seed=2)
    g = run(s, client)["people"]
    assert len(g.records) == 30
    assert all(r["age"] >= 30 and len(r["tags"]) <= 2 for r in g.records)
    assert any(k.startswith("check") for k in g.stats.rejected)
    assert "<hard_checks>" in client.model.calls[-1]["system"][0]["text"]
    assert any("rejected by validation" in c["messages"][0]["content"] for c in client.model.calls)


def test_impossible_schema_stops_with_clear_error(basic, mock):
    s = schema_from_dict(basic(count=10, checks=["age > 1000"]))
    with pytest.raises(GenerationError, match="age > 1000"):
        run(s, mock(seed=1))


def test_fatal_api_error(basic, mock):
    import anthropic
    import httpx2

    client = mock()

    async def boom(**_):
        req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
        raise anthropic.BadRequestError("Schema is too complex for compilation",
                                        response=httpx2.Response(400, request=req), body=None)
    client.messages.create = boom
    with pytest.raises(GenerationError, match="too complex"):
        run(schema_from_dict(basic(count=5)), client)


def test_resume_after_crash(tmp_path, basic, mock):
    run(schema_from_dict(basic(count=20)), mock(seed=1), out_dir=str(tmp_path))
    path = tmp_path / "people.jsonl"
    with path.open("a") as fh:
        fh.write('{"id": "P0021", "na')  # torn last line
    g = run(schema_from_dict(basic(count=35)), mock(seed=2), out_dir=str(tmp_path), resume=True)["people"]
    assert [r["id"] for r in g.records] == [f"P{i:04d}" for i in range(1, 36)]
    lines = path.read_text().splitlines()
    assert len(lines) == 35 and all(json.loads(x) for x in lines)
    assert len({r["name"].lower() for r in g.records}) == 35


def test_multi_table_references(tmp_path, mock):
    raw = {"seed": 4, "tables": [
        {"name": "customers", "count": 12, "batch_size": 5, "fields": [
            {"name": "customer_id", "type": "string", "generator": "sequence", "format": "C{:03d}"},
            {"name": "name"}, {"name": "country", "values": ["DE", "FR"]},
            {"name": "since", "type": "date", "min": "2020-01-01", "max": "2021-01-01", "distribution": "uniform"}]},
        {"name": "orders", "count": 40, "batch_size": 8, "fields": [
            {"name": "order_id", "type": "integer", "generator": "sequence", "start": 1000},
            {"name": "customer_id", "ref": "customers.customer_id", "ref_context": ["country", "since"],
             "ref_skew": 1.2},
            {"name": "order_date", "type": "date", "min": "2022-01-01", "max": "2022-12-31"},
            {"name": "amount", "type": "number", "min": 1, "max": 500}],
         "checks": ["order_date >= date.fromisoformat(parent['customer_id']['since'])"]},
    ]}
    s = schema_from_dict(raw)
    client = mock(seed=4)
    gens = run(s, client, out_dir=str(tmp_path))
    cust = {c["customer_id"] for c in gens["customers"].records}
    orders = gens["orders"].records
    assert len(orders) == 40 and {o["customer_id"] for o in orders} <= cust
    assert orders[0]["order_id"] == 1000
    assert Counter(o["customer_id"] for o in orders).most_common(1)[0][1] >= 6  # skewed popularity
    order_prompt = next(c for c in client.model.calls if '"orders"' in c["messages"][0]["content"])
    assert "related customers record" in order_prompt["messages"][0]["content"]
    # child table alone: parents are loaded from disk
    g2 = run(s, mock(seed=5), out_dir=str(tmp_path), only=["orders"], counts={"orders": 10})
    assert len(g2["orders"].records) == 10


def test_batch_api_mode(tmp_path, basic, mock):
    s = schema_from_dict(basic(count=33, diversity={"scenarios": 5}))
    client = mock(seed=6, invalid_rate=0.15)
    g = run(s, client, out_dir=str(tmp_path), batch_api=True, poll_interval=0)["people"]
    assert len(g.records) == 33
    assert len(client.messages.batches.store) >= 2  # a top-up round replaced rejected records
    state = Store(tmp_path).load_state("people")
    assert state["pending_batch"] is None and state["scenarios"]


def test_preset_scenarios_and_seed_reproduce_code_side_values(basic, mock):
    """Same seed + same scenario plan -> identical code-sampled values (LLM text itself is not deterministic)."""
    def sampled(seed):
        s = schema_from_dict(basic(count=20, seed=seed))
        client = mock(seed=0)
        run(s, client, scenarios={"people": ["a", "b", "c"]})
        values = [m for c in client.model.calls
                  for m in re.findall(r"age = (\d+); joined = \"([\d-]+)\"", c["messages"][0]["content"])]
        return values[:20]
    assert sampled(123) == sampled(123)
    assert sampled(123) != sampled(124)


def _sdk_client(handler):
    import anthropic
    import httpx2
    return anthropic.AsyncAnthropic(api_key="test-key", max_retries=0,
                                    http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)))


@pytest.mark.parametrize("mode", ["json", "tool"])
def test_real_sdk_request_roundtrip(basic, mode):
    """Drive the real anthropic SDK over an in-process HTTP transport: the request we build is accepted
    by the SDK and serialized as the API expects, and real Message objects parse."""
    import httpx2

    from src.mock_client import MockModel

    model = MockModel(seed=8)
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append((request.url.path, request.headers, body))
        msg = model.respond(body)
        content = []
        for b in msg.content:
            if b.type == "tool_use":
                content.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
            else:
                content.append({"type": "text", "text": b.text})
        return httpx2.Response(200, json={
            "id": f"msg_{len(seen)}", "type": "message", "role": "assistant", "model": body["model"],
            "content": content, "stop_reason": msg.stop_reason, "stop_sequence": None,
            "usage": {"input_tokens": 1200, "output_tokens": msg.usage.output_tokens,
                      "cache_read_input_tokens": 1000, "cache_creation_input_tokens": 0}})

    s = schema_from_dict(basic(count=20, mode=mode, diversity={"scenarios": 4}))
    g = run(s, _sdk_client(handler))["people"]
    assert len(g.records) == 20 and g.stats.cache_read_tokens > 0
    path, headers, body = seen[-1]
    assert path == "/v1/messages"
    assert "anthropic-beta" not in headers  # GA structured outputs: no beta header
    if mode == "json":
        assert body["output_config"]["format"]["type"] == "json_schema"
    else:
        assert body["tools"][0]["strict"] is True and body["tool_choice"]["type"] == "tool"


def test_real_sdk_batch_roundtrip(tmp_path, basic):
    import httpx2

    from src.mock_client import MockModel

    model = MockModel(seed=12)
    batches: dict[str, list] = {}
    polls: Counter = Counter()

    def batch_json(bid, status):
        return {"id": bid, "type": "message_batch", "processing_status": status,
                "request_counts": {"processing": 0, "succeeded": len(batches[bid]), "errored": 0, "canceled": 0,
                                   "expired": 0},
                "created_at": "2026-01-01T00:00:00Z", "expires_at": "2026-01-02T00:00:00Z",
                "ended_at": "2026-01-01T00:10:00Z" if status == "ended" else None,
                "cancel_initiated_at": None, "archived_at": None,
                "results_url": f"https://api.anthropic.com/v1/messages/batches/{bid}/results"
                if status == "ended" else None}

    def handler(request):
        path = request.url.path
        if request.method == "POST" and path == "/v1/messages/batches":
            bid = f"msgbatch_{len(batches) + 1}"
            batches[bid] = json.loads(request.content)["requests"]
            return httpx2.Response(200, json=batch_json(bid, "in_progress"))
        if path.endswith("/results"):
            bid = path.split("/")[-2]
            lines = []
            for r in batches[bid]:
                m = model.respond(r["params"])
                lines.append(json.dumps({"custom_id": r["custom_id"], "result": {"type": "succeeded", "message": {
                    "id": "msg_x", "type": "message", "role": "assistant", "model": r["params"]["model"],
                    "content": [{"type": "text", "text": m.content[0].text}], "stop_reason": m.stop_reason,
                    "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 10}}}}))
            return httpx2.Response(200, content="\n".join(lines).encode())
        bid = path.split("/")[-1]
        polls[bid] += 1
        return httpx2.Response(200, json=batch_json(bid, "ended" if polls[bid] > 1 else "in_progress"))

    s = schema_from_dict(basic(count=25))
    g = run(s, _sdk_client(handler), out_dir=str(tmp_path), batch_api=True, poll_interval=0)["people"]
    assert len(g.records) == 25


def test_throughput_1000_records(basic, mock):
    """Engine overhead check: 1000 records with 0.25 s simulated latency per request, 50 in flight."""
    import time
    s = schema_from_dict(basic(count=1000, batch_size=10, concurrency=50))
    t0 = time.time()
    g = run(s, mock(seed=7, latency=0.25))["people"]
    elapsed = time.time() - t0
    assert len(g.records) == 1000
    assert elapsed < 30, elapsed
