import json
from pathlib import Path

import pytest

from tickfold.collector.book import OrderBook

FIXTURES = Path(__file__).parent / "fixtures"

def msg(U, u, b=(), a=()):
    return {"U": U, "u": u, "b": list(b), "a": list(a)}

def snapshot(last_update_id, bids=(), asks=()):
    return {"lastUpdateId": last_update_id, "bids": list(bids), "asks": list(asks)}

def test_sequence_and_gap():
    book = OrderBook()
    book.apply_snapshot(snapshot(156))
    assert book.apply_delta(msg(157, 160)) == "ok"
    assert book.apply_delta(msg(161, 161)) == "ok"
    assert book.apply_delta(msg(162, 165)) == "ok"
    assert book.apply_delta(msg(170, 172)) == "gap"   # 166~169 놓침
    assert not book.synced
    assert book.buffer == [msg(170, 172)]              # 재동기화 후 재적용 대상

def test_duplicate_dropped():
    book = OrderBook()
    book.apply_snapshot(snapshot(156))
    book.apply_delta(msg(157, 165))
    assert book.apply_delta(msg(162, 165)) == "dup"
    assert book.last_u == 165

def test_pending_before_snapshot_then_drained():
    book = OrderBook()
    assert book.apply_delta(msg(150, 155)) == "pending"   # 스냅샷에 이미 반영됨 -> dup
    assert book.apply_delta(msg(156, 158)) == "pending"   # 156 <= 157 <= 158 -> 첫 적용
    assert book.apply_delta(msg(159, 160)) == "pending"
    results = book.apply_snapshot(snapshot(156))
    assert results == ["dup", "ok", "ok"]
    assert book.last_u == 160

def test_first_event_must_straddle_snapshot():
    book = OrderBook()
    book.apply_snapshot(snapshot(156))
    assert book.apply_delta(msg(158, 160)) == "gap"       # 157을 놓침

def test_zero_quantity_removes_level():
    book = OrderBook()
    book.apply_snapshot(snapshot(100, bids=[["50000.00", "2.0"]], asks=[["50001.00", "1.5"]]))
    book.apply_delta(msg(101, 101, b=[["50000.00", "2.5"]], a=[["50001.00", "0.00000000"]]))
    assert book.bids["50000.00"] == "2.5"
    assert "50001.00" not in book.asks

@pytest.mark.skipif(
    not (FIXTURES / "btcusdt_depth.ndjson").exists(), reason="fixture not captured yet"
)
def test_fixture_replay_has_no_gap():
    # 캡처 스크립트는 연결 직후 스냅샷을 받으므로: 전부 pending으로 버퍼링 -> 스냅샷 적용 -> 버퍼 소진
    snap = json.loads((FIXTURES / "btcusdt_snapshot.json").read_bytes())
    with open(FIXTURES / "btcusdt_depth.ndjson", "rb") as f:
        msgs = [json.loads(line)["m"] for line in f]
 
    book = OrderBook()
    for m in msgs:
        assert book.apply_delta(m) == "pending"
    results = book.apply_snapshot(snap)
 
    assert "gap" not in results
    assert "ok" in results
    assert book.synced