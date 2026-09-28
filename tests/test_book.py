import json
from pathlib import Path

import pytest

from tickfold.collector.orderbook import OrderBook
from tickfold.collector.binance_stream import DEPTH, split_stream

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

def test_invalidate_buffers_deltas_until_the_next_snapshot():
    book = OrderBook()
    book.apply_snapshot(snapshot(100))
    book.apply_delta(msg(101, 105))

    book.invalidate()
    assert not book.synced
    assert book.apply_delta(msg(106, 110)) == "pending"  # 갭이 아니라 대기
    assert book.apply_delta(msg(111, 115)) == "pending"

    results = book.apply_snapshot(snapshot(108))  # 버퍼 중간에 걸치는 스냅샷
    assert results == ["ok", "ok"]
    assert book.last_u == 115

def ladder():
    # 매수 100~91, 매도 101~110. 양쪽 폭이 9
    bids = [[f"{price}.00", "1"] for price in range(100, 90, -1)]
    asks = [[f"{price}.00", "1"] for price in range(101, 111)]
    return snapshot(156, bids, asks)

def test_headroom_shrinks_as_price_nears_the_edge_of_the_snapshot():
    book = OrderBook()
    book.apply_snapshot(ladder())
    assert book.range_headroom(100.0, 101.0) == pytest.approx(1.0)   # 스냅샷 직후
    assert book.range_headroom(92.8, 93.8) == pytest.approx(0.2)     # 아래쪽 여유가 9 중 1.8 남음
    assert book.range_headroom(108.0, 108.2) == pytest.approx(0.2)   # 위쪽도 같은 규칙
    assert book.range_headroom(90.0, 91.0) < 0                       # 아는 범위를 벗어남

def test_headroom_is_unknown_without_a_usable_range():
    book = OrderBook()
    assert book.range_headroom(100.0, 101.0) is None   # 스냅샷 전
    book.apply_snapshot(snapshot(156))                 # 빈 스냅샷
    assert book.range_headroom(100.0, 101.0) is None
    book.apply_snapshot(ladder())
    book.apply_delta(msg(170, 172))                    # 갭으로 무효가 됨
    assert book.range_headroom(100.0, 101.0) is None

@pytest.mark.skipif(
    not (FIXTURES / "btcusdt_combined.ndjson").exists(), reason="fixture not captured yet"
)
def test_fixture_replay_has_no_gap():
    # 캡처 스크립트는 연결 직후 스냅샷을 받으므로: 전부 pending으로 버퍼링 -> 스냅샷 적용 -> 버퍼 소진
    snap = json.loads((FIXTURES / "btcusdt_snapshot.json").read_bytes())
    with open(FIXTURES / "btcusdt_combined.ndjson", "rb") as f:
        frames = [json.loads(line)["m"] for line in f]
    msgs = [fr["data"] for fr in frames if split_stream(fr["stream"])[1] == DEPTH]
 
    book = OrderBook()
    for m in msgs:
        assert book.apply_delta(m) == "pending"
    results = book.apply_snapshot(snap)
 
    assert "gap" not in results
    assert "ok" in results
    assert book.synced