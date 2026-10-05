import json
import asyncio
import pytest

from datetime import datetime, timezone
from fakes import FakeResponse, FakeSession

from tickfold.collector import binance_stream
from tickfold.collector.binance_stream import StreamHandler, split_stream, stream_names
from tickfold.collector.writer import RawWriter

@pytest.fixture(autouse=True)
def no_settle_wait(monkeypatch):
    # 실제 대기는 스냅샷이 버퍼보다 오래되는 경합을 막으려는 것이라 가짜 세션에서는 필요 없다
    monkeypatch.setattr(binance_stream, "SNAPSHOT_SETTLE_S", 0.0)

def test_stream_names_covers_symbol_times_stream():
    assert stream_names(["BTCUSDT"]) == "btcusdt@depth@100ms/btcusdt@trade/btcusdt@depth20@100ms"
    assert stream_names(["BTCUSDT", "ETHUSDT"]).count("/") == 5

def test_split_stream_drops_speed_suffix():
    assert split_stream("btcusdt@depth@100ms") == ("BTCUSDT", "depth")
    assert split_stream("btcusdt@depth20@100ms") == ("BTCUSDT", "depth20")
    assert split_stream("btcusdt@trade") == ("BTCUSDT", "trade")

def ns(y, m, d, h, mi=0):
    return int(datetime(y, m, d, h, mi, tzinfo=timezone.utc).timestamp() * 1_000_000_000)

T = ns(2026, 9, 15, 0)

def frame(stream: str, data: dict) -> bytes:
    return json.dumps({"stream": stream, "data": data}, separators=(",", ":")).encode()

def test_frame_is_stored_verbatim_under_symbol_and_stream(tmp_path):
    w = RawWriter(tmp_path)
    h = StreamHandler(None, w, ["BTCUSDT"])
    h.books["BTCUSDT"].apply_snapshot({"lastUpdateId": 156, "bids": [], "asks": []})
    f = frame("btcusdt@depth@100ms", {"U": 157, "u": 160})
    h.handle(f, T)
    w.flush()

    line = (tmp_path / "BTCUSDT" / "2026-09-15" / "00.depth.ndjson").read_bytes().splitlines()[0]
    assert line == b'{"rx":%d,"m":' % T + f + b"}"

def test_streams_go_to_separate_files(tmp_path):
    w = RawWriter(tmp_path)
    h = StreamHandler(None, w, ["BTCUSDT"])
    h.books["BTCUSDT"].apply_snapshot({"lastUpdateId": 156, "bids": [], "asks": []})
    h.handle(frame("btcusdt@depth@100ms", {"U": 157, "u": 160}), T)
    h.handle(frame("btcusdt@trade", {"t": 1}), T)
    h.handle(frame("btcusdt@depth20@100ms", {"lastUpdateId": 1}), T)
    w.close()

    day = tmp_path / "BTCUSDT" / "2026-09-15"
    assert sorted(p.name for p in day.glob("*.zst")) == [
        "00.depth.ndjson.zst", "00.depth20.ndjson.zst", "00.trade.ndjson.zst"
    ]

def test_unknown_frame_is_kept_not_raised(tmp_path):
    w = RawWriter(tmp_path)
    h = StreamHandler(None, w, ["BTCUSDT"])
    f = b'{"result":null,"id":1}'
    h.handle(f, T)   # 예외가 나면 연결이 끊긴다
    w.flush()

    line = (tmp_path / "_control" / "2026-09-15" / "00.raw.ndjson").read_bytes().splitlines()[0]
    assert line == b'{"rx":%d,"m":' % T + f + b"}"

def depth(U: int, u: int) -> dict:
    return {"e": "depthUpdate", "s": "BTCUSDT", "U": U, "u": u, "b": [], "a": []}

def test_depth_advances_the_book_after_snapshot(tmp_path):
    w = RawWriter(tmp_path)
    h = StreamHandler(None, w, ["BTCUSDT"])
    book = h.books["BTCUSDT"]
    book.apply_snapshot({"lastUpdateId": 100, "bids": [], "asks": []})

    h.handle(frame("btcusdt@depth@100ms", depth(101, 105)), T)
    assert book.last_u == 105

def test_trade_and_depth20_leave_the_book_alone(tmp_path):
    w = RawWriter(tmp_path)
    h = StreamHandler(None, w, ["BTCUSDT"])
    book = h.books["BTCUSDT"]
    book.apply_snapshot({"lastUpdateId": 100, "bids": [], "asks": []})

    h.handle(frame("btcusdt@trade", {"e": "trade", "t": 1}), T)
    h.handle(frame("btcusdt@depth20@100ms", {"lastUpdateId": 999, "bids": [], "asks": []}), T)
    assert book.last_u == 100  # 순번이 안 움직인다

async def test_gap_is_recorded_with_how_many_were_missed(tmp_path):
    w = RawWriter(tmp_path)
    h = StreamHandler(FakeSession(FakeResponse(200, payload=SNAP)), w, ["BTCUSDT"])
    book = h.books["BTCUSDT"]
    book.apply_snapshot({"lastUpdateId": 100, "bids": [], "asks": []})
    h.handle(frame("btcusdt@depth@100ms", depth(101, 105)), T)

    h.handle(frame("btcusdt@depth@100ms", depth(120, 125)), T)  # 106~119 를 놓쳤다
    w.flush()

    assert h.gaps == 1
    assert not book.synced  # create_task 는 던지기만 했고 아직 안 돈다
    rec = json.loads((tmp_path / "BTCUSDT" / "2026-09-15" / "00.gap.ndjson").read_bytes())["m"]
    assert (rec["last_u"], rec["U"], rec["missed"]) == (105, 120, 14)

    await drain(h)  # 안 거두면 "coroutine was never awaited" 경고가 뜬다

SNAP = {"lastUpdateId": 133, "bids": [], "asks": []}

async def drain(h):
    await asyncio.gather(*list(h.tasks))

async def test_buffered_deltas_are_replayed_onto_the_snapshot(tmp_path):
    w = RawWriter(tmp_path)
    h = StreamHandler(FakeSession(FakeResponse(200, payload=SNAP)), w, ["BTCUSDT"])
    book = h.books["BTCUSDT"]
    book.apply_snapshot({"lastUpdateId": 100, "bids": [], "asks": []})
    h.handle(frame("btcusdt@depth@100ms", depth(101, 105)), T)

    h.handle(frame("btcusdt@depth@100ms", depth(120, 125)), T)  # 갭. 여기서 버퍼가 시작된다
    h.handle(frame("btcusdt@depth@100ms", depth(126, 130)), T)  # 스냅샷 받는 동안 도착
    h.handle(frame("btcusdt@depth@100ms", depth(131, 135)), T)  # 이것도
    assert not book.synced
    assert len(h.tasks) == 1  # 스냅샷 요청은 한 번만

    await drain(h)

    # 스냅샷 133 기준으로 120~125 와 126~130 은 이미 반영돼 있어 dup 으로 버려지고,
    # 133 을 걸치는 131~135 만 적용된다
    assert book.synced
    assert book.last_u == 135

async def test_first_frame_on_a_fresh_book_triggers_resync(tmp_path):
    w = RawWriter(tmp_path)
    h = StreamHandler(FakeSession(FakeResponse(200, payload=SNAP)), w, ["BTCUSDT"])
    book = h.books["BTCUSDT"]
    assert not book.synced  # 연결 직후에는 스냅샷이 없다

    h.handle(frame("btcusdt@depth@100ms", depth(131, 135)), T)
    assert h.gaps == 0        # 갭이 아니라 그냥 아직 동기화 전이다
    assert len(h.tasks) == 1  # 그래도 스냅샷을 받으러 간다
    await drain(h)

    assert book.synced
    assert list(tmp_path.rglob("snapshots.ndjson"))

async def test_gap_while_book_is_unsynced_is_still_recorded(tmp_path):
    w = RawWriter(tmp_path)
    h = StreamHandler(FakeSession(FakeResponse(200, payload=SNAP)), w, ["BTCUSDT"])
    h.handle(frame("btcusdt@depth@100ms", depth(101, 105)), T)
    assert not h.books["BTCUSDT"].synced  # 스냅샷 전이라 호가창은 버퍼에 쌓기만 한다

    h.handle(frame("btcusdt@depth@100ms", depth(120, 125)), T)  # 106~119 를 놓쳤다
    h.handle(frame("btcusdt@depth@100ms", depth(126, 135)), T)
    await drain(h)
    w.flush()

    assert h.books["BTCUSDT"].synced  # 호가창은 스냅샷으로 멀쩡히 섰고 갭을 모른다
    assert h.gaps == 1                # 순번 추적은 안다
    rec = json.loads((tmp_path / "BTCUSDT" / "2026-09-15" / "00.gap.ndjson").read_bytes())["m"]
    assert (rec["last_u"], rec["U"], rec["missed"]) == (105, 120, 14)

async def test_duplicate_and_overlap_are_not_gaps(tmp_path):
    w = RawWriter(tmp_path)
    h = StreamHandler(FakeSession(FakeResponse(200, payload=SNAP)), w, ["BTCUSDT"])
    h.handle(frame("btcusdt@depth@100ms", depth(101, 105)), T)
    h.handle(frame("btcusdt@depth@100ms", depth(103, 105)), T)  # 같은 구간을 또 받음
    h.handle(frame("btcusdt@depth@100ms", depth(104, 108)), T)  # 앞과 겹치지만 빠진 건 없다
    await drain(h)

    assert h.gaps == 0
    assert not list(tmp_path.rglob("*.gap.ndjson"))

def ladder_snapshot(last_update_id: int) -> dict:
    # 매수 100~91, 매도 101~110. 양쪽 폭이 9
    bids = [[f"{price}.00", "1"] for price in range(100, 90, -1)]
    asks = [[f"{price}.00", "1"] for price in range(101, 111)]
    return {"lastUpdateId": last_update_id, "bids": bids, "asks": asks}

def top_levels(best_bid: str, best_ask: str) -> dict:
    return {"lastUpdateId": 1, "bids": [[best_bid, "1"]], "asks": [[best_ask, "1"]]}

async def test_snapshot_is_refreshed_when_price_nears_the_edge(tmp_path):
    w = RawWriter(tmp_path)
    s = FakeSession(FakeResponse(200, payload=ladder_snapshot(200)))
    h = StreamHandler(s, w, ["BTCUSDT"])
    book = h.books["BTCUSDT"]
    book.apply_snapshot(ladder_snapshot(100))

    h.handle(frame("btcusdt@depth20@100ms", top_levels("96.00", "97.00")), T)  # 여유 56%
    assert book.synced and not h.tasks

    h.handle(frame("btcusdt@depth20@100ms", top_levels("92.00", "93.00")), T)  # 여유 11%
    assert not book.synced  # 새로 세우려고 무효로 돌렸다
    await drain(h)

    assert book.synced
    assert h.gaps == 0      # 갱신은 갭이 아니다
    assert len(s.calls) == 1

async def test_refresh_is_not_repeated_within_the_minimum_interval(tmp_path):
    w = RawWriter(tmp_path)
    s = FakeSession(FakeResponse(200, payload=ladder_snapshot(200)))
    h = StreamHandler(s, w, ["BTCUSDT"])
    h.books["BTCUSDT"].apply_snapshot(ladder_snapshot(100))

    h.handle(frame("btcusdt@depth20@100ms", top_levels("92.00", "93.00")), T)
    await drain(h)
    h.handle(frame("btcusdt@depth20@100ms", top_levels("92.00", "93.00")), T)  # 여전히 끝 근처

    assert not h.tasks           # 방금 받았으니 다시 안 받는다
    assert len(s.calls) == 1

async def test_snapshot_request_waits_so_the_buffer_starts_first(tmp_path, monkeypatch):
    monkeypatch.setattr(binance_stream, "SNAPSHOT_SETTLE_S", 0.01)
    buffered_at_request = []
    holder = {}

    class RecordingSession:
        def get(self, url, params=None, timeout=None):
            buffered_at_request.append(len(holder["book"].buffer))
            return FakeResponse(200, payload=SNAP)

    h = StreamHandler(RecordingSession(), RawWriter(tmp_path), ["BTCUSDT"])
    holder["book"] = h.books["BTCUSDT"]

    h.handle(frame("btcusdt@depth@100ms", depth(126, 130)), T)
    await asyncio.sleep(0)  # 재동기화 태스크가 돌기 시작한다
    h.handle(frame("btcusdt@depth@100ms", depth(131, 135)), T)  # 기다리는 동안 도착
    await drain(h)

    assert buffered_at_request == [2]  # 요청이 나갈 때 둘 다 버퍼에 있었다
    assert h.books["BTCUSDT"].last_u == 135

async def test_stale_snapshot_is_refreshed_regardless_of_range(tmp_path, monkeypatch):
    w = RawWriter(tmp_path)
    s = FakeSession(
        FakeResponse(200, payload=ladder_snapshot(200)),
        FakeResponse(200, payload=ladder_snapshot(300)),
    )
    h = StreamHandler(s, w, ["BTCUSDT"])
    h.handle(frame("btcusdt@depth@100ms", depth(201, 205)), T)  # 첫 스냅샷을 받게 한다
    await drain(h)

    h.refresh_stale()  # 방금 받은 스냅샷이라 아직 오래되지 않았다
    assert not h.tasks

    monkeypatch.setattr(binance_stream, "SNAPSHOT_MAX_AGE_S", 0.0)  # 시간이 흐른 것을 흉내
    monkeypatch.setattr(binance_stream, "REFRESH_MIN_INTERVAL_S", 0.0)
    h.refresh_stale()
    await drain(h)
    w.flush()

    assert len(s.calls) == 2
    assert h.gaps == 0  # 주기 갱신은 갭이 아니다
    snapshot_file = next(tmp_path.rglob("snapshots.ndjson"))
    assert len(snapshot_file.read_bytes().splitlines()) == 2

async def test_gap_across_a_restart_is_recorded(tmp_path):
    before = RawWriter(tmp_path)
    h = StreamHandler(None, before, ["BTCUSDT"])
    h.books["BTCUSDT"].apply_snapshot({"lastUpdateId": 125, "bids": [], "asks": []})
    h.handle(frame("btcusdt@depth@100ms", depth(126, 130)), T)
    h.handle(frame("btcusdt@depth@100ms", depth(131, 135)), T)
    before.close()  # 수집기가 내려갔다

    w = RawWriter(tmp_path)
    s = FakeSession(FakeResponse(200, payload={"lastUpdateId": 140, "bids": [], "asks": []}))
    h = StreamHandler(s, w, ["BTCUSDT"])
    h.resume_sequence(tmp_path)
    h.handle(frame("btcusdt@depth@100ms", depth(141, 145)), T)  # 내려가 있던 동안 136~140 을 놓쳤다
    await drain(h)
    w.flush()

    assert h.gaps == 1
    rec = json.loads(next(tmp_path.rglob("*.gap.ndjson")).read_bytes())["m"]
    assert (rec["last_u"], rec["U"], rec["missed"]) == (135, 141, 5)

def test_shutdown_notice_is_kept_not_raised(tmp_path):
    w = RawWriter(tmp_path)
    h = StreamHandler(None, w, ["BTCUSDT"])
    f = b'{"stream":"!serverShutdown","data":{"e":"serverShutdown","E":1770123456789}}'
    h.handle(f, T)  # 스트림 이름에 @ 가 없다. 예외가 나면 알림은 저장되지 않고 연결이 끊긴다
    w.flush()

    line = (tmp_path / "_control" / "2026-09-15" / "00.raw.ndjson").read_bytes().splitlines()[0]
    assert line == b'{"rx":%d,"m":' % T + f + b"}"