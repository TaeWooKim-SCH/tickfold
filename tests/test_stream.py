import json
import asyncio

from datetime import datetime, timezone
from fakes import FakeResponse, FakeSession
from tickfold.collector.binance_stream import StreamHandler, split_stream, stream_names
from tickfold.collector.writer import RawWriter

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