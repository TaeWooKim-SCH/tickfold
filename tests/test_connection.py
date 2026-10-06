import json

from test_stream import T, depth, frame

from tickfold.collector.binance_stream import StreamHandler
from tickfold.collector.connection import Feed
from tickfold.collector.writer import RawWriter

def delta(U: int, u: int) -> bytes:
    return frame("btcusdt@depth@100ms", depth(U, u))

def synced_handler(tmp_path):
    w = RawWriter(tmp_path)
    h = StreamHandler(None, w, ["BTCUSDT"])
    h.books["BTCUSDT"].apply_snapshot({"lastUpdateId": 100, "bids": [], "asks": []})
    return w, h

def stored_first_ids(tmp_path, w):
    w.flush()
    lines = (tmp_path / "BTCUSDT" / "2026-09-15" / "00.depth.ndjson").read_bytes().splitlines()
    return [json.loads(line)["m"]["data"]["U"] for line in lines]

def test_standby_frames_wait_and_the_overlap_is_written_once(tmp_path):
    w, h = synced_handler(tmp_path)
    old, new = Feed(h, live=True), Feed(h, live=False)
    for U, u in ((101, 105), (106, 110)):
        old.receive(delta(U, u), T)
        new.receive(delta(U, u), T)
    new.receive(delta(111, 115), T)  # 새 연결이 한 발 앞서 받았다

    assert stored_first_ids(tmp_path, w) == [101, 106]  # 대기 연결이 받은 것은 아직 파일에 없다
    new.go_live()
    new.receive(delta(116, 120), T)

    assert stored_first_ids(tmp_path, w) == [101, 106, 111, 116]
    assert h.duplicates_dropped == 2  # 겹친 둘
    assert h.gaps == 0

def test_frame_arriving_late_on_the_new_connection_is_not_written_twice(tmp_path):
    w, h = synced_handler(tmp_path)
    old, new = Feed(h, live=True), Feed(h, live=False)
    old.receive(delta(101, 105), T)
    new.receive(delta(101, 105), T)
    old.receive(delta(106, 110), T)  # 새 연결은 이 프레임을 아직 못 받았다

    new.go_live()
    new.receive(delta(106, 110), T)  # 넘겨받은 뒤에 도착한다
    new.receive(delta(111, 115), T)

    assert stored_first_ids(tmp_path, w) == [101, 106, 111]
    assert h.gaps == 0