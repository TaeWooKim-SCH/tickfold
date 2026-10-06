import asyncio
import json
import time

import pytest
from fakes import FakeConnect, FakeSocket
from test_stream import T, depth, frame

from tickfold.collector import connection
from tickfold.collector.binance_stream import StreamHandler
from tickfold.collector.connection import Feed, connect_forever
from tickfold.collector.writer import RawWriter

@pytest.fixture(autouse=True)
def fast_and_fixed_clock(monkeypatch):
    # 실제 값은 초 단위라 테스트가 느려진다. 순서만 같으면 된다
    monkeypatch.setattr(connection, "CONNECTION_CHECK_S", 0.005)
    monkeypatch.setattr(connection, "OVERLAP_S", 0.05)
    monkeypatch.setattr(connection, "BACKOFF_S", 0.01)
    # 소켓으로 받은 프레임은 지금 시각으로 찍힌다. 고정하지 않으면 정각을 넘는 순간 파일이 갈린다
    monkeypatch.setattr(time, "time_ns", lambda: T)

NOTICE = b'{"stream":"!serverShutdown","data":{"e":"serverShutdown","E":1770123456789}}'

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

async def until(condition, timeout=2.0):
    deadline = time.monotonic() + timeout
    while (not condition()):
        assert time.monotonic() < deadline, "조건이 시간 안에 참이 되지 않았다"
        await asyncio.sleep(0.002)

async def stop(task):
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

def test_backoff_doubles_and_stops_at_the_cap(monkeypatch):
    monkeypatch.setattr(connection, "BACKOFF_S", 1.0)
    assert [connection._backoff(attempt) for attempt in (0, 1, 2, 6, 5000)] == [1.0, 2.0, 4.0, 60.0, 60.0]

async def test_lost_connection_is_reopened_and_its_shutdown_notice_is_forgotten(tmp_path, monkeypatch):
    first, second, third = FakeSocket(), FakeSocket(), FakeSocket()
    connect = FakeConnect(first, second, third)
    monkeypatch.setattr(connection.websockets, "connect", connect)
    w, h = synced_handler(tmp_path)
    task = asyncio.create_task(connect_forever(h, "wss://test"))

    first.push(delta(101, 105), ConnectionError("끊김"))
    await until(lambda: connect.calls == 2)  # 끊긴 것을 알아채고 다시 붙었다
    second.push(delta(106, 110), NOTICE, ConnectionError("끊김"))  # 알림을 받고 갈아타기 전에 끊겼다
    await until(lambda: connect.calls == 3)
    third.push(delta(111, 115))
    await until(lambda: stored_first_ids(tmp_path, w) == [101, 106, 111])
    await asyncio.sleep(connection.OVERLAP_S * 2)
    await stop(task)

    assert connect.calls == 3  # 다시 붙은 연결을 옛 알림 때문에 또 갈아타지 않는다

async def test_old_connection_is_replaced_by_age_without_gap(tmp_path, monkeypatch):
    monkeypatch.setattr(connection, "ROTATE_AFTER_S", 0.03)
    monkeypatch.setattr(connection, "OVERLAP_S", 0.3)
    old, new = FakeSocket(), FakeSocket()
    connect = FakeConnect(old, new)
    monkeypatch.setattr(connection.websockets, "connect", connect)
    w, h = synced_handler(tmp_path)
    task = asyncio.create_task(connect_forever(h, "wss://test"))

    old.push(delta(101, 105))
    await until(lambda: connect.calls == 2)  # 나이가 차서 새 연결을 열었다
    monkeypatch.setattr(connection, "ROTATE_AFTER_S", 3600.0)  # 새 연결까지 또 갈아타지 않게
    old.push(delta(106, 110))
    new.push(delta(106, 110), delta(111, 115))
    await asyncio.sleep(0.05)
    assert not old.closed  # 겹치는 시간이 차기 전에는 넘겨받지 않는다
    await until(lambda: old.closed)
    new.push(delta(116, 120))
    await until(lambda: stored_first_ids(tmp_path, w) == [101, 106, 111, 116])
    await stop(task)

    assert h.gaps == 0
    assert connect.calls == 2

async def test_shutdown_notice_with_a_failed_new_connection_keeps_the_old_one_and_tries_again(tmp_path, monkeypatch):
    old, new = FakeSocket(), FakeSocket()
    new.push(delta(106, 110), delta(111, 115))  # 열리자마자 받는다. 테스트가 타이밍에 기대지 않게 미리 넣어 둔다
    connect = FakeConnect(old, OSError("연결 거부"), new)
    monkeypatch.setattr(connection.websockets, "connect", connect)
    w, h = synced_handler(tmp_path)
    task = asyncio.create_task(connect_forever(h, "wss://test"))

    old.push(delta(101, 105), NOTICE)
    await until(lambda: connect.calls >= 2)  # 알림을 보고 연 첫 후임은 열리지 못했다
    assert not old.closed  # 그래도 옛 연결은 살아 있다
    old.push(delta(106, 110))
    await until(lambda: old.closed)  # 다시 연 후임이 넘겨받았다
    await until(lambda: stored_first_ids(tmp_path, w) == [101, 106, 111])
    await stop(task)

    assert h.gaps == 0
    assert not h.shutdown_noticed
    assert connect.calls == 3

async def test_new_connection_waits_until_the_old_one_catches_up(tmp_path, monkeypatch):
    old, new = FakeSocket(), FakeSocket()
    connect = FakeConnect(old, new)
    monkeypatch.setattr(connection.websockets, "connect", connect)
    w, h = synced_handler(tmp_path)
    task = asyncio.create_task(connect_forever(h, "wss://test"))

    old.push(delta(101, 105), NOTICE)
    await until(lambda: connect.calls == 2)
    new.push(delta(111, 115))  # 옛 연결은 106~110 을 아직 못 줬다. 지금 넘겨받으면 그만큼이 빈다
    await asyncio.sleep(connection.OVERLAP_S * 4)
    assert not old.closed
    old.push(delta(106, 110), delta(111, 115))
    await until(lambda: old.closed)
    await stop(task)

    assert stored_first_ids(tmp_path, w) == [101, 106, 111]
    assert h.gaps == 0
    assert connect.calls == 2

async def test_new_connection_that_went_quiet_does_not_take_over(tmp_path, monkeypatch):
    monkeypatch.setattr(connection, "OVERLAP_S", 0.2)
    monkeypatch.setattr(connection, "STANDBY_QUIET_S", 0.05)
    monkeypatch.setattr(connection, "OVERLAP_MAX_S", 0.4)
    old, quiet, new = FakeSocket(), FakeSocket(), FakeSocket()
    connect = FakeConnect(old, quiet, new)
    monkeypatch.setattr(connection.websockets, "connect", connect)
    w, h = synced_handler(tmp_path)
    task = asyncio.create_task(connect_forever(h, "wss://test"))

    old.push(delta(101, 105), NOTICE)
    await until(lambda: connect.calls == 2)
    quiet.push(delta(101, 105))  # 옛 연결과 겹치는 프레임 하나만 주고 멈춘다
    await until(lambda: quiet.closed)  # 받다가 멈춘 후임은 버린다
    assert not old.closed  # 그동안 옛 연결은 살아 있다
    old.push(delta(106, 110))
    deadline = time.monotonic() + 2.0
    while (not old.closed):  # 다음 후임은 계속 받고 있어서 넘겨받는다
        assert time.monotonic() < deadline
        new.push(delta(106, 110))
        await asyncio.sleep(0.01)
    await stop(task)

    assert stored_first_ids(tmp_path, w) == [101, 106]
    assert h.gaps == 0
    assert connect.calls == 3

async def test_new_connection_takes_over_at_once_when_the_old_one_is_lost_mid_overlap(tmp_path, monkeypatch):
    monkeypatch.setattr(connection, "OVERLAP_S", 30.0)
    old, new = FakeSocket(), FakeSocket()
    connect = FakeConnect(old, new)
    monkeypatch.setattr(connection.websockets, "connect", connect)
    w, h = synced_handler(tmp_path)
    task = asyncio.create_task(connect_forever(h, "wss://test"))

    old.push(delta(101, 105), NOTICE)
    await until(lambda: connect.calls == 2)
    new.push(delta(101, 105), delta(106, 110))
    await asyncio.sleep(0.02)
    old.push(ConnectionError("끊김"))  # 겹치는 시간이 차기 전에 옛 연결이 끊긴다
    await until(lambda: stored_first_ids(tmp_path, w) == [101, 106])
    await stop(task)

    assert h.gaps == 0
    assert connect.calls == 2  # 다시 붙지 않고 후임이 이어받았다