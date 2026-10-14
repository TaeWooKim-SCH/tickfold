import json
import re
import runpy
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from tickfold.replay import segment as segment_module
from tickfold.replay.depth20 import parse_depth20
from tickfold.replay.segment import load_segment, read_segment, save_segment

PROJECT = Path(__file__).parent.parent
FIXTURE = PROJECT / "tests" / "fixtures" / "btcusdt_combined.ndjson"
FIXTURE_START = datetime(2026, 9, 27, 6, 14, 0, tzinfo=timezone.utc)
FIXTURE_END = datetime(2026, 9, 27, 6, 16, 0, tzinfo=timezone.utc)


def fixture_lines(stream: bytes) -> list[bytes]:
    return [line for line in FIXTURE.read_bytes().splitlines() if b"@" + stream + b"@" in line]


def write_day(symbol_directory: Path, day: str, hour: str, depth: list[bytes], depth20: list[bytes]) -> None:
    (symbol_directory / day).mkdir(parents=True, exist_ok=True)
    (symbol_directory / day / f"{hour}.depth.ndjson").write_bytes(b"\n".join(depth) + b"\n")
    (symbol_directory / day / f"{hour}.depth20.ndjson").write_bytes(b"\n".join(depth20) + b"\n")


def exchange_time_of(line: bytes) -> datetime:
    return datetime.fromtimestamp(json.loads(line)["m"]["data"]["E"] / 1000, tz=timezone.utc)


def received_a_day_later(line: bytes) -> bytes:
    # 줄의 첫 숫자가 수신 시각(나노초)이다
    return re.sub(rb"\d+", lambda received_at: b"%d" % (int(received_at[0]) + 86_400 * 10**9), line, count=1)


def test_segment_holds_the_frames_inside_the_window_and_the_gap_between_them(tmp_path):
    depth, depth20 = fixture_lines(b"depth"), fixture_lines(b"depth20")
    # 갭에서는 두 스트림이 같이 빈다. depth 셋과 같은 번호의 depth20 을 같이 뺀다
    lost = {json.loads(line)["m"]["data"]["u"] for line in depth[100:103]}
    kept_depth20 = [line for line in depth20 if json.loads(line)["m"]["data"]["lastUpdateId"] not in lost]
    write_day(tmp_path, "2026-09-27", "06", depth[:100] + depth[103:], kept_depth20)
    start, end = exchange_time_of(depth[50]), exchange_time_of(depth[400])

    segment = read_segment(tmp_path, start, end)

    exchange_time_by_identifier = {
        record["m"]["data"]["u"]: record["m"]["data"]["E"] for record in map(json.loads, depth)
    }
    expected = parse_depth20(kept_depth20)
    expected_time = np.array([exchange_time_by_identifier[identifier] for identifier in expected.last_update_id])
    inside = (expected_time >= start.timestamp() * 1000) & (expected_time < end.timestamp() * 1000)
    assert segment.exchange_time.tolist() == expected_time[inside].tolist()
    assert segment.last_update_id.tolist() == expected.last_update_id[inside].tolist()
    assert segment.received_at.tolist() == expected.received_at[inside].tolist()
    assert np.array_equal(segment.book, expected.book[inside])
    assert segment.gap_last_exchange_time.tolist() == [json.loads(depth[99])["m"]["data"]["E"]]
    assert segment.gap_next_exchange_time.tolist() == [json.loads(depth[103])["m"]["data"]["E"]]
    assert segment.gap_missed_updates.tolist() == [
        json.loads(depth[103])["m"]["data"]["U"] - json.loads(depth[99])["m"]["data"]["u"] - 1
    ]
    # 갭 안에서 시작하는 구간에는 그 갭이 든다. 갭이 끝난 프레임에서 시작하거나 갭 직전 프레임에서 끝나는 구간에는 안 든다
    edges = [(depth[101], depth[400]), (depth[103], depth[400]), (depth[50], depth[99])]
    windows = [read_segment(tmp_path, exchange_time_of(first), exchange_time_of(last)) for first, last in edges]
    assert [len(window.gap_missed_updates) for window in windows] == [1, 0, 0]


def test_saved_segment_is_read_back_unchanged(tmp_path):
    write_day(tmp_path / "raw", "2026-09-27", "06", fixture_lines(b"depth"), fixture_lines(b"depth20"))
    segment = read_segment(tmp_path / "raw", FIXTURE_START, FIXTURE_END)

    save_segment(segment, tmp_path / "segment.npz")
    loaded = load_segment(tmp_path / "segment.npz")

    for name, array in vars(segment).items():
        assert np.array_equal(getattr(loaded, name), array), name
        assert getattr(loaded, name).dtype == array.dtype, name


def test_frames_delivered_late_into_the_next_day_folder_are_still_found(tmp_path, monkeypatch):
    depth, depth20 = fixture_lines(b"depth"), fixture_lines(b"depth20")
    boundary = json.loads(depth[300])["m"]["data"]["U"]
    on_time = [line for line in depth20 if json.loads(line)["m"]["data"]["lastUpdateId"] < boundary]
    # 늦게 온 프레임은 수신 시각도 다음 날이다. 구간은 수신 시각이 아니라 거래소 시각으로 자르므로 그래도 든다
    late = [received_a_day_later(line) for line in depth20[len(on_time) :]]
    write_day(tmp_path, "2026-09-27", "23", depth[:300], on_time)
    write_day(tmp_path, "2026-09-28", "00", depth[300:], late)
    # 픽스처는 자정 근처가 아니라서, 여유를 하루로 늘려 다음 날 폴더까지 보게 한다. 앞 날짜 폴더는 없다
    monkeypatch.setattr(segment_module, "DELIVERY_MARGIN", timedelta(days=1))

    segment = read_segment(tmp_path, FIXTURE_START, FIXTURE_END)

    expected = parse_depth20(on_time + late)
    for name in ("last_update_id", "received_at", "book"):
        assert np.array_equal(getattr(segment, name), getattr(expected, name)), name


def test_depth20_frame_without_its_depth_frame_is_rejected(tmp_path):
    depth, depth20 = fixture_lines(b"depth"), fixture_lines(b"depth20")
    write_day(tmp_path, "2026-09-27", "06", depth[:100] + depth[103:], depth20)

    with pytest.raises(ValueError, match="거래소 시각을 못 얻는다"):
        read_segment(tmp_path, FIXTURE_START, FIXTURE_END)


def test_depth20_frames_outside_the_depth_frames_read_are_left_out(tmp_path):
    depth, depth20 = fixture_lines(b"depth"), fixture_lines(b"depth20")
    # 수집 중인 파일에서는 depth20 이 depth 보다 몇 프레임 먼저 디스크에 닿는다. 폴더의 첫 프레임도 짝이 앞 날짜 폴더에 있을 수 있다
    write_day(tmp_path, "2026-09-27", "06", depth[2:-3], depth20)
    written = {json.loads(line)["m"]["data"]["u"] for line in depth[2:-3]}

    segment = read_segment(tmp_path, FIXTURE_START, FIXTURE_END)

    expected = parse_depth20([line for line in depth20 if json.loads(line)["m"]["data"]["lastUpdateId"] in written])
    assert 0 < len(expected.last_update_id) < len(depth20)
    assert segment.last_update_id.tolist() == expected.last_update_id.tolist()
    assert np.array_equal(segment.book, expected.book)


def test_window_without_frames_is_rejected(tmp_path):
    write_day(tmp_path, "2026-09-27", "06", fixture_lines(b"depth"), fixture_lines(b"depth20"))

    with pytest.raises(ValueError, match="프레임이 없다"):
        read_segment(tmp_path, FIXTURE_END, FIXTURE_END + timedelta(minutes=5))


def test_export_script_refuses_to_write_under_the_raw_folder(tmp_path, monkeypatch):
    write_day(tmp_path / "raw" / "BTCUSDT", "2026-09-27", "06", fixture_lines(b"depth"), fixture_lines(b"depth20"))
    script = str(PROJECT / "scripts" / "export_replay_segment.py")
    window = ["2026-09-27T06:14:00", "2026-09-27T06:16:00"]
    # 원본 폴더는 기본값처럼 상대 경로로 준다. 출력 경로를 돌려 적어도 원본 폴더 아래 어디든 막아야 한다
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(sys, "argv", [script, *window, "raw-export/a.npz", "--data-root", "raw"])
    runpy.run_path(script, run_name="__main__")
    for refused in ("raw-export/../raw/segments/a.npz", "raw-export/b.npy"):
        monkeypatch.setattr(sys, "argv", [script, *window, refused, "--data-root", "raw"])
        with pytest.raises(SystemExit):
            runpy.run_path(script, run_name="__main__")

    assert len(load_segment(tmp_path / "raw-export" / "a.npz").exchange_time) == 598
    assert not (tmp_path / "raw" / "segments").exists()
    assert not (tmp_path / "raw-export" / "b.npy.npz").exists()


    
def test_export_script_refuses_a_window_longer_than_the_screen_can_take(tmp_path, monkeypatch):
    script = str(PROJECT / "scripts" / "export_replay_segment.py")
    # 화면은 구간을 웹소켓 메시지 하나로 받는다. 하루치는 750MB 라 Streamlit 한도 200MB 를 넘는다. 읽기 전에 막는다
    window = ["2026-09-27T06:00:00", "2026-09-27T08:00:00.001"]
    output = tmp_path / "a.npz"
    monkeypatch.setattr(sys, "argv", [script, *window, str(output), "--data-root", str(tmp_path / "raw")])

    with pytest.raises(SystemExit):
        runpy.run_path(script, run_name="__main__")

    assert not output.exists()