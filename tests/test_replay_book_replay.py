import json
import re
from datetime import datetime, timezone
from pathlib import Path

import zstandard

from tickfold.collector.writer import RawWriter
from tickfold.replay.book_replay import BookReplay, BookStatus

FIXTURES = Path(__file__).parent / "fixtures"
SNAPSHOT = (FIXTURES / "btcusdt_snapshot.json").read_bytes()
SECOND = 10**9


def utc_day(day: int) -> datetime:
    return datetime(2026, 9, day, tzinfo=timezone.utc)


def depth_lines() -> list[bytes]:
    return [line for line in (FIXTURES / "btcusdt_combined.ndjson").read_bytes().splitlines() if b"@depth@" in line]


def received_at(line: bytes) -> int:
    return int(re.match(rb'\{"rx":(\d+)', line)[1])


def delta(line: bytes) -> dict:
    return json.loads(line)["m"]["data"]


def one_day_later(line: bytes) -> bytes:
    line = re.sub(rb'^\{"rx":(\d+)', lambda found: b'{"rx":%d' % (int(found[1]) + 86_400 * SECOND), line)
    return re.sub(rb'"E":(\d+)', lambda found: b'"E":%d' % (int(found[1]) + 86_400_000), line, count=1)


def store(root: Path, depth: list[bytes], snapshots: list[tuple[int, bytes]]) -> Path:
    """수집기의 RawWriter 로 쓴다. 복원기가 읽는 것은 이 writer 가 쓴 파일이다."""
    writer = RawWriter(root)
    for line in depth:
        writer.write("BTCUSDT", "depth", received_at(line), line[line.index(b'"m":') + 4 : -1])
    for received, snapshot in snapshots:
        writer.write_snapshot("BTCUSDT", received, snapshot)
    writer.close()
    return root / "BTCUSDT"


def advanced(snapshot: dict, depth: list[bytes]) -> tuple[dict, dict]:
    """스냅샷에 그 뒤 프레임의 레벨을 차례로 덮어쓴 호가창. 수량은 절대값이고 0 은 레벨 삭제다."""
    bids, asks = dict(snapshot["bids"]), dict(snapshot["asks"])
    for line in depth:
        if delta(line)["u"] <= snapshot["lastUpdateId"]:
            continue
        for side, levels in ((bids, delta(line)["b"]), (asks, delta(line)["a"])):
            for price, quantity in levels:
                if float(quantity) == 0:
                    side.pop(price, None)
                else:
                    side[price] = quantity
    return bids, asks


def test_fixture_replay_ends_at_the_snapshot_advanced_by_later_frames_though_the_snapshot_arrived_last(tmp_path):
    depth, snapshot = depth_lines(), json.loads(SNAPSHOT)
    # 뒤 프레임이 다 저장된 뒤에 온 스냅샷이다. 수신 시각 순으로 끼우면 끝 호가창이 스냅샷 그대로가 된다
    symbol_directory = store(tmp_path, depth, [(received_at(depth[-1]) + SECOND, SNAPSHOT)])

    steps = list(BookReplay(symbol_directory, utc_day(27), utc_day(28)).steps())

    # 픽스처 스냅샷의 lastUpdateId 는 21번째 프레임의 u 다
    assert [step.status for step in steps] == [BookStatus.NO_SNAPSHOT_YET] * 21 + [BookStatus.SYNCED] * 578
    assert [step.placed[0].snapshot for step in steps if step.placed] == [snapshot]
    assert (steps[-1].book.bids, steps[-1].book.asks) == advanced(snapshot, depth)


def test_book_is_invalid_from_a_gap_until_the_next_snapshot_and_deltas_are_not_buffered_meanwhile(tmp_path):
    depth, first = depth_lines(), json.loads(SNAPSHOT)
    # 갭 뒤 스냅샷은 빠진 프레임까지 반영한 상태다. 300번째 프레임 자리에서 받았다고 친다
    bids, asks = advanced(first, depth[:301])
    second = {"lastUpdateId": delta(depth[300])["u"], "bids": list(bids.items()), "asks": list(asks.items())}
    # 순번이 빠진 프레임 안에 든 스냅샷은 그 뒤 프레임과 이어지지 않아 못 쓴다
    in_gap = dict(second, lastUpdateId=delta(depth[101])["u"])
    snapshots = [
        (received_at(depth[20]) + SECOND // 2, SNAPSHOT),
        (received_at(depth[101]) + SECOND // 2, json.dumps(in_gap, separators=(",", ":")).encode()),
        (received_at(depth[300]) + SECOND // 2, json.dumps(second, separators=(",", ":")).encode()),
    ]
    symbol_directory = store(tmp_path, depth[:100] + depth[103:], snapshots)

    replay = BookReplay(symbol_directory, utc_day(27), utc_day(28))
    steps = list(replay.steps())

    # 100 ~ 102번 프레임을 뺐다. 103번부터 무효이고 두 번째 스냅샷을 끼운 301번부터 다시 유효하다
    assert [step.status for step in steps] == (
        [BookStatus.NO_SNAPSHOT_YET] * 21
        + [BookStatus.SYNCED] * (100 - 21)
        + [BookStatus.AFTER_GAP] * (301 - 103)
        + [BookStatus.SYNCED] * (599 - 301)
    )
    assert [step.applied_delta is not None for step in steps] == [step.status is BookStatus.SYNCED for step in steps]
    assert max(len(step.book.buffer) for step in steps) == 0
    assert replay.unplaced_snapshots == [in_gap["lastUpdateId"]]
    assert (steps[-1].book.bids, steps[-1].book.asks) == advanced(first, depth)


def test_frames_before_the_days_first_snapshot_warm_up_from_the_last_snapshot_in_an_earlier_day_folder(tmp_path):
    depth, first = depth_lines(), json.loads(SNAPSHOT)
    # 앞 300프레임은 9월 27일 06시에 두고 나머지를 하루 뒤로 옮긴다. 27일 스냅샷은 18시간 앞이고 28일 첫 스냅샷은 500번째 프레임 자리다
    later_half = [one_day_later(line) for line in depth[300:]]
    bids, asks = advanced(first, depth[:501])
    second = {"lastUpdateId": delta(depth[500])["u"], "bids": list(bids.items()), "asks": list(asks.items())}
    snapshots = [
        (received_at(depth[20]) + SECOND // 2, SNAPSHOT),
        (received_at(later_half[200]) + SECOND // 2, json.dumps(second, separators=(",", ":")).encode()),
    ]
    symbol_directory = store(tmp_path, depth[:300] + later_half, snapshots)

    steps = list(BookReplay(symbol_directory, utc_day(28), utc_day(29)).steps())

    assert [step.status for step in steps] == [BookStatus.SYNCED] * 299
    [(row, [placed])] = [(index, step.placed) for index, step in enumerate(steps) if step.placed]
    assert row == 501 - 300 and placed.snapshot["lastUpdateId"] == second["lastUpdateId"]
    # 대조는 끼우기 직전 호가창을 스냅샷과 견준다. 새로 끼운 호가창과 같은 객체면 스냅샷을 자기와 견준다
    assert (placed.replaced.bids, placed.replaced.asks) == (bids, asks) and placed.replaced is not steps[row].book
    assert (steps[-1].book.bids, steps[-1].book.asks) == advanced(first, depth)


def test_snapshot_seen_in_two_files_is_placed_once(tmp_path):
    depth = depth_lines()
    symbol_directory = store(tmp_path, depth, [(received_at(depth[20]) + SECOND // 2, SNAPSHOT)])
    compressed = symbol_directory / "2026-09-27" / "snapshots.ndjson.zst"
    # 압축한 파일로 이름을 바꾼 뒤 원본을 지우기 전에 죽으면 같은 스냅샷이 원본에도 남는다
    compressed.with_suffix("").write_bytes(zstandard.ZstdDecompressor().decompress(compressed.read_bytes()))

    steps = list(BookReplay(symbol_directory, utc_day(27), utc_day(28)).steps())

    assert [len(step.placed) for step in steps if step.placed] == [1]