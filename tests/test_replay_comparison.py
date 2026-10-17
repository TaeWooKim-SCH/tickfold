import json
import re
from dataclasses import replace

import numpy as np
import pytest
import zstandard

from test_replay_book_replay import (
    SECOND, SNAPSHOT, advanced, delta, depth_lines, received_at, store, utc_day,
)
from test_replay_depth20 import fixture_depth20_lines
from tickfold.collector.writer import RawWriter
from tickfold.replay.book_replay import BookReplay
from tickfold.replay.comparison import (
    Depth20Comparison, Depth20Line, LevelCounts, compare_depth20, compare_snapshots,
)
from tickfold.replay.depth20 import ASK, BID, PRICE
from tickfold.replay.top_levels import collect_top_levels
from tickfold.replay.valid_mask import mark_valid

NO_POINTS = Depth20Line(0, 0, LevelCounts(0, 0, 0, 0), 0)


def store_with_depth20(root, depth, depth20, snapshots):
    symbol_directory = store(root, depth, snapshots)
    writer = RawWriter(root)
    for line in depth20:
        writer.write("BTCUSDT", "depth20", received_at(line), line[line.index(b'"m":') + 4 : -1])
    writer.close()
    return symbol_directory


def with_best_bid_quantity(depth20_line: bytes, quantity: bytes) -> bytes:
    return re.sub(rb'("bids":\[\["\d+\.\d+",")\d+\.\d+', lambda found: found[1] + quantity, depth20_line, count=1)


def moved(line: bytes, milliseconds: int) -> bytes:
    """수신 시각과 거래소 시각을 같이 옮긴다. depth20 에는 거래소 시각이 없다."""
    line = re.sub(rb'^\{"rx":(\d+)', lambda found: b'{"rx":%d' % (int(found[1]) + milliseconds * 10**6), line)
    return re.sub(rb'"E":(\d+)', lambda found: b'"E":%d' % (int(found[1]) + milliseconds), line, count=1)


def as_snapshot(last_update_id: int, bids: dict, asks: dict) -> bytes:
    snapshot = {"lastUpdateId": last_update_id, "bids": list(bids.items()), "asks": list(asks.items())}
    return json.dumps(snapshot, separators=(",", ":")).encode()


def compared(symbol_directory, start, end):
    replay = BookReplay(symbol_directory, start, end)
    top = collect_top_levels(replay)
    depth20 = compare_depth20(top, mark_valid(top, replay), symbol_directory)
    # REST 스냅샷 대조는 재생을 한 번 더 돈다. collect_top_levels 가 steps() 를 다 쓴다
    return depth20, compare_snapshots(BookReplay(symbol_directory, start, end))


def test_one_changed_quantity_in_a_stored_depth20_frame_is_exactly_one_level_at_one_point(tmp_path):
    depth, depth20 = depth_lines(), fixture_depth20_lines()
    depth20[300] = with_best_bid_quantity(depth20[300], b"9.99999000")
    snapshots = [(received_at(depth[20]) + SECOND // 2, SNAPSHOT)]
    symbol_directory = store_with_depth20(tmp_path, depth, depth20, snapshots)

    comparison, _ = compared(symbol_directory, utc_day(27), utc_day(28))

    # 첫 스냅샷 전 21행은 견줄 호가창이 없고, 마지막 depth 프레임은 픽스처에 짝인 depth20 이 없다
    assert comparison == Depth20Comparison(
        inside_range=Depth20Line(577, 1, LevelCounts(577 * 40, 0, 1, 0), 0),
        outside_range=NO_POINTS,
        unsynced_points=21,
        unpaired_depth20=0,
        unpaired_rows=1,
    )
    assert comparison.inside_range.point_mismatch_rate == pytest.approx(1 / 577)
    assert comparison.inside_range.level_counts.mismatch_rate == pytest.approx(1 / (577 * 40))


@pytest.mark.parametrize("side", [BID, ASK], ids=["bid", "ask"])
@pytest.mark.parametrize("change, missing, extra", [("drop", 1, 0), ("insert", 0, 1)])
def test_levels_are_matched_by_price_so_ranks_shifted_by_a_dropped_or_inserted_price_are_not_counted(
    tmp_path, side, change, missing, extra
):
    depth = depth_lines()
    snapshots = [(received_at(depth[20]) + SECOND // 2, SNAPSHOT)]
    symbol_directory = store_with_depth20(tmp_path, depth, fixture_depth20_lines(), snapshots)
    replay = BookReplay(symbol_directory, utc_day(27), utc_day(28))
    top = collect_top_levels(replay)
    book = top.book.copy()
    levels, deeper = book[300, side], -1.0 if side == BID else 1.0
    if change == "drop":
        # 6번째 가격을 빼면 아래 14레벨이 한 칸씩 올라오고 거래소 20번째보다 깊은 21번째가 들어온다. 순위끼리면 15레벨이 다르다
        book[300, side] = np.concatenate([levels[:5], levels[6:], [[levels[-1, PRICE] + deeper, 1.0]]])
    else:
        # 거래소에 없는 가격을 4번째 다음에 끼우면 아래가 한 칸씩 밀려 거래소 20번째가 우리 21번째가 된다
        between = (levels[3, PRICE] + levels[4, PRICE]) / 2
        book[300, side] = np.concatenate([levels[:4], [[between, 1.0]], levels[4:-1]])
    top = replace(top, book=book)

    comparison = compare_depth20(top, mark_valid(top, replay), symbol_directory)

    assert comparison.inside_range.mismatched_points == 1
    assert comparison.inside_range.level_counts == LevelCounts(577 * 40, missing, 0, extra)
    assert comparison.inside_range.level_counts.mismatch_rate == pytest.approx(1 / (577 * 40))


def test_out_of_range_rows_and_unsynced_rows_are_counted_on_their_own_lines(tmp_path):
    depth, first = depth_lines(), json.loads(SNAPSHOT)
    # 100 ~ 102번 프레임을 빼서 갭을 내고, 갭 뒤에는 양쪽 10레벨만 주는 스냅샷을 300번 프레임 자리에 끼운다
    bids, asks = advanced(first, depth[:301])
    best_ten_bids = {price: bids[price] for price in sorted(bids, key=float, reverse=True)[:10]}
    best_ten_asks = {price: asks[price] for price in sorted(asks, key=float)[:10]}
    snapshots = [
        (received_at(depth[20]) + SECOND // 2, SNAPSHOT),
        (received_at(depth[300]) + SECOND // 2, as_snapshot(delta(depth[300])["u"], best_ten_bids, best_ten_asks)),
    ]
    symbol_directory = store_with_depth20(tmp_path, depth[:100] + depth[103:], fixture_depth20_lines(), snapshots)

    comparison, _ = compared(symbol_directory, utc_day(27), utc_day(28))

    # 행 번호는 프레임 셋을 빼서 3 작다. 21 ~ 99행만 범위 안이고, 298행부터는 동기화됐지만 범위 밖이다
    assert comparison.inside_range == Depth20Line(79, 0, LevelCounts(79 * 40, 0, 0, 0), 0)
    assert comparison.unsynced_points == 21 + (298 - 100)
    # 뺀 프레임 셋의 depth20 은 짝이 없고, 마지막 행은 픽스처에 짝인 depth20 이 없다
    assert (comparison.unpaired_depth20, comparison.unpaired_rows) == (3, 1)
    outside = comparison.outside_range
    assert outside.points == 596 - 298 - 1 and outside.mismatched_points > 0
    # 우리가 아는 레벨은 다 맞고, 빠진 것은 전부 10레벨 스냅샷이 모르는 가격이다
    assert outside.level_counts.missing == outside.missing_beyond_known_range > 0
    assert (outside.level_counts.quantity_differences, outside.level_counts.extra) == (0, 0)
    assert comparison.whole == comparison.inside_range + outside


def test_snapshot_compared_with_the_book_it_replaces_counts_each_difference_once_on_the_line_of_its_price(tmp_path):
    depth, first = depth_lines(), json.loads(SNAPSHOT)
    bids, asks = advanced(first, depth[:301])
    lowest_known_bid, highest_known_ask = float(first["bids"][-1][0]), float(first["asks"][-1][0])
    # 매수는 500레벨에서 자른다. 그보다 깊은 호가창 레벨은 스냅샷이 말하지 않은 곳이라 남는 레벨이 아니다
    bids = {price: bids[price] for price in sorted(bids, key=float, reverse=True)[:500]}
    best_bid, tenth_bid = sorted(bids, key=float, reverse=True)[:10:9]
    bids[best_bid] = "9.99999000"  # 수량 차이
    del bids[tenth_bid]  # 호가창에만 있다
    bids["84599.99500000"] = "1.00000000"  # 범위 안인데 호가창에 없다
    asks["86123.45000000"] = "1.00000000"  # 첫 스냅샷이 아는 범위 밖이라 호가창이 모른다
    second = as_snapshot(delta(depth[300])["u"], bids, asks)
    snapshots = [(received_at(depth[20]) + SECOND // 2, SNAPSHOT), (received_at(depth[300]) + SECOND // 2, second)]
    symbol_directory = store_with_depth20(tmp_path, depth, fixture_depth20_lines(), snapshots)

    _, comparison = compared(symbol_directory, utc_day(27), utc_day(28))

    beyond_bids = sum(float(price) < lowest_known_bid for price in bids)
    beyond = beyond_bids + sum(float(price) > highest_known_ask for price in asks)
    # 첫 스냅샷은 앞 호가창이 동기화되기 전에 끼워서 견줄 것이 없다
    assert (comparison.compared_snapshots, comparison.uncomparable_snapshots) == (1, 1)
    assert comparison.inside_range == LevelCounts(len(bids) + len(asks) - beyond, 1, 1, 1)
    assert comparison.outside_range == LevelCounts(beyond, 1, 0, 0)
    assert comparison.whole == LevelCounts(len(bids) + len(asks), 2, 1, 1)


def test_depth20_frames_seen_in_two_files_are_compared_once(tmp_path):
    depth = depth_lines()
    snapshots = [(received_at(depth[20]) + SECOND // 2, SNAPSHOT)]
    symbol_directory = store_with_depth20(tmp_path, depth, fixture_depth20_lines(), snapshots)
    compressed = symbol_directory / "2026-09-27" / "06.depth20.ndjson.zst"
    # 압축한 파일로 이름을 바꾼 뒤 원본을 지우기 전에 죽으면 같은 프레임이 원본에도 남는다
    compressed.with_suffix("").write_bytes(zstandard.ZstdDecompressor().decompress(compressed.read_bytes()))

    comparison, _ = compared(symbol_directory, utc_day(27), utc_day(28))

    assert (comparison.whole.points, comparison.unsynced_points, comparison.unpaired_depth20) == (577, 21, 0)


def test_two_days_compared_apart_add_up_to_both_days_compared_at_once(tmp_path):
    depth, depth20, first = depth_lines(), fixture_depth20_lines(), json.loads(SNAPSHOT)
    # 300번 프레임이 28일 자정에 오게 옮긴다. 두 날짜 모두 여유로 자정 건너편 depth20 파일까지 읽는다
    offset = int(utc_day(28).timestamp() * 1000) - delta(depth[300])["E"]
    # 날마다 depth20 수량 하나를 바꾸고, 28일 스냅샷은 상위 20 밖 수량 하나가 다르다.
    # 스냅샷의 틀린 수량은 끼운 뒤 호가창으로 들어가서 상위 20 안이면 depth20 대조까지 바뀐다
    depth20[200] = with_best_bid_quantity(depth20[200], b"9.99999000")
    depth20[400] = with_best_bid_quantity(depth20[400], b"9.99999000")
    later_bids, later_asks = advanced(first, depth[:451])
    later_bids[sorted(later_bids, key=float, reverse=True)[99]] = "9.99999000"
    snapshots = [
        (received_at(depth[20]), SNAPSHOT),
        (received_at(depth[150]), as_snapshot(delta(depth[150])["u"], *advanced(first, depth[:151]))),
        (received_at(depth[450]), as_snapshot(delta(depth[450])["u"], later_bids, later_asks)),
    ]
    snapshots = [(received + SECOND // 2 + offset * 10**6, snapshot) for received, snapshot in snapshots]
    moved_depth, moved_depth20 = [moved(line, offset) for line in depth], [moved(line, offset) for line in depth20]
    # 27일 마지막 행의 depth20 이 28일 00시 파일로 늦게 왔다. 읽는 범위에 여유가 없으면 27일이 이 짝을 잃는다
    moved_depth20[299] = moved(moved_depth20[299], 1000)
    symbol_directory = store_with_depth20(tmp_path, moved_depth, moved_depth20, snapshots)

    first_depth20, first_snapshots = compared(symbol_directory, utc_day(27), utc_day(28))
    second_depth20, second_snapshots = compared(symbol_directory, utc_day(28), utc_day(29))
    both_depth20, both_snapshots = compared(symbol_directory, utc_day(27), utc_day(29))

    assert (first_depth20.whole.mismatched_points, second_depth20.whole.mismatched_points) == (1, 1)
    assert first_depth20 + second_depth20 == both_depth20
    # 28일 재생은 27일 스냅샷부터 데운다. 데우는 동안 끼운 스냅샷은 27일 몫이라 28일에는 안 센다
    assert (first_snapshots.compared_snapshots, second_snapshots.compared_snapshots) == (1, 1)
    assert both_snapshots.inside_range.quantity_differences == 1
    assert first_snapshots + second_snapshots == both_snapshots