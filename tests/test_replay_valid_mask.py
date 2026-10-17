import json
from dataclasses import replace

import numpy as np
import pytest

from test_replay_book_replay import (
    SECOND, SNAPSHOT, advanced, delta, depth_lines, one_day_later, received_at, store, utc_day,
)
from tickfold.replay.book_replay import BookReplay
from tickfold.replay.depth20 import ASK, BID, PRICE
from tickfold.replay.top_levels import collect_top_levels
from tickfold.replay.valid_mask import Coverage, ValidColumns, mark_valid, measure_coverage


@pytest.fixture(scope="module")
def replayed(tmp_path_factory):
    """100 ~ 102번 프레임을 빼서 갭을 내고, 갭 뒤에는 양쪽 10레벨만 주는 스냅샷을 300번 프레임 자리에 끼운 재생."""
    depth, first = depth_lines(), json.loads(SNAPSHOT)
    bids, asks = advanced(first, depth[:301])
    second = {
        "lastUpdateId": delta(depth[300])["u"],
        "bids": [[price, bids[price]] for price in sorted(bids, key=float, reverse=True)[:10]],
        "asks": [[price, asks[price]] for price in sorted(asks, key=float)[:10]],
    }
    snapshots = [
        (received_at(depth[20]) + SECOND // 2, SNAPSHOT),
        (received_at(depth[300]) + SECOND // 2, json.dumps(second, separators=(",", ":")).encode()),
    ]
    symbol_directory = store(tmp_path_factory.mktemp("raw"), depth[:100] + depth[103:], snapshots)
    replay = BookReplay(symbol_directory, utc_day(27), utc_day(28))
    top = collect_top_levels(replay)
    return replay, top, mark_valid(top, replay)


def valid_columns(symbol_directory, start, end) -> ValidColumns:
    replay = BookReplay(symbol_directory, start, end)
    return mark_valid(collect_top_levels(replay), replay)


def joined(earlier: ValidColumns, later: ValidColumns, earlier_rows=slice(None), later_rows=slice(None)):
    """두 범위의 컬럼을 잇는다. 날짜 하나를 잴 때 붙이는 앞 날짜 끝 행이나 뒤 날짜 첫 행은 rows 로 고른다."""
    pairs = zip(vars(earlier).values(), vars(later).values())
    return ValidColumns(*(np.concatenate([before[earlier_rows], after[later_rows]]) for before, after in pairs))


def test_only_synced_rows_whose_twentieth_level_is_known_on_both_sides_are_inside_the_snapshot_range(replayed):
    replay, top, valid = replayed

    # 프레임 셋을 빼서 행 번호가 원래보다 3 작다. 첫 스냅샷은 21번째 프레임, 두 번째 스냅샷은 298번 행 앞에 낀다
    assert np.flatnonzero(valid.resumed_after_gap).tolist() == [100]
    assert np.flatnonzero(valid.synced).tolist() == list(range(21, 100)) + list(range(298, 596))
    # 두 번째 스냅샷은 3.4 USDT 폭만 알아서 그 뒤 20번째 레벨은 비었거나 범위 밖 가격이다. 둘 다 있어야 시험이 두 경우를 본다
    twentieth_bid, twentieth_ask = top.book[298:, BID, -1, PRICE], top.book[298:, ASK, -1, PRICE]
    assert np.isnan(twentieth_bid).any() and (twentieth_bid < top.lowest_known_bid[298:]).any()
    assert np.isnan(twentieth_ask).any() and (twentieth_ask > top.highest_known_ask[298:]).any()
    assert np.flatnonzero(valid.inside_snapshot_range).tolist() == list(range(21, 100))

    # 픽스처에는 양쪽이 같이 비거나 같이 벗어난 행뿐이라 한쪽만 그런 행을 범위 안 행에 만들어 넣는다. 범위 끝 가격 그대로면 안이다
    book, lowest, highest = top.book.copy(), top.lowest_known_bid.copy(), top.highest_known_ask.copy()
    book[30, BID, -1, PRICE] = book[31, ASK, -1, PRICE] = lowest[32] = highest[33] = np.nan
    book[34, BID, -1, PRICE], book[35, ASK, -1, PRICE] = lowest[34], highest[35]
    book[36, BID, -1, PRICE] = lowest[36] - 0.01
    edited = mark_valid(replace(top, book=book, lowest_known_bid=lowest, highest_known_ask=highest), replay)
    assert np.flatnonzero(edited.inside_snapshot_range).tolist() == [*range(21, 30), *range(34, 36), *range(37, 100)]


def test_coverage_gives_each_frame_the_time_until_the_next_frame_and_the_time_across_a_gap_to_the_gap(replayed):
    _, _, valid = replayed
    time = valid.exchange_time

    coverage = measure_coverage(valid, int(time[0]), int(time[-1]))

    # 99번 행에서 100번 행까지가 갭이다. 동기화 안 된 행은 0 ~ 20 과 100 ~ 297, 범위 밖 행은 298 부터 끝까지다
    assert coverage == Coverage(
        window_milliseconds=time[-1] - time[0],
        gap_milliseconds=time[100] - time[99],
        unsynced_milliseconds=(time[21] - time[0]) + (time[298] - time[100]),
        outside_range_milliseconds=time[-1] - time[298],
    )


def test_time_across_a_gap_is_gap_even_after_an_unsynced_or_out_of_range_frame_and_each_ratio_drops_one_more_kind():
    # 2026-09-28 06:25 갭은 범위 밖 행 바로 뒤에 났다. 5번 행은 갭 바로 뒤에 스냅샷이 낀 행이다
    valid = ValidColumns(
        exchange_time=np.array([0, 100, 300, 400, 600, 700, 1000]),
        resumed_after_gap=np.array([False, False, True, False, False, True, False]),
        synced=np.array([False, False, False, True, True, True, True]),
        inside_snapshot_range=np.array([False, False, False, True, False, False, False]),
    )

    coverage = measure_coverage(valid, 0, 1000)

    # 갭은 1 → 2 와 4 → 5, 동기화 안 됨은 0 → 1 과 2 → 3, 범위 밖은 5 → 6 이다
    assert coverage == Coverage(
        window_milliseconds=1000,
        gap_milliseconds=200 + 100,
        unsynced_milliseconds=100 + 100,
        outside_range_milliseconds=300,
    )
    assert (coverage.gap_free_ratio, coverage.synced_ratio, coverage.valid_ratio) == pytest.approx((0.7, 0.5, 0.2))


def test_two_adjacent_windows_add_up_to_one_window_whether_split_on_a_frame_between_frames_or_inside_a_gap(replayed):
    _, _, valid = replayed
    time = valid.exchange_time
    # 창 시작이 프레임 사이라서 첫 조각은 창 앞 프레임 몫이다
    start, end = int(time[0]) + 50, int(time[-1])

    whole = measure_coverage(valid, start, end)

    # 차례로 프레임 시각, 동기화 전 프레임 사이, 갭 안, 재동기화 대기 안, 범위 밖 구간 안이다
    for split in (int(time[50]), int(time[10]) + 37, int(time[99]) + 200, int(time[150]) + 1, int(time[400]) + 99):
        assert measure_coverage(valid, start, split) + measure_coverage(valid, split, end) == whole


def test_days_replayed_apart_add_up_to_both_days_replayed_at_once_when_a_gap_crosses_midnight(tmp_path):
    depth, first = depth_lines(), json.loads(SNAPSHOT)
    # 100 ~ 102번 프레임을 빼고 103번부터 하루 뒤로 옮긴다. 28일 재생은 27일 스냅샷부터 데우므로 첫 행이 갭 뒤 행이다
    later = [one_day_later(line) for line in depth[103:]]
    bids, asks = advanced(first, depth[:301])
    second = {"lastUpdateId": delta(depth[300])["u"], "bids": list(bids.items()), "asks": list(asks.items())}
    snapshots = [
        (received_at(depth[20]) + SECOND // 2, SNAPSHOT),
        (received_at(later[300 - 103]) + SECOND // 2, json.dumps(second, separators=(",", ":")).encode()),
    ]
    symbol_directory = store(tmp_path, depth[:100] + later, snapshots)
    day27 = valid_columns(symbol_directory, utc_day(27), utc_day(28))
    day28 = valid_columns(symbol_directory, utc_day(28), utc_day(29))
    both = valid_columns(symbol_directory, utc_day(27), utc_day(29))
    midnight = int(utc_day(28).timestamp()) * 1000
    start, end = int(both.exchange_time[0]), int(both.exchange_time[-1])

    # 날짜 하나는 앞 날짜 끝 행과 뒤 날짜 첫 행을 붙여 잰다
    first_day = measure_coverage(joined(day27, day28, later_rows=slice(None, 1)), start, midnight)
    second_day = measure_coverage(joined(day27, day28, earlier_rows=slice(-1, None)), midnight, end)

    assert first_day.gap_milliseconds == midnight - day27.exchange_time[-1]
    assert second_day.gap_milliseconds == day28.exchange_time[0] - midnight
    assert first_day + second_day == measure_coverage(both, start, end)


def test_window_without_a_frame_at_or_before_its_start_or_at_or_after_its_end_is_refused(replayed):
    _, _, valid = replayed
    time = valid.exchange_time

    # 앞 날짜 끝 행이나 뒤 날짜 첫 행을 안 붙인 날짜 하나가 이렇게 된다. 그대로 세면 모르는 시간을 아는 것처럼 센다
    with pytest.raises(ValueError, match="앞 프레임이 없다"):
        measure_coverage(valid, int(time[0]) - 1, int(time[-1]))
    with pytest.raises(ValueError, match="뒤 프레임이 없다"):
        measure_coverage(valid, int(time[0]), int(time[-1]) + 1)