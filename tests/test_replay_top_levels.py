import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from test_replay_book_replay import SECOND, SNAPSHOT, advanced, delta, depth_lines, received_at, store, utc_day
from test_replay_depth20 import fixture_depth20_lines
from tickfold.replay.book_replay import BookReplay, BookStatus, ReplayStep
from tickfold.replay.depth20 import ASK, BID, parse_depth20
from tickfold.replay.top_levels import collect_top_levels


def whole_book_sorted(step: ReplayStep, levels: int) -> np.ndarray:
    """정렬 목록 없이 그 프레임의 dict 를 통째로 정렬해 구한 상위 levels 레벨."""
    expected = np.full((2, levels, 2), np.nan)
    if step.status is BookStatus.SYNCED:
        for side, book_side, highest_first in ((BID, step.book.bids, True), (ASK, step.book.asks, False)):
            prices = sorted(book_side, key=float, reverse=highest_first)[:levels]
            expected[side, : len(prices)] = [[float(price), float(book_side[price])] for price in prices]
    return expected


@pytest.mark.parametrize("levels", [20, 3])
def test_every_frame_holds_its_whole_book_sorted_and_nan_where_unsynced_or_short_of_levels(tmp_path, levels):
    depth, first = depth_lines(), json.loads(SNAPSHOT)
    # 두 번째 스냅샷은 양쪽 10레벨만 준다고 친다. 끼운 뒤 호가창이 바뀌고, 레벨이 20개보다 적은 프레임이 생긴다
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
    # 100 ~ 102번 프레임을 빼서 103번부터 두 번째 스냅샷 전까지 무효다
    replay = BookReplay(store(tmp_path, depth[:100] + depth[103:], snapshots), utc_day(27), utc_day(28))
    replay_steps, expected = replay.steps, []

    # 재생이 내준 호가창을 다음 프레임이 고치기 전에 통째로 정렬해 둔다
    def steps_sorted_on_the_way():
        for step in replay_steps():
            edges = [np.nan, np.nan]
            if step.status is BookStatus.SYNCED and step.book.lowest_known_bid is not None:
                edges = [step.book.lowest_known_bid, step.book.highest_known_ask]
            expected.append((whole_book_sorted(step, levels), step.status, edges))
            yield step

    replay.steps = steps_sorted_on_the_way
    top = collect_top_levels(replay, levels)

    assert np.array_equal(top.book, np.array([book for book, _, _ in expected]), equal_nan=True)
    assert top.status.tolist() == [status for _, status, _ in expected]
    assert (top.status != BookStatus.SYNCED).sum() == 21 + (301 - 103)
    assert np.isnan(top.book[top.status != BookStatus.SYNCED]).all()
    # 10레벨만 준 스냅샷 뒤에는 20칸을 다 못 채운 프레임이 있어야 시험이 그 경우를 본다
    assert np.isnan(top.book[top.status == BookStatus.SYNCED]).any() == (levels == 20)
    top_edges = np.stack([top.lowest_known_bid, top.highest_known_ask], axis=1)
    assert np.array_equal(top_edges, [edges for _, _, edges in expected], equal_nan=True)


def test_top_twenty_from_a_range_after_the_warm_up_equal_the_fixture_depth20_where_last_update_id_is_u(tmp_path):
    depth = depth_lines()
    symbol_directory = store(tmp_path, depth, [(received_at(depth[20]) + SECOND // 2, SNAPSHOT)])
    # 실제 하루는 늘 범위 앞의 스냅샷부터 데운다. 데우기 행이 있어야 행 번호와 순번 컬럼이 어긋나지 않는지 본다
    start = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(milliseconds=delta(depth[100])["E"])

    top = collect_top_levels(BookReplay(symbol_directory, start, utc_day(28)))

    assert top.exchange_time.tolist() == [delta(line)["E"] for line in depth[100:]]
    assert (top.status == BookStatus.SYNCED).all()
    depth20 = parse_depth20(fixture_depth20_lines())
    _, rows, depth20_rows = np.intersect1d(top.final_update_id, depth20.last_update_id, return_indices=True)
    # 마지막 depth 프레임은 픽스처에 짝인 depth20 이 없다
    assert len(rows) == 599 - 100 - 1
    assert (top.book[rows] == depth20.book[depth20_rows]).all()