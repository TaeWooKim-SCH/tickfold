"""재생한 호가창을 거래소가 따로 준 두 채널과 견준다. depth20 은 같은 연결의 다른 표현이고 REST 스냅샷은 다른 경로다.

이 대조가 보이는 것은 우리 쪽 처리(델타 적용, 순번과 갭 판정, 스냅샷 자리, 상위 20 뽑기)다. 20레벨 아래의 근거는 REST 스냅샷이다.
"""

import math
from collections.abc import Iterator
from dataclasses import dataclass, fields, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from tickfold.replay.book_replay import BookReplay, PlacedSnapshot
from tickfold.replay.depth20 import ASK, BID, LEVELS, PRICE, QUANTITY, Depth20Frames, parse_depth20
from tickfold.replay.files import read_lines, stream_files_between
from tickfold.replay.repeats import first_time_read
from tickfold.replay.segment import DELIVERY_MARGIN
from tickfold.replay.top_levels import TopLevels
from tickfold.replay.valid_mask import ValidColumns

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


class _Summed:
    # 결과를 개수로만 담아 필드끼리 더한다. 날짜별로 낸 것을 더하면 그 날짜들을 한 범위로 낸 것과 같다.
    # Coverage 처럼 astuple 로 더하면 안에 든 LevelCounts 가 튜플로 풀려 이어 붙는다. 필드를 꺼내 안쪽도 같은 + 를 탄다
    def __add__(self, other):
        return type(self)(*(getattr(self, field.name) + getattr(other, field.name) for field in fields(self)))


@dataclass(frozen=True)
class LevelCounts(_Summed):
    """거래소가 준 레벨과 우리 호가창을 가격끼리 맞춘 집합으로 센 개수. 순위끼리 세면 누락 하나가 아래 레벨을 다 밀어 5배로 부푼다."""

    exchange_levels: int  # 거래소가 준 레벨. 레벨 불일치율의 분모
    # 거래소에 있는데 우리에게 없다. depth20 은 우리 상위 20 의 가격 폭 안만 센다. 폭 밖은 우리 상위 20 이 말하지 않은 곳이다
    missing: int
    quantity_differences: int  # 같은 가격인데 수량이 다르다
    # 우리에게 있는데 거래소가 준 가격 폭 안에서 거래소에 없다. 폭 밖은 거래소가 말하지 않은 곳이라 안 센다.
    # 두 쪽에 같은 규칙을 걸어서 가격 하나가 빠지면 누락 1, 거래소에 없는 가격 하나가 끼면 남는 레벨 1 이다
    extra: int

    @property
    def mismatch_rate(self) -> float:
        """레벨 기준 불일치율. 누락, 수량 차이, 남는 레벨을 더해 거래소가 준 레벨로 나눈다."""
        mismatched = self.missing + self.quantity_differences + self.extra
        return mismatched / self.exchange_levels if self.exchange_levels else math.nan


@dataclass(frozen=True)
class Depth20Line(_Summed):
    """depth20 대조의 한 줄. 범위 안과 범위 밖을 따로 담고, 둘을 더하면 전체 줄이다."""

    points: int  # 짝지어 견준 시점(프레임)
    mismatched_points: int  # 레벨이 하나라도 다른 시점
    level_counts: LevelCounts
    # 누락 가운데 가격이 그 행 스냅샷이 아는 범위 밖인 것. 범위를 모르면 밖으로 센다. 관측 한계인지 처리 잘못인지 가른다
    missing_beyond_known_range: int

    @property
    def point_mismatch_rate(self) -> float:
        """시점 기준 불일치율."""
        return self.mismatched_points / self.points if self.points else math.nan


@dataclass(frozen=True)
class Depth20Comparison(_Summed):
    """저장된 depth20 과 재생한 상위 20 을 같은 순번끼리 견준 결과. 불일치율은 범위 안, 범위 밖, 전체 세 줄로 싣는다."""

    inside_range: Depth20Line  # 동기화됐고 20번째 레벨이 양쪽 다 마지막 스냅샷 범위 안인 행(valid 두 컬럼이 다 참)
    outside_range: Depth20Line  # 동기화됐지만 범위 밖인 행. 범위 밖 가격의 레벨은 우리가 모를 수 있다
    unsynced_points: int  # 짝은 있지만 동기화 안 된 행. 견줄 호가창이 없어서 어느 줄의 분모에도 안 넣는다
    unpaired_depth20: int  # 순번이 범위 첫 행과 끝 행 사이인데 같은 순번의 행이 없는 depth20 프레임
    unpaired_rows: int  # 같은 순번의 depth20 이 없는 행

    @property
    def whole(self) -> Depth20Line:
        """전체 줄. 범위 안과 밖을 더한다."""
        return self.inside_range + self.outside_range


@dataclass(frozen=True)
class SnapshotComparison(_Summed):
    """REST 스냅샷을 끼우기 직전 호가창과 견준 결과. 레벨은 가격이 직전 스냅샷이 아는 범위 안인지 밖인지로 나눈다."""

    compared_snapshots: int
    # 끼우기 직전 호가창이 동기화돼 있고 순번이 lastUpdateId 와 같을 때만 견준다(#6). 수집 첫 스냅샷과 갭 뒤 스냅샷이 여기 든다
    uncomparable_snapshots: int
    inside_range: LevelCounts
    outside_range: LevelCounts  # 범위 밖에서 쉬던 주문은 델타에 안 나와서 여기 누락은 관측 한계다

    @property
    def whole(self) -> LevelCounts:
        """전체 줄. 범위 안과 밖을 더한다."""
        return self.inside_range + self.outside_range


_NO_LEVELS = LevelCounts(0, 0, 0, 0)
_NO_POINTS = Depth20Line(0, 0, _NO_LEVELS, 0)


def compare_depth20(top: TopLevels, valid: ValidColumns, symbol_directory: Path) -> Depth20Comparison:
    """top 의 행과 저장된 depth20 을 lastUpdateId == u 인 짝마다 견준다. valid 는 top 에서 낸 컬럼이다.

    depth20 은 수신 시각 한 시간치 파일씩 읽어 그 파일의 짝만 견주므로 메모리가 범위 길이와 상관없다.
    """
    if not len(top.final_update_id) == len(valid.synced) == top.book.shape[0]:
        raise ValueError("top 과 valid 의 행 수가 다르다")
    if top.book.shape[2] != LEVELS:
        raise ValueError(f"depth20 과 견주려면 상위 {LEVELS} 레벨이어야 하는데 {top.book.shape[2]} 레벨이다")
    # 파일은 수신 시각으로 나뉘고 행은 거래소 시각으로 잘렸다. 여유를 두고 읽어야 경계의 짝이 안 빠진다
    first = _EPOCH + timedelta(milliseconds=int(top.exchange_time[0])) - DELIVERY_MARGIN
    last = _EPOCH + timedelta(milliseconds=int(top.exchange_time[-1])) + DELIVERY_MARGIN
    read_identifiers = np.empty(0, dtype=np.int64)
    result = Depth20Comparison(_NO_POINTS, _NO_POINTS, 0, 0, 0)
    for path in stream_files_between(symbol_directory, "depth20", first, last):
        frames = parse_depth20(read_lines(path))
        # 같은 내용이 두 파일로 보이는 때가 있다. 앞 파일까지 읽은 식별자와 이어 붙여야 파일을 넘어 가린다
        read_identifiers = np.concatenate([read_identifiers, frames.last_update_id])
        is_new = first_time_read(read_identifiers, path)[len(read_identifiers) - len(frames.last_update_id) :]
        result += _compare_depth20_file(top, valid, frames, is_new)
    paired = result.whole.points + result.unsynced_points
    return replace(result, unpaired_rows=len(top.final_update_id) - paired)


def compare_snapshots(replay: BookReplay) -> SnapshotComparison:
    """replay 를 끝까지 돌리며 범위 안 프레임 앞에 끼운 스냅샷마다 끼우기 직전 호가창과 견준다. replay.steps() 를 여기서 다 쓴다."""
    result = SnapshotComparison(0, 0, _NO_LEVELS, _NO_LEVELS)
    for _, counts in compare_each_snapshot(replay):
        result += counts
    return result


def compare_each_snapshot(replay: BookReplay) -> Iterator[tuple[int, SnapshotComparison]]:
    """compare_snapshots 를 스냅샷마다 (끼운 바로 다음 행, 결과) 로 내준다. 행은 범위 첫 행부터 센다.

    REST 대조는 호가창 전체가 들어서 저장한 배열로 다시 못 낸다. 날짜 중간에서 자르는 창이 이 행으로 스냅샷을 고른다.
    """
    for step in replay.steps():
        for placed in step.placed:
            yield step.row - replay.first_row, _compare_snapshot(placed)


def _compare_depth20_file(
    top: TopLevels, valid: ValidColumns, frames: Depth20Frames, is_new: np.ndarray
) -> Depth20Comparison:
    identifiers, exchange_book = frames.last_update_id[is_new], frames.book[is_new]
    # 범위 앞뒤 여유로 읽은 프레임은 옆 범위 몫이다. 이렇게 나눠야 날짜별 결과를 더해도 두 번 세지 않는다
    owned = (identifiers >= top.final_update_id[0]) & (identifiers <= top.final_update_id[-1])
    position = np.searchsorted(top.final_update_id, identifiers[owned])
    paired = top.final_update_id[position] == identifiers[owned]
    rows, exchange_book = position[paired], exchange_book[owned][paired]
    synced, inside = valid.synced[rows], valid.inside_snapshot_range[rows]
    inside_line = _compare_rows(top, rows[synced & inside], exchange_book[synced & inside])
    outside_line = _compare_rows(top, rows[synced & ~inside], exchange_book[synced & ~inside])
    return Depth20Comparison(inside_line, outside_line, int((~synced).sum()), int((~paired).sum()), 0)


def _compare_rows(top: TopLevels, rows: np.ndarray, exchange: np.ndarray) -> Depth20Line:
    ours = top.book[rows]
    our_price, exchange_price = ours[..., PRICE], exchange[..., PRICE]  # (시점, 매수·매도, 레벨)
    same_price = exchange_price[..., :, None] == our_price[..., None, :]  # (시점, 매수·매도, 거래소 레벨, 우리 레벨)
    found = same_price.any(axis=3)
    # 한쪽 안에서 가격은 겹치지 않아 맞는 칸은 많아야 하나다. 못 찾은 칸의 argmax 0 은 found 가 가린다
    our_quantity = np.take_along_axis(ours[..., QUANTITY], same_price.argmax(axis=3), axis=2)
    quantity_differs = found & (our_quantity != exchange[..., QUANTITY])
    within_exchange = np.stack(
        [our_price[:, BID] > exchange_price[:, BID, -1:], our_price[:, ASK] < exchange_price[:, ASK, -1:]], axis=1
    )
    extra = within_exchange & ~same_price.any(axis=2)
    # 우리 20번째 칸이 비었으면(NaN) 견주면 거짓이라 거래소 가격을 다 센다. 그때는 우리 호가창 끝까지 본 것이다
    beyond_ours = np.stack(
        [exchange_price[:, BID] < our_price[:, BID, -1:], exchange_price[:, ASK] > our_price[:, ASK, -1:]], axis=1
    )
    missing = ~found & ~beyond_ours
    # NaN 과 견주면 거짓이라 범위를 모르는 행의 누락은 범위 밖으로 센다. valid_mask 의 범위 안 컬럼과 같은 규칙이다
    inside_known = np.stack(
        [
            exchange_price[:, BID] >= top.lowest_known_bid[rows, None],
            exchange_price[:, ASK] <= top.highest_known_ask[rows, None],
        ],
        axis=1,
    )
    mismatched = (missing | quantity_differs | extra).any(axis=(1, 2))
    level_counts = LevelCounts(found.size, int(missing.sum()), int(quantity_differs.sum()), int(extra.sum()))
    return Depth20Line(len(rows), int(mismatched.sum()), level_counts, int((missing & ~inside_known).sum()))


def _compare_snapshot(placed: PlacedSnapshot) -> SnapshotComparison:
    if placed.replaced is None:
        return SnapshotComparison(0, 1, _NO_LEVELS, _NO_LEVELS)
    book = placed.replaced
    # 매도는 부호를 뒤집어 매수와 같은 방향(클수록 안쪽)으로 견준다
    bids = _compare_side(placed.snapshot["bids"], book.bids, 1.0, book.lowest_known_bid)
    asks = _compare_side(placed.snapshot["asks"], book.asks, -1.0, book.highest_known_ask)
    return SnapshotComparison(1, 0, bids[0] + asks[0], bids[1] + asks[1])


def _compare_side(
    snapshot_levels: list[list[str]], book_side: dict[str, str], sign: float, known_edge: float | None
) -> tuple[LevelCounts, LevelCounts]:
    """(직전 스냅샷 범위 안, 밖) 개수. 원본 문자열에서 읽은 float64 끼리라 == 로 견준다."""
    exchange = np.array(snapshot_levels, dtype=np.float64).reshape(-1, 2)
    our_price = np.fromiter(map(float, book_side), np.float64, len(book_side))
    our_quantity = np.fromiter(map(float, book_side.values()), np.float64, len(book_side))
    _, our_index, exchange_index = np.intersect1d(our_price, exchange[:, PRICE], return_indices=True)
    found, quantity_differs = np.zeros((2, len(exchange)), dtype=bool)
    found[exchange_index] = True
    quantity_differs[exchange_index] = our_quantity[our_index] != exchange[exchange_index, QUANTITY]
    # 스냅샷도 5,000레벨에서 잘린다. 그 밖의 우리 레벨은 스냅샷이 말하지 않은 곳이다
    extra = sign * our_price >= np.min(sign * exchange[:, PRICE], initial=np.inf)
    extra[our_index] = False
    # 범위를 모르는 스냅샷 뒤라면 NaN 이 되어 전부 범위 밖으로 센다
    edge = math.nan if known_edge is None else sign * known_edge
    exchange_inside, ours_inside = sign * exchange[:, PRICE] >= edge, sign * our_price >= edge
    return tuple(
        LevelCounts(
            int(exchange_part.sum()),
            int((~found & exchange_part).sum()),
            int((quantity_differs & exchange_part).sum()),
            int((extra & ours_part).sum()),
        )
        for exchange_part, ours_part in ((exchange_inside, ours_inside), (~exchange_inside, ~ours_inside))
    )