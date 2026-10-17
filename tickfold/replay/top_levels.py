"""재생한 호가창에서 프레임마다 상위 N 레벨을 뽑아 depth20 과 같은 모양의 배열로 모은다. 대조와 학습이 이 배열을 읽는다."""

from bisect import bisect_left
from dataclasses import dataclass

import numpy as np

from tickfold.replay.book_replay import BookReplay, BookStatus
from tickfold.replay.depth20 import ASK, BID, LEVELS, PRICE, QUANTITY


@dataclass(frozen=True)
class TopLevels:
    """범위 안 depth 프레임마다 한 행. book 이 depth20 배열과 모양이 같아서 같은 순번끼리 바로 견준다."""

    exchange_time: np.ndarray  # (프레임,) int64. 거래소 시각 E, 밀리초
    final_update_id: np.ndarray  # (프레임,) int64. 거래소 필드 u. depth20 의 lastUpdateId 와 같은 번호다
    status: np.ndarray  # (프레임,) int8. BookStatus 값
    book: np.ndarray  # (프레임, 매수·매도, 레벨, 가격·수량) float64. 동기화 안 된 행과 레벨이 모자란 칸은 NaN
    # valid 마스크가 읽는 스냅샷 범위 끝. 재생 루프 밖에서는 다시 구할 수 없어서 같이 담는다
    lowest_known_bid: np.ndarray  # (프레임,) float64. 동기화 안 됐거나 스냅샷이 범위를 모르면 NaN
    highest_known_ask: np.ndarray  # (프레임,) float64. NaN 은 위와 같다


def collect_top_levels(replay: BookReplay, levels: int = LEVELS) -> TopLevels:
    """replay 를 끝까지 돌리며 범위 안 프레임마다 상위 levels 레벨을 뽑는다. replay.steps() 는 한 번만 돌아서 여기서 다 쓴다."""
    frames = replay.end_row - replay.first_row
    book = np.full((frames, 2, levels, 2), np.nan)
    status = np.empty(frames, dtype=np.int8)
    lowest_known_bid, highest_known_ask = np.full(frames, np.nan), np.full(frames, np.nan)
    current_book = None
    for step in replay.steps():
        # 스냅샷을 끼우면 호가창이 새 객체가 된다. 그때만 목록을 새로 만들고, 아니면 바뀐 가격만 고친다
        if step.book is not current_book:
            current_book = step.book
            sides = (_SortedSide(current_book.bids, 1.0), _SortedSide(current_book.asks, -1.0))
        elif step.applied_delta is not None:
            sides[BID].update(step.applied_delta.get("b", []))
            sides[ASK].update(step.applied_delta.get("a", []))
        row = step.row - replay.first_row
        status[row] = step.status
        if step.status is BookStatus.SYNCED:
            sides[BID].fill(book[row, BID])
            sides[ASK].fill(book[row, ASK])
            if current_book.lowest_known_bid is not None:
                lowest_known_bid[row] = current_book.lowest_known_bid
                highest_known_ask[row] = current_book.highest_known_ask
    rows = slice(replay.first_row, replay.end_row)
    exchange_time, final_update_id = replay.sequence.exchange_time[rows], replay.sequence.final_update_id[rows]
    return TopLevels(exchange_time, final_update_id, status, book, lowest_known_bid, highest_known_ask)


class _SortedSide:
    """호가창 한쪽 옆에 두는 정렬된 가격 목록. 매 프레임 dict 를 정렬해 뽑으면 하루 추출만 10분이 넘는다."""

    def __init__(self, levels: dict[str, str], sign: float):
        self._levels = levels
        self._sign = sign
        # 바뀌는 가격은 대부분 최우선 호가 근처라 최우선을 목록 끝에 둔다. 리스트 앞에서 넣고 빼면 뒤 원소를 전부 민다.
        # 매도는 낮은 가격이 최우선이라 부호를 뒤집어 넣는다
        ordered = sorted((sign * float(price), price) for price in levels)
        self._keys = [key for key, _ in ordered]
        self._prices = [price for _, price in ordered]

    def update(self, changed_levels: list[list[str]]) -> None:
        """apply_delta 가 바꾼 가격만 호가창의 지금 상태에 맞춘다."""
        for price, _ in changed_levels:
            key = self._sign * float(price)
            position = bisect_left(self._keys, key)
            listed = position < len(self._keys) and self._keys[position] == key
            if price in self._levels:
                if not listed:
                    self._keys.insert(position, key)
                    self._prices.insert(position, price)
            elif listed:
                del self._keys[position]
                del self._prices[position]

    def fill(self, side_row: np.ndarray) -> None:
        """(레벨, 가격·수량) 칸에 최우선부터 채운다. 레벨이 모자라면 남는 칸은 건드리지 않는다."""
        best_first = slice(None, -len(side_row) - 1, -1)  # 목록 끝에서부터 거꾸로 칸 수만큼
        keys = self._keys[best_first]
        # 부호 뒤집기는 float 에서 정확해서 원본 문자열을 읽은 값이 그대로 돌아온다
        side_row[: len(keys), PRICE] = keys if self._sign > 0 else [-key for key in keys]
        side_row[: len(keys), QUANTITY] = [float(self._levels[price]) for price in self._prices[best_first]]