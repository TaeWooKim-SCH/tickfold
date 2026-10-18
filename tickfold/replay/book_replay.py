"""저장된 REST 스냅샷과 depth 프레임으로 호가창을 다시 돌린다. 상위 N 배열, valid 마스크, 대조가 이 위에 얹힌다."""

import json
import re
from dataclasses import fields
from datetime import datetime, timedelta, timezone
from enum import IntEnum
from itertools import compress
from pathlib import Path
from typing import Iterator, NamedTuple

import numpy as np

from tickfold.collector.orderbook import OrderBook
from tickfold.replay.depth_sequence import DepthSequence, find_gaps, parse_depth_sequence
from tickfold.replay.files import DELIVERY_MARGIN, read_lines, stream_files, stream_files_between
from tickfold.replay.repeats import first_time_read

# 스냅샷 한 줄은 5,000 + 5,000레벨이라 30만 바이트쯤이다. 자리를 정할 때는 머리만 읽고 json 은 끼울 때 푼다
_SNAPSHOT_HEAD = re.compile(rb'\{"rx":(\d+),"m":\{"lastUpdateId":(\d+),')
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


class BookStatus(IntEnum):
    """프레임 하나를 처리한 뒤 호가창을 믿을 수 있는지, 못 믿으면 왜인지. valid 마스크가 이 값에서 나온다."""

    SYNCED = 0
    NO_SNAPSHOT_YET = 1  # 거슬러 올라가도 스냅샷이 없다. 수집을 막 시작했을 때뿐이다
    AFTER_GAP = 2  # 순번이 끊긴 뒤 아직 스냅샷을 못 만났다


class PlacedSnapshot(NamedTuple):
    """프레임 앞에 끼운 스냅샷. 대조가 끼우기 직전의 호가창과 견준다."""

    snapshot: dict  # REST 응답 그대로. lastUpdateId, bids, asks
    replaced: OrderBook | None  # 같은 순번에서 동기화돼 있던 앞 호가창. 순번이 다르거나 무효였으면 None


class ReplayStep(NamedTuple):
    """범위 안 depth 프레임 하나를 처리한 뒤의 모습. 상위 N 과 valid 마스크가 프레임마다 이것을 읽는다."""

    row: int  # BookReplay.sequence 의 행. 거래소 시각과 순번은 거기 있다
    status: BookStatus
    book: OrderBook  # 스냅샷을 끼우면 새 객체가 된다. 호가창에서 만든 정렬 목록은 객체가 바뀔 때 다시 만든다
    applied_delta: dict | None  # 호가창에 적용한 depth 프레임의 data. 적용하지 않았으면 None
    placed: tuple[PlacedSnapshot, ...]  # 이 프레임을 적용하기 전에 끼운 스냅샷


class BookReplay:
    """거래소 시각 범위 하나의 호가창을 범위 앞의 가장 가까운 스냅샷부터 다시 돌린다. steps() 는 한 번만 돈다."""

    def __init__(self, symbol_directory: Path, start: datetime, end: datetime):
        snapshots = _snapshots_from_warm_up(symbol_directory, start, end)
        read_from = min(start, _EPOCH + timedelta(microseconds=snapshots[0][0] // 1000)) if snapshots else start
        lines, self.sequence = _read_depth(symbol_directory, read_from - DELIVERY_MARGIN, end + DELIVERY_MARGIN)
        exchange_time = self.sequence.exchange_time
        # 범위를 searchsorted 로 자르므로 거래소 시각이 뒤로 가면 범위 밖 프레임이 섞인다
        if (np.diff(exchange_time) < 0).any():
            raise ValueError(f"{symbol_directory}: depth 프레임의 거래소 시각이 뒤로 간다")
        bounds = np.searchsorted(exchange_time, [_milliseconds(start), _milliseconds(end)])
        self.first_row, self.end_row = bounds.tolist()
        if self.first_row == self.end_row:
            raise ValueError(f"{start} ~ {end} 에 depth 프레임이 없다")
        # 수집을 막 시작한 범위는 맨 앞 스냅샷이 범위 안이다. 거기서부터 돌면 그 앞의 범위 프레임이 통째로 빠진다
        self.warm_up_row = self.first_row
        if snapshots:
            snapshot_row = np.searchsorted(self.sequence.final_update_id, snapshots[0][1], side="right")
            self.warm_up_row = min(int(snapshot_row), self.first_row)
        self.gaps = find_gaps(self.sequence)
        self.unplaced_snapshots: list[int] = []  # 순번 사슬에 자리가 없어 못 끼운 스냅샷의 lastUpdateId
        # 뒤 프레임이 먼저 저장된 스냅샷이 있어서 수신 시각이 아니라 순번 순으로 끼운다
        self._snapshots = sorted((last_update_id, line) for _, last_update_id, line in snapshots)
        self._lines = lines[self.warm_up_row : self.end_row]
        self._book = OrderBook()
        self._status = BookStatus.NO_SNAPSHOT_YET

    def steps(self) -> Iterator[ReplayStep]:
        """데우기부터 범위 끝까지 프레임을 차례로 적용하고, 범위 안 프레임마다 ReplayStep 을 내준다."""
        first_update_ids = self.sequence.first_update_id.tolist()
        final_update_ids = self.sequence.final_update_id.tolist()
        resumed_rows = set(self.gaps.resumed_row.tolist())
        snapshots = iter(self._snapshots)
        upcoming = next(snapshots, None)
        for row, line in enumerate(self._lines, self.warm_up_row):
            if row in resumed_rows:
                self._status = BookStatus.AFTER_GAP
            placed = []
            # lastUpdateId 가 L 인 스냅샷은 u > L 인 첫 프레임 앞에 끼운다. 그 프레임이 L + 1 을 품어야 이어진다
            while upcoming is not None and upcoming[0] < final_update_ids[row]:
                if first_update_ids[row] <= upcoming[0] + 1:
                    placed.append(self._place(json.loads(upcoming[1])["m"]))
                else:
                    self.unplaced_snapshots.append(upcoming[0])
                upcoming = next(snapshots, None)
            applied_delta = None
            # 무효인 호가창에 apply_delta 를 부르면 다음 스냅샷까지 버퍼에 쌓이기만 한다
            if self._status is BookStatus.SYNCED:
                applied_delta = json.loads(line)["m"]["data"]
                # 순번 사슬과 스냅샷 자리를 위에서 맞췄으므로 ok 가 아니면 이 루프의 잘못이다
                if (result := self._book.apply_delta(applied_delta)) != "ok":
                    raise RuntimeError(f"순번 {final_update_ids[row]} 프레임이 {result} 로 나왔다")
            if row >= self.first_row:
                yield ReplayStep(row, self._status, self._book, applied_delta, tuple(placed))

    def _place(self, snapshot: dict) -> PlacedSnapshot:
        same_position = self._status is BookStatus.SYNCED and self._book.last_u == snapshot["lastUpdateId"]
        replaced = self._book if same_position else None
        # 앞 호가창을 고치지 않고 새로 만든다. 대조가 끼우기 직전 호가창을 스냅샷과 견줄 수 있게
        self._book = OrderBook()
        self._book.apply_snapshot(snapshot)
        self._status = BookStatus.SYNCED
        return PlacedSnapshot(snapshot, replaced)


def _snapshots_from_warm_up(symbol_directory: Path, start: datetime, end: datetime) -> list[tuple[int, int, bytes]]:
    """범위 앞의 가장 가까운 스냅샷부터 범위 끝 무렵까지를 (수신 시각, lastUpdateId, 줄)로 돌려준다.

    범위 앞의 스냅샷이 나올 때까지 날짜 폴더를 하루씩 거슬러 간다. 수집을 시작하기 전이라 없으면 범위 안의 것만 돌려준다.
    """
    # 수신 시각이 범위 시작보다 DELIVERY_MARGIN 넘게 앞이어야 그 순번의 프레임도 범위 앞이다.
    # 범위 끝보다 DELIVERY_MARGIN 넘게 뒤에 온 스냅샷은 순번도 범위 뒤다
    warm_up_before, received_until = _nanoseconds(start - DELIVERY_MARGIN), _nanoseconds(end + DELIVERY_MARGIN)
    last_day = str((end + DELIVERY_MARGIN).date())
    days = sorted(path.name for path in symbol_directory.iterdir() if path.is_dir() and path.name <= last_day)
    found: list[tuple[int, int, bytes]] = []
    for day in reversed(days):
        found = [head for head in _snapshot_heads(symbol_directory / day) if head[0] <= received_until] + found
        earlier = [index for index, head in enumerate(found) if head[0] < warm_up_before]
        if earlier:
            return found[earlier[-1] :]
    return found


def _snapshot_heads(day_directory: Path) -> list[tuple[int, int, bytes]]:
    heads = []
    for path in stream_files(day_directory, "snapshots"):
        for line in read_lines(path):
            head = _SNAPSHOT_HEAD.match(line)
            if head is None:
                raise ValueError(f"{path}: 스냅샷 줄 모양이 아니다")
            heads.append((int(head[1]), int(head[2]), line))
    # 압축한 파일로 이름을 바꾼 뒤 원본을 지우기 전에 죽으면 같은 스냅샷이 두 파일에 남는다. 두 번 끼우면 대조가 스냅샷을 자기와 견준다
    is_new = first_time_read(np.array([head[1] for head in heads], dtype=np.int64), day_directory)
    return list(compress(heads, is_new))


def _read_depth(symbol_directory: Path, first: datetime, last: datetime) -> tuple[list[bytes], DepthSequence]:
    """수신 시각이 first 의 시간대부터 last 의 시간대까지인 depth 줄과 그 순번을 쓰인 순서로 읽는다."""
    lines_per_file = [read_lines(path) for path in stream_files_between(symbol_directory, "depth", first, last)]
    parts = [parse_depth_sequence(lines) for lines in lines_per_file]
    is_new = first_time_read(np.concatenate([part.final_update_id for part in parts]), symbol_directory)
    sequence = DepthSequence(
        *(np.concatenate([getattr(part, field.name) for part in parts])[is_new] for field in fields(DepthSequence))
    )
    return list(compress((line for lines in lines_per_file for line in lines), is_new)), sequence


def _milliseconds(time: datetime) -> int:
    return (time - _EPOCH) // timedelta(milliseconds=1)


def _nanoseconds(time: datetime) -> int:
    return (time - _EPOCH) // timedelta(microseconds=1) * 1000