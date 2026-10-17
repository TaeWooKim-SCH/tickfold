"""종목 하나의 UTC 날짜 하나를 복원해 .npz 하나로 쓰고 읽는다. 학습은 이 파일을 날짜로 읽고, 논문 수치는 재생 없이 이 파일들을 더해 낸다."""

import operator
from dataclasses import dataclass, fields, is_dataclass
from datetime import date, datetime, timedelta, timezone
from functools import reduce
from pathlib import Path

import numpy as np

from tickfold.replay.book_replay import BookReplay
from tickfold.replay.comparison import Depth20Comparison, SnapshotComparison, compare_depth20, compare_each_snapshot
from tickfold.replay.top_levels import TopLevels, collect_top_levels
from tickfold.replay.valid_mask import Coverage, ValidColumns, mark_valid, measure_coverage

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_DAY, _MILLISECOND = timedelta(days=1), timedelta(milliseconds=1)


@dataclass(frozen=True)
class RestoredDay:
    """한 날짜의 상위 20, valid, 대조 개수. 커버리지와 depth20 대조는 이것만으로 다시 내고, REST 대조는 남긴 개수로만 낸다."""

    start: int  # 거래소 시각 범위 시작, 밀리초. 그 날짜 00:00 UTC
    end: int  # 다음 날짜 00:00 UTC. 끝은 제외
    top: TopLevels
    valid: ValidColumns  # exchange_time 은 top 의 것과 같은 배열이다
    depth20: Depth20Comparison  # 그날 행 전부를 견준 결과
    # REST 대조는 호가창 전체가 들어서 파일로 다시 못 낸다. 날짜 중간에서 자르는 창이 고를 수 있게 스냅샷마다 남긴다
    snapshot_rows: np.ndarray  # (스냅샷,) int64. 스냅샷을 끼운 바로 다음 행. top 의 행 번호다
    snapshots: tuple[SnapshotComparison, ...]  # snapshot_rows 와 같은 차례


@dataclass(frozen=True)
class WindowSummary:
    """거래소 시각 창 하나의 대조와 커버리지. 논문 4장의 표와 72시간 판정이 읽는다."""

    rows: int  # 창 안의 행
    depth20: Depth20Comparison
    snapshots: SnapshotComparison
    # 저장된 첫 프레임 앞과 끝 프레임 뒤는 모르는 시간이라 커버리지는 그 안으로 줄여 잰다. 수집 첫날과 마지막 날이 그렇다
    coverage_start: int
    coverage_end: int
    coverage: Coverage
    # 창에 걸친 갭마다 (갭 길이, 갭 뒤 첫 동기화 행까지) 밀리초. 저장된 날짜 안에서 다시 동기화되지 않았으면 뒤가 None
    gaps: tuple[tuple[int, int | None], ...]


def restore_day(symbol_directory: Path, day: date) -> RestoredDay:
    """한 날짜를 재생해 상위 20 과 valid 를 뽑아 depth20 과 견주고, 재생을 한 번 더 돌려 REST 스냅샷을 스냅샷마다 견준다."""
    start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    replay = BookReplay(symbol_directory, start, start + _DAY)
    top = collect_top_levels(replay)
    valid = mark_valid(top, replay)
    del replay  # 두 번째 재생이 하루치 depth 줄을 다시 들기 전에 놓는다
    depth20 = compare_depth20(top, valid, symbol_directory)
    # collect_top_levels 가 steps() 를 다 써서 재생을 새로 돈다
    placed = list(compare_each_snapshot(BookReplay(symbol_directory, start, start + _DAY)))
    rows, snapshots = np.array([row for row, _ in placed], dtype=np.int64), tuple(counts for _, counts in placed)
    first, end = ((edge - _EPOCH) // _MILLISECOND for edge in (start, start + _DAY))
    return RestoredDay(first, end, top, valid, depth20, rows, snapshots)


def save_restored_day(day: RestoredDay, path: Path) -> None:
    """압축하지 않은 npz 하나로 쓴다. 학습이 같은 날짜를 여러 번 읽으므로 읽을 때마다 푸는 시간을 안 낸다."""
    counts = _flat(day.depth20, "depth20")
    # 스냅샷이 없는 날에도 칸이 있어야 읽을 때 빈 배열로 돌아온다
    for name in _flat(_empty(SnapshotComparison), "snapshots"):
        counts[name] = [_flat(each, "snapshots")[name] for each in day.snapshots]
    counts = {name: np.asarray(values, dtype=np.int64) for name, values in counts.items()}
    columns = {**vars(day.top), **vars(day.valid)}
    # 쓰다 죽으면 반쯤 쓴 파일이 날짜 이름으로 남고, 다시 쓰던 중이면 앞서 쓴 온전한 파일까지 잃는다. 다 쓴 뒤 이름을 바꾼다
    partial = path.with_suffix(".partial.npz")
    np.savez(partial, start=day.start, end=day.end, snapshot_rows=day.snapshot_rows, **columns, **counts)
    partial.replace(path)


def load_restored_day(path: Path) -> RestoredDay:
    """save_restored_day 가 쓴 파일을 다시 읽는다."""
    with np.load(path) as archive:
        arrays = {name: archive[name] for name in archive.files}
    top, valid = (kind(*(arrays[field.name] for field in fields(kind))) for kind in (TopLevels, ValidColumns))
    depth20 = _built(Depth20Comparison, "depth20", lambda name: int(arrays[name]))
    rows = arrays["snapshot_rows"]
    snapshots = tuple(
        _built(SnapshotComparison, "snapshots", lambda name: int(arrays[name][index])) for index in range(len(rows))
    )
    return RestoredDay(int(arrays["start"]), int(arrays["end"]), top, valid, depth20, rows, snapshots)


def summarize_window(days_directory: Path, start: int, end: int, symbol_directory: Path) -> WindowSummary:
    """저장된 날짜 파일만으로 거래소 시각 start 이상 end 미만(밀리초)을 잰다. 재생은 안 한다.

    창에 다 든 날짜는 저장한 depth20 개수를 더하고, 일부만 걸친 날짜는 행을 잘라 원본 depth20 과 다시 견준다. REST 는 끼운 행이
    창 안인 스냅샷만 더한다. 커버리지는 창 앞뒤 날짜 파일이 있으면 그 끝 행과 첫 행을 붙여 잰다.
    """
    first, last = ((_EPOCH + edge * _MILLISECOND).date() for edge in (start, end - 1))
    stored = {date.fromisoformat(path.stem): path for path in days_directory.glob("????-??-??.npz")}
    # 가운데 날짜가 빠진 채 이어 붙이면 그날 하루가 앞 날짜 끝 행 몫으로 세어진다
    missing = [str(day) for day in _days(first, last) if day not in stored]
    if missing:
        raise FileNotFoundError(f"{days_directory} 에 창이 걸친 날짜 파일 {len(missing)}개가 없다: {missing[:3]}")
    valid_parts, depth20, snapshots, rows = [], [], [], 0
    for day in (day for day in _days(first - _DAY, last + _DAY) if day in stored):
        restored = load_restored_day(stored[day])
        inside = (restored.top.exchange_time >= start) & (restored.top.exchange_time < end)
        if start <= restored.start and restored.end <= end:
            depth20.append(restored.depth20)
        elif inside.any():
            sliced = (_rows(restored.top, inside), _rows(restored.valid, inside))
            depth20.append(compare_depth20(*sliced, symbol_directory))
        snapshots += [counts for counts, row in zip(restored.snapshots, restored.snapshot_rows) if inside[row]]
        valid_parts.append(restored.valid)
        rows += int(inside.sum())
    valid = ValidColumns(*map(np.concatenate, zip(*(vars(part).values() for part in valid_parts))))
    if start < valid.exchange_time[0] and valid.resumed_after_gap[0]:  # 창 시작을 첫 행으로 줄이면 이 갭이 조용히 빠진다
        raise FileNotFoundError(f"{days_directory} 에 {first - _DAY} 파일이 없는데 {first} 첫 행이 갭 뒤다. 앞 날짜도 복원한다")
    edges = max(start, int(valid.exchange_time[0])), min(end, int(valid.exchange_time[-1]))
    depth20_sum = reduce(operator.add, depth20, _empty(Depth20Comparison))
    snapshot_sum = reduce(operator.add, snapshots, _empty(SnapshotComparison))
    coverage = measure_coverage(valid, *edges)
    return WindowSummary(rows, depth20_sum, snapshot_sum, *edges, coverage, _gaps(valid, *edges))


def _gaps(valid: ValidColumns, start: int, end: int) -> tuple[tuple[int, int | None], ...]:
    time = valid.exchange_time
    resumed = np.flatnonzero(valid.resumed_after_gap)
    # 갭은 앞 프레임부터 갭 뒤 첫 프레임까지다. 창 시작에 끝난 갭은 창 안 시간이 0 이라 뺀다
    resumed = resumed[(time[resumed] > start) & (time[resumed - 1] < end)]
    synced_rows = np.flatnonzero(valid.synced)
    first_synced = np.searchsorted(synced_rows, resumed)
    return tuple(
        (int(time[row] - time[row - 1]), int(time[synced_rows[at]] - time[row]) if at < len(synced_rows) else None)
        for row, at in zip(resumed.tolist(), first_synced.tolist())
    )


def _rows(columns, selection: np.ndarray):
    return type(columns)(*(getattr(columns, field.name)[selection] for field in fields(columns)))


def _flat(counts, prefix: str) -> dict[str, int]:
    # 칸 이름에 필드 경로를 적어 두면 npz 를 연 사람이 뜻을 읽고, 필드 차례가 바뀌어도 그대로 읽힌다
    flat = {}
    for field in fields(counts):
        value, name = getattr(counts, field.name), f"{prefix}.{field.name}"
        flat.update(_flat(value, name) if is_dataclass(value) else {name: value})
    return flat


def _built(kind, prefix: str, value_of):
    """_flat 을 거꾸로 한다. value_of 가 칸 이름마다 정수를 준다."""
    names = (f"{prefix}.{field.name}" for field in fields(kind))
    return kind(*(_built(field.type, name, value_of) if is_dataclass(field.type) else value_of(name)
                  for field, name in zip(fields(kind), names)))


def _empty(kind):
    return _built(kind, "", lambda name: 0)


def _days(first: date, last: date) -> list[date]:
    return [first + timedelta(days=offset) for offset in range((last - first).days + 1)]