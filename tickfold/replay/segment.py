"""재생 화면에 쓸 구간 하나를 원본 파일에서 잘라 파일 하나로 내보낸다. 화면은 그 파일만 읽고 data/raw 에는 붙지 않는다."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from tickfold.replay.depth20 import read_depth20_day
from tickfold.replay.depth_sequence import DepthSequence, exchange_time_at, find_gaps, read_depth_sequence_day
from tickfold.replay.files import DELIVERY_MARGIN


@dataclass(frozen=True)
class ReplaySegment:
    """거래소 시각으로 자른 한 구간. 프레임마다 한 행이고, 갭은 행이 없는 자리라서 따로 든다."""

    exchange_time: np.ndarray  # (프레임,) int64. 거래소 시각, 밀리초
    received_at: np.ndarray  # (프레임,) int64. 수신 시각 나노초
    last_update_id: np.ndarray  # (프레임,) int64
    book: np.ndarray  # (프레임, 매수·매도, 레벨, 가격·수량) float64
    gap_last_exchange_time: np.ndarray  # (갭,) 갭 직전 프레임의 거래소 시각, 밀리초
    gap_next_exchange_time: np.ndarray  # (갭,) 갭 직후 프레임의 거래소 시각, 밀리초
    gap_missed_updates: np.ndarray  # (갭,) 놓친 변경 수


def read_segment(symbol_directory: Path, start: datetime, end: datetime) -> ReplaySegment:
    """거래소 시각이 start 이상 end 미만인 depth20 프레임과 그 사이에 걸친 갭을 읽는다."""
    first_day, last_day = (start - DELIVERY_MARGIN).date(), (end + DELIVERY_MARGIN).date()
    candidates = [first_day + timedelta(days=offset) for offset in range((last_day - first_day).days + 1)]
    # 여유로 넣은 앞뒤 날짜는 폴더가 없을 수 있다. 수집을 시작하기 전날이거나 아직 안 온 내일이다
    day_directories = [symbol_directory / str(day) for day in candidates if (symbol_directory / str(day)).is_dir()]
    if not day_directories:
        raise FileNotFoundError(f"{symbol_directory} 에 {first_day} ~ {last_day} 날짜 폴더가 없다")
    # depth20 을 먼저 읽는다. 쓰이고 있는 폴더에서 depth 를 먼저 읽으면 그 사이에 더 쓰인 depth20 프레임이 짝을 못 찾는다
    frames = [read_depth20_day(directory) for directory in day_directories]
    sequences = [read_depth_sequence_day(directory) for directory in day_directories]
    sequence = DepthSequence(
        np.concatenate([part.received_at for part in sequences]),
        np.concatenate([part.exchange_time for part in sequences]),
        np.concatenate([part.first_update_id for part in sequences]),
        np.concatenate([part.final_update_id for part in sequences]),
    )
    last_update_id = np.concatenate([part.last_update_id for part in frames])
    exchange_time, found = exchange_time_at(sequence, last_update_id)
    # 읽은 depth 의 번호 범위 밖은 짝이 아직 안 쓰였거나 옆 날짜 폴더에 있는 프레임이다. 구간에서 60초 넘게 떨어져 있으므로 뺀다.
    # 수집 중인 파일에서는 depth20 이 depth 보다 몇 프레임 먼저 디스크에 닿아서, 읽는 순서만으로는 짝이 다 맞지 않는다
    beyond_depth = (last_update_id < sequence.final_update_id[0]) | (last_update_id > sequence.final_update_id[-1])
    if not (found | beyond_depth).all():
        orphans = (~(found | beyond_depth)).sum()
        raise ValueError(f"depth20 프레임 {orphans}개가 같은 번호의 depth 프레임이 없어 거래소 시각을 못 얻는다")
    rows = np.flatnonzero(found)

    start_time, end_time = int(start.timestamp() * 1000), int(end.timestamp() * 1000)
    inside = (exchange_time >= start_time) & (exchange_time < end_time)
    if not inside.any():
        raise ValueError(f"{start} ~ {end} 에 depth20 프레임이 없다")
    gaps = find_gaps(sequence)
    overlapping = (gaps.next_exchange_time > start_time) & (gaps.last_exchange_time < end_time)
    return ReplaySegment(
        exchange_time[inside],
        np.concatenate([part.received_at for part in frames])[rows[inside]],
        last_update_id[rows[inside]],
        np.concatenate([part.book for part in frames])[rows[inside]],
        gaps.last_exchange_time[overlapping],
        gaps.next_exchange_time[overlapping],
        gaps.missed_updates[overlapping],
    )


def save_segment(segment: ReplaySegment, path: Path) -> None:
    """구간을 압축한 npz 파일 하나로 쓴다."""
    np.savez_compressed(path, **vars(segment))


def load_segment(path: Path) -> ReplaySegment:
    """save_segment 가 쓴 파일을 다시 읽는다."""
    with np.load(path) as archive:
        return ReplaySegment(**{name: archive[name] for name in archive.files})