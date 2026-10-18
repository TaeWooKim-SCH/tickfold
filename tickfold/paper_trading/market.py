"""날짜별 복원 파일에서 모의 매매가 읽는 최우선 호가와 그 행을 쓸 수 있는지를 꺼내 날짜를 이어 붙인다."""

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from tickfold.replay.restored_day import load_restored_day


@dataclass(frozen=True)
class Market:
    """이어 붙인 날짜들의 행마다 최우선 호가. 날짜 파일의 행을 차례로 이어서, 갭이 아니면 자정을 넘어도 행이 이어진다."""

    days: np.ndarray  # (날짜,) datetime64[D]. 이어 붙인 차례
    day_first_rows: np.ndarray  # (날짜 + 1,) int64. 날짜마다 그날 0행의 자리. 끝 칸은 전체 행 수
    exchange_time: np.ndarray  # (행,) int64. 거래소 시각 E, 밀리초
    best: np.ndarray  # (행, 매수·매도, 가격·수량) float64. 복원 배열의 book[:, :, 0, :]. 동기화 안 된 행은 NaN
    # 학습 표본과 같은 valid 두 컬럼(동기화됨, N번째 레벨이 스냅샷 범위 안)이 다 참인 행. 이 행에서만 들고 닫는다
    tradable: np.ndarray  # (행,) bool
    resumed_after_gap: np.ndarray  # (행,) bool. 이 행 바로 앞에서 순번이 끊겼다


def load_market(days_directory: Path, first_day: date, last_day: date) -> Market:
    """first_day 부터 last_day 까지(둘 다 포함) 날짜 파일을 이어 붙인다. 가운데 하루가 없으면 행이 안 이어지므로 멈춘다."""
    days = [first_day + timedelta(days=offset) for offset in range((last_day - first_day).days + 1)]
    times, bests, tradables, resumed = [], [], [], []
    for day in days:
        restored = load_restored_day(days_directory / f"{day}.npz")
        times.append(restored.top.exchange_time)
        # 하루 book 은 0.55GB 다. 최우선 레벨만 복사해 두고 나머지는 다음 날짜를 읽기 전에 놓는다
        bests.append(restored.top.book[:, :, 0, :].copy())
        tradables.append(restored.valid.synced & restored.valid.inside_snapshot_range)
        resumed.append(restored.valid.resumed_after_gap)
    lengths = [len(time) for time in times]
    day_first_rows = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
    exchange_time = np.concatenate(times)
    # 날짜 차례가 어긋나거나 같은 날짜가 두 번 들면 청산 시각을 찾는 searchsorted 가 조용히 틀린다
    if (np.diff(exchange_time) < 0).any():
        raise ValueError(f"{days_directory} 의 {first_day} ~ {last_day} 거래소 시각이 뒤로 간다")
    return Market(
        np.array(days, dtype="datetime64[D]"), day_first_rows, exchange_time,
        np.concatenate(bests), np.concatenate(tradables), np.concatenate(resumed),
    )