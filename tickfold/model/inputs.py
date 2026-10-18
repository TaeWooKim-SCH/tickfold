"""날짜별 복원 파일을 거래소 시각 차례로 이어 읽어 DeepLOB 열 배치로 바꾸고 정규화한다. 라벨과 학습이 이 행 번호를 같이 쓴다."""

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from tickfold.model.settings import TRAIN, Settings
from tickfold.replay.depth20 import ASK, BID, LEVELS, PRICE, QUANTITY
from tickfold.replay.restored_day import load_restored_day

# 가격이 0.01 USDT 단위(BTCUSDT tickSize)라 매수가 + 매도가에 100 을 곱하면 정수다. mid 를 0.005 단위 정수로 들고 라벨을 정확히 센다
PRICE_STEPS_PER_UNIT = 100


@dataclass(frozen=True)
class JoinedDays:
    """날짜 파일을 이어 붙인 행. 날짜 사이도 행이 이어져 창과 라벨이 자정을 넘을 수 있고, 갭은 resumed_after_gap 이 가른다."""

    days: tuple[date, ...]
    day_index: np.ndarray  # (행,) int16. days 의 번호
    day_row: np.ndarray  # (행,) int32. 그 날짜 파일 안의 행 번호. 저장한 예측을 모의 매매가 복원 파일 행에 맞출 때 쓴다
    split: np.ndarray  # (행,) int8. TRAIN, VALIDATION, EVALUATION
    exchange_time: np.ndarray  # (행,) int64. 거래소 시각 E, 밀리초
    mid_half_cents: np.ndarray  # (행,) int64. (최우선 매수가 + 최우선 매도가) × 100. 쓸 수 없는 행은 0
    usable: np.ndarray  # (행,) bool. 동기화됐고 20번째 레벨이 스냅샷 범위 안(valid 두 컬럼)
    resumed_after_gap: np.ndarray  # (행,) bool. 이 행 바로 앞에서 순번이 끊겼다
    columns: np.ndarray  # (행, 4 × 20) float32. deeplob_columns 의 값. normalize 가 그 자리에서 바꾼다


@dataclass(frozen=True)
class Normalization:
    """학습 구간 쓸 수 있는 행의 열마다 평균과 표준편차. 검증·평가 행도 이 값으로 바꿔 학습 밖 통계가 섞이지 않는다."""

    mean: np.ndarray  # (열,) float64
    standard_deviation: np.ndarray  # (열,) float64


def deeplob_columns(book: np.ndarray, levels: int) -> np.ndarray:
    """(행, 매수·매도, 레벨, 가격·수량) 을 (행, 4 × levels) 로. 레벨마다 매도가, 매도량, 매수가, 매수량(원 논문 식 6).

    레벨이 바깥 차례라 levels 를 줄인 열은 늘린 열의 앞부분과 같다. 가격은 그 행 mid 기준 상대값(가격 / mid − 1)이다. 원값은
    날이 바뀌면 학습 분포 밖으로 나간다. 수량은 몇 BTC 짜리 레벨 몇 개가 나머지를 덮지 않게 log1p 로 줄인다.
    """
    mid = (book[:, BID, 0, PRICE] + book[:, ASK, 0, PRICE]) / 2
    columns = np.empty((len(book), levels, 4), dtype=np.float32)
    for position, (side, value) in enumerate(((ASK, PRICE), (ASK, QUANTITY), (BID, PRICE), (BID, QUANTITY))):
        cells = book[:, side, :levels, value]
        columns[:, :, position] = cells / mid[:, None] - 1 if value == PRICE else np.log1p(cells)
    return columns.reshape(len(book), levels * 4)


def read_days(days_directory: Path, settings: Settings) -> JoinedDays:
    """학습 첫 날짜부터 평가 끝 날짜까지 날짜 파일을 차례로 이어 읽는다. 가운데 날짜가 빠지면 이어 붙이지 않고 멈춘다."""
    days = settings.days()
    paths = [days_directory / f"{day}.npz" for day in days]
    # 빠진 날짜를 건너뛰고 붙이면 그 앞뒤 행이 이어진 것처럼 보여 창과 라벨이 하루를 건너뛴다
    missing = [path.name for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{days_directory} 에 날짜 파일 {len(missing)}개가 없다: {missing[:3]}")
    parts = []
    for index, (day, path) in enumerate(zip(days, paths)):
        restored = load_restored_day(path)
        top, valid = restored.top, restored.valid
        rows = len(top.exchange_time)
        usable = valid.synced & valid.inside_snapshot_range
        best = top.book[:, BID, 0, PRICE] + top.book[:, ASK, 0, PRICE]
        mid = np.rint(np.where(usable, best, 0) * PRICE_STEPS_PER_UNIT).astype(np.int64)  # NaN 을 정수로 바꾸면 값이 정해지지 않는다
        parts.append((
            np.full(rows, index, dtype=np.int16), np.arange(rows, dtype=np.int32),
            np.full(rows, settings.split_of(day), dtype=np.int8), top.exchange_time, mid, usable,
            valid.resumed_after_gap, deeplob_columns(top.book, LEVELS),
        ))
        del restored, top  # 하루치 float64 호가창 0.55GB 를 다음 날짜를 읽기 전에 놓는다
    joined = JoinedDays(tuple(days), *(np.concatenate(column) for column in zip(*parts)))
    if np.any(np.diff(joined.exchange_time) <= 0):
        raise ValueError(f"{days_directory} 의 날짜 파일을 이으니 거래소 시각이 늘기만 하지 않는다")
    return joined


def fit_normalization(joined: JoinedDays) -> Normalization:
    """학습 구간의 쓸 수 있는 행만으로 열마다 평균과 표준편차를 잰다."""
    training = joined.columns[(joined.split == TRAIN) & joined.usable]
    mean = training.mean(axis=0, dtype=np.float64)
    standard_deviation = training.std(axis=0, dtype=np.float64)
    if not np.all(standard_deviation > 0):  # 0 으로 나누면 그 열이 전부 inf 나 NaN 이 되어 학습이 조용히 망가진다
        raise ValueError(f"학습 구간에서 값이 하나뿐인 열: {np.flatnonzero(~(standard_deviation > 0)).tolist()}")
    return Normalization(mean, standard_deviation)


def normalize(columns: np.ndarray, normalization: Normalization, clip: float) -> None:
    """열마다 z-score 로 바꾸고 ±clip 에서 자른다. 20일치 열이 5.4GB 라 사본을 만들지 않고 그 자리에서 바꾼다."""
    columns -= normalization.mean.astype(np.float32)
    columns /= normalization.standard_deviation.astype(np.float32)
    np.clip(columns, -clip, clip, out=columns)