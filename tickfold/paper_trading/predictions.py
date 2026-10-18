"""모델 갈래가 쓴 예측 파일을 읽고, 그 예측을 사자·팔자 신호로 바꾸는 규칙."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tickfold.paper_trading.market import Market

DOWN, FLAT, UP = 0, 1, 2  # 확률 칸의 차례. 라벨의 내림·보합·오름 차례와 같다(#3)


@dataclass(frozen=True)
class Predictions:
    """평가 구간 시점마다 한 칸. 행 번호가 날짜별 복원 파일의 행이라 모델이 본 호가 그대로 매매한다."""

    day: np.ndarray  # (시점,) datetime64[D]. UTC 날짜
    row: np.ndarray  # (시점,) int64. 그날 복원 파일의 행
    probabilities: np.ndarray  # (시점, 3) float. 내림·보합·오름


def load_predictions(path: Path) -> Predictions:
    """모델 갈래(tickfold/model/evaluation.py 의 write_run)가 쓴 predictions.npz 를 읽는다. 날짜는 days[day_index] 다."""
    with np.load(path) as archive:
        days = archive["days"].astype("datetime64[D]")
        return Predictions(days[archive["day_index"]], archive["day_row"].astype(np.int64), archive["probabilities"])


def random_predictions(market: Market, every_rows: int, seed: int) -> Predictions:
    """모델 예측이 나오기 전에 규칙·체결·손익을 끝까지 돌려 보는 무작위 예측. 쓸 수 있는 행 every_rows 개마다 하나."""
    rows = np.flatnonzero(market.tradable)[::every_rows]
    day_index = np.searchsorted(market.day_first_rows, rows, side="right") - 1
    probabilities = np.random.default_rng(seed).dirichlet(np.ones(3), len(rows))
    return Predictions(market.days[day_index], rows - market.day_first_rows[day_index], probabilities)


def prediction_rows(market: Market, predictions: Predictions) -> tuple[np.ndarray, np.ndarray]:
    """market 의 날짜에 든 예측만 골라 이어 붙인 행 번호(오름차순)와 그 행의 확률로 바꾼다."""
    inside = np.isin(predictions.day, market.days)
    # 날짜가 하나도 안 맞으면 모든 문턱이 거래 0건이 되어 오류 없이 빈 결과가 찍힌다
    if not inside.any():
        raise ValueError(f"{market.days[0]} ~ {market.days[-1]} 에 든 예측이 없다")
    day_index, row = np.searchsorted(market.days, predictions.day[inside]), predictions.row[inside]
    first = market.day_first_rows[day_index]
    if ((row < 0) | (row >= market.day_first_rows[day_index + 1] - first)).any():
        raise ValueError("예측의 행 번호가 그날 복원 파일의 행 수를 넘는다. 다른 복원 파일로 만든 예측이다")
    rows = first + row
    order = np.argsort(rows, kind="stable")
    rows = rows[order]
    # 한 행에 예측이 둘이면 어느 쪽을 따를지 정할 수 없다
    if (np.diff(rows) == 0).any():
        raise ValueError("같은 행에 예측이 둘 이상 있다")
    return rows, predictions.probabilities[inside][order]


def signals_of(market: Market, predictions: Predictions, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """예측을 (행, 쪽) 신호로 바꾼다. 행은 오름차순이고 쪽은 1 사자, -1 팔자다.

    상승 확률이 문턱을 넘으면 사자, 하락 확률이 넘으면 팔자다. 문턱이 0.5 아래라 둘 다 넘으면 큰 쪽이고 같으면 신호가 없다.
    """
    rows, probabilities = prediction_rows(market, predictions)
    up, down = probabilities[:, UP], probabilities[:, DOWN]
    sides = np.where((up > threshold) & (up > down), 1, np.where((down > threshold) & (down > up), -1, 0))
    chosen = sides != 0
    return rows[chosen], sides[chosen]