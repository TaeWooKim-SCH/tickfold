"""식 3 라벨과 보합 폭, 표본 고르기, 모델 없는 기준. 입력 창과 라벨 구간이 갭·쓸 수 없는 행·구간 경계를 넘는 표본은 버린다."""

from dataclasses import dataclass

import numpy as np

from tickfold.model.inputs import JoinedDays
from tickfold.model.settings import Settings

DOWN, STATIONARY, UP = 0, 1, 2


@dataclass(frozen=True)
class Samples:
    """행마다 라벨의 상대 변화와 쓸 수 있는지. 학습·검증·평가 표본과 예측을 저장할 행이 여기서 나온다."""

    change: np.ndarray  # (행,) float64. 식 3 의 l = (m+ − m(t)) / m(t). 라벨 구간이 온전하지 않은 행은 NaN
    window_ok: np.ndarray  # (행,) bool. 이 행에서 끝나는 입력 창이 한 덩어리 안이다. 예측은 이 행마다 낸다
    labelled: np.ndarray  # (행,) bool. 창과 라벨 구간이 한 덩어리 안이고 날짜 행 번호가 sample_every 의 배수다
    window_drift: np.ndarray  # (행,) float64. (지금 mid − 입력 창 mid 평균) / 지금 mid. 모델 없는 기준이 읽는다


def select_samples(joined: JoinedDays, settings: Settings) -> Samples:
    """행마다 식 3 의 l 을 재고 창과 라벨 구간이 온전한지 가린다.

    덩어리는 쓸 수 있는 행이 갭 없이 같은 구간 안에서 이어진 것이다. 창은 끝 행까지 window_rows 행이 한 덩어리여야 한다. 라벨
    구간은 지금 행부터 거래소 시각이 t + horizon 이하인 마지막 행까지 한 덩어리이고, 그다음 행이 갭 뒤가 아니고 같은 구간이어야
    한다. 그 사이에 순번이 끊겼으면 horizon 안 프레임을 다 봤다고 못 한다.
    """
    rows = len(joined.exchange_time)
    usable, resumed, split, time = joined.usable, joined.resumed_after_gap, joined.split, joined.exchange_time
    joins_previous = np.zeros(rows, dtype=bool)
    joins_previous[1:] = usable[1:] & usable[:-1] & ~resumed[1:] & (split[1:] == split[:-1])
    piece = np.cumsum(~joins_previous)

    row_numbers = np.arange(rows)
    first = row_numbers - (settings.window_rows - 1)
    window_ok = usable & (first >= 0) & (piece[np.maximum(first, 0)] == piece)

    first_after_now = np.searchsorted(time, time, side="right")
    first_after_horizon = np.searchsorted(time, time + settings.horizon_milliseconds, side="right")
    last_inside, following = np.maximum(first_after_horizon - 1, 0), np.minimum(first_after_horizon, rows - 1)
    label_ok = (first_after_horizon < rows) & (first_after_horizon > first_after_now) & (piece[last_inside] == piece)
    label_ok &= ~resumed[following] & (split[following] == split) & usable

    mid = joined.mid_half_cents
    divisor = np.where(usable, mid, 1)  # 쓸 수 없는 행은 mid 가 0 이라 나누기 전에 바꿔 둔다. 그 행 값은 NaN 으로 덮는다
    total = np.concatenate(([0], np.cumsum(mid)))
    count = np.where(label_ok, first_after_horizon - first_after_now, 1)
    future_total = total[first_after_horizon] - total[first_after_now]
    # 정수로 빼서 나누므로 mid 가 그대로인 구간은 l 이 정확히 0 이다. float 합끼리 빼면 0 근처 오차가 보합 경계를 넘나든다
    change = np.where(label_ok, (future_total - count * mid) / (count * divisor), np.nan)
    window_total = total[row_numbers + 1] - total[np.maximum(first, 0)]
    drift = (settings.window_rows * mid - window_total) / (settings.window_rows * divisor)
    drift = np.where(window_ok, drift, np.nan)
    labelled = window_ok & label_ok & (joined.day_row % settings.sample_every == 0)
    return Samples(change, window_ok, labelled, drift)


def stationary_width(change: np.ndarray, quantile: float) -> float:
    """보합 폭 α. 학습 표본의 |l| 분위수다. 부르는 쪽이 학습 표본만 넘긴다."""
    return float(np.quantile(np.abs(change), quantile))


def classes(change: np.ndarray, width: float) -> np.ndarray:
    """l 이 width 보다 크면 오름, −width 보다 작으면 내림, 나머지는 보합. 모델 없는 기준도 같은 규칙으로 자른다."""
    label = np.full(len(change), STATIONARY, dtype=np.int64)
    label[change > width] = UP
    label[change < -width] = DOWN
    return label


def baseline_threshold(drift: np.ndarray, truth: np.ndarray) -> float:
    """모델 없는 기준의 문턱. 학습 표본에서 |drift| 분위수 50개 가운데 정확도가 가장 높은 값을 고른다(분석 보고서 3.4)."""
    grid = np.unique(np.quantile(np.abs(drift), np.linspace(0, 0.97, 50)))
    accuracy = [np.mean(classes(drift, threshold) == truth) for threshold in grid]
    return float(grid[int(np.argmax(accuracy))])