from dataclasses import replace

import numpy as np

from test_model_inputs import FIRST_DAY, SMALL
from tickfold.model.inputs import JoinedDays
from tickfold.model.labels import DOWN, STATIONARY, UP, baseline_threshold, classes, select_samples, stationary_width
from tickfold.model.settings import EVALUATION, TRAIN, VALIDATION

# 창 5행, horizon 1초(100ms 간격이면 10행), 두 행마다 표본이라 백 행 안에서 모든 경우가 나온다
LABELS = replace(SMALL, horizon_milliseconds=1_000, window_rows=5, sample_every=2)


def joined_rows(time, mid_half_cents, usable=None, resumed=None, split=None) -> JoinedDays:
    """라벨과 표본 고르기가 읽는 칸만 채운 행들. 하루짜리이고 행 번호가 곧 날짜 행 번호다."""
    rows = len(time)
    usable = np.ones(rows, dtype=bool) if usable is None else np.asarray(usable)
    resumed = np.zeros(rows, dtype=bool) if resumed is None else np.asarray(resumed)
    split = np.full(rows, TRAIN, dtype=np.int8) if split is None else np.asarray(split, dtype=np.int8)
    return JoinedDays(
        (FIRST_DAY,), np.zeros(rows, dtype=np.int16), np.arange(rows, dtype=np.int32), split,
        np.asarray(time, dtype=np.int64), np.where(usable, np.asarray(mid_half_cents, dtype=np.int64), 0), usable,
        resumed, np.zeros((rows, 80), dtype=np.float32),
    )


def test_change_is_the_mean_mid_after_now_up_to_the_horizon_in_exchange_time_relative_to_mid_now():
    # 프레임 간격이 고르지 않아서 horizon 안 프레임 수가 행마다 다르다
    time = np.cumsum([0, 100, 100, 300, 50, 650, 100, 100, 900, 100, 100, 100])
    random = np.random.default_rng(2)
    mid = 16_800_001 + 2 * random.integers(-3, 4, size=len(time))
    samples = select_samples(joined_rows(time, mid), LABELS)

    for row in range(len(time)):
        after = (time > time[row]) & (time <= time[row] + LABELS.horizon_milliseconds)
        complete = time[-1] > time[row] + LABELS.horizon_milliseconds  # horizon 뒤 프레임이 있어야 그 안을 다 봤다
        if complete and after.any():
            # float 평균에서 빼면 mid 1,680만에 비해 차이가 작아 자릿수를 잃는다. 정수로 뺀 기준값과 견준다
            count = int(after.sum())
            expected = (int(mid[after].sum()) - count * int(mid[row])) / (count * int(mid[row]))
            assert np.isclose(samples.change[row], expected, rtol=1e-12, atol=0)
        else:
            assert np.isnan(samples.change[row])
        # 모델 없는 기준은 지금 mid 가 입력 창(이 행까지 window_rows 행) mid 평균에서 얼마나 떨어졌나를 본다
        window = mid[max(row - LABELS.window_rows + 1, 0) : row + 1]
        if row >= LABELS.window_rows - 1:
            expected = (len(window) * int(mid[row]) - int(window.sum())) / (len(window) * int(mid[row]))
            assert np.isclose(samples.window_drift[row], expected, rtol=1e-12, atol=0)
        else:
            assert np.isnan(samples.window_drift[row])
    flat = select_samples(joined_rows(time, np.full(len(time), 16_800_001)), LABELS).change
    assert np.all(flat[~np.isnan(flat)] == 0)  # mid 가 그대로면 보합 경계에 걸리지 않게 정확히 0 이다


def test_windows_and_labels_never_reach_across_a_gap_an_unusable_row_or_a_split_edge():
    rows = 100
    time = 100 * np.arange(rows)
    usable, resumed = np.ones(rows, dtype=bool), np.zeros(rows, dtype=bool)
    usable[12] = False  # 재동기화 대기나 범위 밖
    resumed[30] = True  # 29 와 30 사이에서 순번이 끊겼다
    split = np.array([TRAIN] * 70 + [VALIDATION] * 20 + [EVALUATION] * 10)
    samples = select_samples(joined_rows(time, np.full(rows, 16_800_001), usable, resumed, split), LABELS)

    for row in range(rows):
        first = row - LABELS.window_rows + 1
        window = range(first, row + 1)
        window_ok = first >= 0 and all(usable[window]) and not any(resumed[first + 1 : row + 1])
        window_ok = window_ok and len(set(split[window])) == 1
        assert samples.window_ok[row] == window_ok, row
        # 라벨 구간은 지금부터 t + horizon 이하 마지막 행까지이고, 그다음 행 앞에서도 끊기지 않아야 한다
        after = int(np.searchsorted(time, time[row] + LABELS.horizon_milliseconds, side="right"))
        label = range(row, min(after, rows - 1) + 1)
        label_ok = after < rows and all(usable[row:after]) and not any(resumed[row + 1 : after + 1])
        label_ok = label_ok and len(set(split[label])) == 1
        assert np.isnan(samples.change[row]) != label_ok, row
        assert samples.labelled[row] == (window_ok and label_ok and row % LABELS.sample_every == 0), row
    # horizon 이 10행이라 갭 앞 11행, 쓸 수 없는 행 앞 11행, 구간 끝 11행은 라벨이, 그 뒤 4행은 창이 안 된다
    assert np.flatnonzero(samples.labelled).tolist() == [18, *range(34, 59, 2), 74, 76, 78]


def test_up_and_down_lie_beyond_the_absolute_quantile_width_and_the_baseline_picks_its_best_threshold():
    assert np.isclose(stationary_width(np.array([-3.0, 1.0, 2.0]), 1 / 3), 5 / 3)  # |l| 1, 2, 3 의 1/3 분위수
    assert classes(np.array([-2.0, 0.0, 2.0]), 1.0).tolist() == [DOWN, STATIONARY, UP]
    drift = np.array([-2.0, -1.0, 1.0, 2.0, 0.1, -0.1])
    truth = np.array([DOWN, DOWN, UP, UP, STATIONARY, STATIONARY])
    # 문턱 후보 가운데 0.1 이상 1 미만만 여섯을 다 맞힌다
    assert classes(drift, baseline_threshold(drift, truth)).tolist() == truth.tolist()