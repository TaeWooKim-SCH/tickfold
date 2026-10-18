from dataclasses import replace
from datetime import date, datetime, timedelta, timezone

import numpy as np

from tickfold.model.inputs import Normalization, deeplob_columns, fit_normalization, normalize, read_days
from tickfold.model.settings import EVALUATION, TRAIN, VALIDATION, Settings
from tickfold.replay.book_replay import BookStatus
from tickfold.replay.comparison import Depth20Comparison, Depth20Line, LevelCounts
from tickfold.replay.depth20 import ASK, BID, PRICE, QUANTITY
from tickfold.replay.restored_day import RestoredDay, save_restored_day
from tickfold.replay.top_levels import TopLevels
from tickfold.replay.valid_mask import ValidColumns

FIRST_DAY = date(2026, 9, 27)
# 학습·검증·평가 하루씩
SMALL = Settings(
    train_days=(FIRST_DAY, FIRST_DAY), validation_days=(FIRST_DAY + timedelta(1),) * 2,
    evaluation_days=(FIRST_DAY + timedelta(2),) * 2,
)
_EMPTY_LINE = Depth20Line(0, 0, LevelCounts(0, 0, 0, 0), 0)


def synthetic_book(mid: np.ndarray, random: np.random.Generator) -> np.ndarray:
    """mid 에서 0.005 떨어진 최우선 호가부터 0.01 간격 20레벨. 가격은 원본 문자열처럼 소수 둘째 자리다."""
    steps = np.arange(20) * 0.01 + 0.005
    book = np.empty((len(mid), 2, 20, 2))
    book[:, BID, :, PRICE] = np.round(mid[:, None] - steps, 2)
    book[:, ASK, :, PRICE] = np.round(mid[:, None] + steps, 2)
    book[..., QUANTITY] = np.round(random.uniform(0.001, 5, size=(len(mid), 2, 20)), 5)
    return book


def write_day(directory, day: date, mid: np.ndarray, random, usable=None, resumed=None, scale=1.0) -> np.ndarray:
    """mid 를 따라 100ms 마다 한 행인 날짜 파일을 쓴다. 수량은 scale 배다. 모델이 안 읽는 칸은 아무 값이다. 쓴 호가창을 돌려준다."""
    rows = len(mid)
    start = int((datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp()) * 1000)
    time = start + 100 * np.arange(rows, dtype=np.int64)
    usable = np.ones(rows, dtype=bool) if usable is None else usable
    book = synthetic_book(mid, random)
    book[..., QUANTITY] *= scale
    status = np.where(usable, BookStatus.SYNCED, BookStatus.AFTER_GAP).astype(np.int8)
    top = TopLevels(time, np.arange(rows, dtype=np.int64), status, book, np.zeros(rows), np.full(rows, 1e9))
    resumed = np.zeros(rows, dtype=bool) if resumed is None else resumed
    valid = ValidColumns(time, resumed, usable, usable)
    depth20 = Depth20Comparison(_EMPTY_LINE, _EMPTY_LINE, 0, 0, 0)
    restored = RestoredDay(start, start + 86_400_000, top, valid, depth20, np.zeros(0, dtype=np.int64), ())
    save_restored_day(restored, directory / f"{day}.npz")
    return book


def random_walk(rows: int, random: np.random.Generator, start: float = 84_000.005) -> np.ndarray:
    return start + 0.01 * np.cumsum(random.integers(-2, 3, size=rows))


def test_ten_level_columns_are_the_first_forty_twenty_level_columns_before_and_after_normalization(tmp_path):
    random = np.random.default_rng(0)
    for offset in range(3):
        write_day(tmp_path, FIRST_DAY + timedelta(offset), random_walk(50, random), random)
    joined = read_days(tmp_path, SMALL)
    twenty_levels = joined.columns
    ten_levels = replace(joined, columns=np.ascontiguousarray(twenty_levels[:, :40]))
    assert np.array_equal(ten_levels.columns, deeplob_columns(_books(tmp_path)[:, :, :10], 10))

    normalization = fit_normalization(joined)
    ten_normalization = fit_normalization(ten_levels)
    normalize(twenty_levels, normalization, SMALL.input_clip)
    normalize(ten_levels.columns, ten_normalization, SMALL.input_clip)

    assert np.array_equal(ten_levels.columns, twenty_levels[:, :40])


def test_days_join_in_order_and_statistics_come_from_usable_training_rows_only(tmp_path):
    random = np.random.default_rng(1)
    usable = np.ones(50, dtype=bool)
    usable[7] = False
    # 84,000.145 언저리는 (매수가 + 매도가) × 100 이 정수보다 조금 작게 계산되는 행이 섞여 있어 버림과 반올림이 갈린다
    books = [write_day(tmp_path, FIRST_DAY, random_walk(50, random, start=84_000.145), random, usable=usable)]
    # 검증·평가 날짜의 수량이 1000배여도 학습 통계에는 안 섞여야 한다
    for offset in (1, 2):
        books.append(write_day(tmp_path, FIRST_DAY + timedelta(offset), random_walk(50, random), random, scale=1000))

    joined = read_days(tmp_path, SMALL)

    assert joined.split.tolist() == [TRAIN] * 50 + [VALIDATION] * 50 + [EVALUATION] * 50
    assert joined.day_row.tolist() == list(range(50)) * 3
    assert joined.day_index.tolist() == [0] * 50 + [1] * 50 + [2] * 50
    best = books[0][:, BID, 0, PRICE] + books[0][:, ASK, 0, PRICE]
    assert joined.mid_half_cents[7] == 0
    assert joined.mid_half_cents[:50][usable].tolist() == [round(value * 100) for value in best[usable]]
    training = deeplob_columns(books[0], 20)[usable].astype(np.float64)
    normalization = fit_normalization(joined)
    assert np.allclose(normalization.mean, training.mean(axis=0))
    assert np.allclose(normalization.standard_deviation, training.std(axis=0))


def test_each_level_is_ask_price_ask_quantity_bid_price_bid_quantity_and_normalize_clips_z_scores():
    book = np.zeros((1, 2, 20, 2))
    book[0, BID, :, PRICE], book[0, ASK, :, PRICE] = 99 - np.arange(20), 101 + np.arange(20)
    book[0, BID, :, QUANTITY], book[0, ASK, :, QUANTITY] = 3, 1
    # mid 가 100 이라 1레벨 가격은 ±0.01, 2레벨은 ±0.02 다
    expected = [0.01, np.log1p(1), -0.01, np.log1p(3), 0.02, np.log1p(1), -0.02, np.log1p(3)]
    assert np.allclose(deeplob_columns(book, 20)[0, :8], expected)

    values = np.array([[-100], [0], [2], [4], [100]], dtype=np.float32)
    normalize(values, Normalization(np.array([2.0]), np.array([2.0])), clip=10)
    assert np.allclose(values[:, 0], [-10, -1, 0, 1, 10])


def _books(directory) -> np.ndarray:
    return np.concatenate([np.load(path)["book"] for path in sorted(directory.glob("????-??-??.npz"))])