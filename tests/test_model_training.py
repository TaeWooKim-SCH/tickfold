from dataclasses import replace
from datetime import timedelta

import numpy as np
import pytest

# 모델용 의존성 그룹(model)을 안 깐 환경에서는 이 파일을 건너뛴다
pytest.importorskip("torch")
pytest.importorskip("sklearn")

import torch

from test_model_inputs import FIRST_DAY, write_day
from test_model_labels import LABELS
from tickfold.model.evaluation import scores, split_scores, write_run
from tickfold.model.inputs import read_days
from tickfold.model.settings import EVALUATION, TRAIN, VALIDATION
from tickfold.model.training import Trained, prepare, train

CPU = torch.device("cpu")
# CPU 에서 몇 초 안에 끝나게 모델을 줄인다. lr 을 키워 검증 손실이 다시 오르게 해서 일찍 멈추기와 되돌리기가 돈다
TRAINING = replace(
    LABELS, conv_filters=2, inception_filters=2, lstm_units=4, batch_size=16, learning_rate=0.03, patience=1,
    max_epochs=6,
)


@pytest.fixture(autouse=True)
def hide_gpu(monkeypatch):
    # CPU 로 돌려도 Adam.step 이 GPU 가 있는지 보다가 GPU 메모리 238MiB 를 잡는다. 돌고 있는 학습 옆에서도 시험을 돌린다
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def prepared_days(directory, later_moves=1):
    """학습·검증·평가 하루씩. later_moves 배로 검증·평가 날짜의 mid 를 크게 흔든다."""
    directory.mkdir()
    random = np.random.default_rng(3)
    for offset, moves in enumerate((1, later_moves, later_moves)):
        mid = 84_000.005 + moves * 0.01 * np.cumsum(random.integers(-2, 3, size=200))
        write_day(directory, FIRST_DAY + timedelta(offset), mid, random)
    return prepare(read_days(directory, TRAINING), TRAINING, CPU)


def test_same_seed_twice_gives_the_same_epochs_scores_and_probabilities(tmp_path):
    prepared = prepared_days(tmp_path / "days")
    results = []
    for attempt in ("first", "second"):
        trained = train(prepared, TRAINING, depth=10, seed=0, device=CPU, splits=(VALIDATION, EVALUATION),
                        log=lambda line: None)
        results.append(write_run(prepared, trained, TRAINING, 10, 0, tmp_path / attempt, {"device": "cpu"}))
        for epoch in results[-1]["epochs"]:
            epoch.pop("seconds")

    # 같은 시드면 비트까지 같아야 해서 계산한 값도 == 로 견준다
    assert results[0] == results[1]
    first, second = (np.load(tmp_path / attempt / "predictions.npz") for attempt in ("first", "second"))
    assert all(np.array_equal(first[name], second[name]) for name in first.files)
    evaluation = results[0]["evaluation"]
    assert sum(map(sum, evaluation["confusion_matrix"])) == evaluation["samples"] > 0
    assert list(evaluation["days"]) == [str(FIRST_DAY + timedelta(2))]
    assert evaluation["majority_share"] == max(evaluation["class_shares"])
    # 모의 매매가 읽는 확률은 검증·평가 구간에서 창이 온전한 행마다 날짜와 날짜 행 번호를 달고 있다
    rows = prepared.samples.window_ok & (prepared.split != TRAIN)
    assert first["day_row"].tolist() == prepared.day_row[rows].tolist()
    assert first["probabilities"].shape == (rows.sum(), 3)
    assert np.allclose(first["probabilities"].sum(axis=1), 1, atol=1e-5)
    # 가장 좋던 epoch 뒤로 patience 만큼 더 돌고 멈췄고, 저장한 확률은 그 epoch 의 모델에서 나왔다
    losses, best = [epoch["validation_loss"] for epoch in results[0]["epochs"]], results[0]["best_epoch"]
    assert best == int(np.argmin(losses)) + 1
    assert len(losses) - best == TRAINING.patience and len(losses) < TRAINING.max_epochs
    validation = (first["split"] == VALIDATION) & first["labelled"]
    picked = first["probabilities"][validation, first["label"][validation]]
    assert np.isclose(-np.log(picked).mean(), losses[best - 1], rtol=1e-4)


def test_depth_ten_never_reads_levels_eleven_to_twenty(tmp_path):
    prepared = prepared_days(tmp_path / "days")
    before = train(prepared, TRAINING, depth=10, seed=0, device=CPU, log=lambda line: None)
    prepared.columns[:, 40:] = float("nan")
    after = train(prepared, TRAINING, depth=10, seed=0, device=CPU, log=lambda line: None)
    assert np.array_equal(before.probabilities, after.probabilities)


def test_stationary_width_baseline_threshold_and_statistics_come_from_training_samples_only(tmp_path):
    calm, wild = prepared_days(tmp_path / "calm"), prepared_days(tmp_path / "wild", later_moves=50)

    assert (calm.stationary_width, calm.baseline_threshold) == (wild.stationary_width, wild.baseline_threshold)
    assert np.array_equal(calm.normalization.mean, wild.normalization.mean)
    for split in (VALIDATION, EVALUATION):
        assert not np.array_equal(calm.label[calm.split == split], wild.label[wild.split == split])
    # 합성 mid 가 84,000 USDT 언저리라 α 를 0.01 USDT 틱으로 바꾸면 α × 84,000 / 0.01 이다
    assert np.isclose(calm.stationary_width_ticks, calm.stationary_width * 84_000 / 0.01, rtol=1e-4)


def test_scores_put_truth_on_rows_and_average_f1_over_classes_not_samples():
    truth, predicted, baseline = np.array([0, 0, 0, 1, 2]), np.array([0, 1, 1, 1, 2]), np.zeros(5, dtype=np.int64)
    result = scores(truth, predicted, baseline)
    assert result["confusion_matrix"] == [[1, 2, 0], [0, 1, 0], [0, 0, 1]]
    assert np.isclose(result["macro_f1"], (0.5 + 0.5 + 1) / 3)  # 클래스마다 F1 0.5, 0.5, 1. 표본 가중이면 0.6
    assert np.isclose(result["baseline_accuracy"], 3 / 5)


def test_each_sample_is_scored_with_the_probabilities_of_its_own_row(tmp_path):
    prepared = prepared_days(tmp_path / "days")
    rows = np.flatnonzero(prepared.samples.window_ok & (prepared.split != TRAIN))
    # 정답 클래스에 확률 1 을 준 예측이라 행이 하나라도 어긋나면 정확도가 1 아래로 내려간다
    perfect = Trained(None, [], 1, (VALIDATION, EVALUATION), rows, np.eye(3, dtype=np.float32)[prepared.label[rows]])
    for split in (VALIDATION, EVALUATION):
        assert np.isclose(split_scores(prepared, perfect, split)["accuracy"], 1)