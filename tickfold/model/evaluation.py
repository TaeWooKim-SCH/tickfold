"""학습 한 번의 결과를 표로 쓸 숫자와 파일로 남긴다. 결과 JSON 은 논문 5장 표가, predictions.npz 는 모의 매매가 읽는다."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score

from tickfold.model.deeplob import parameter_count
from tickfold.model.labels import DOWN, STATIONARY, UP
from tickfold.model.settings import EVALUATION, TRAIN, VALIDATION, Settings
from tickfold.model.training import Prepared, Trained

_CLASSES = [DOWN, STATIONARY, UP]


def scores(truth: np.ndarray, predicted: np.ndarray, baseline: np.ndarray) -> dict:
    """정확도, macro-F1, 혼동행렬(행이 정답, 열이 예측. 내림·보합·오름 차례), 클래스 비율, 다수 클래스 비율, 모델 없는 기준.

    다수 클래스 비율은 그 구간의 정답에서 가장 많은 클래스의 몫이다. 늘 그 클래스를 찍는 규칙의 정확도라서 정확도 옆에 싣는다.
    """
    shares = np.bincount(truth, minlength=3) / max(len(truth), 1)
    return {
        "samples": len(truth),
        "accuracy": float(np.mean(predicted == truth)) if len(truth) else float("nan"),
        "macro_f1": float(f1_score(truth, predicted, labels=_CLASSES, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(truth, predicted, labels=_CLASSES).tolist(),
        "class_shares": shares.tolist(),
        "predicted_shares": (np.bincount(predicted, minlength=3) / max(len(predicted), 1)).tolist(),
        "majority_share": float(shares.max()),
        "baseline_accuracy": float(np.mean(baseline == truth)) if len(truth) else float("nan"),
        "baseline_macro_f1": float(f1_score(truth, baseline, labels=_CLASSES, average="macro", zero_division=0)),
    }


def split_scores(prepared: Prepared, trained: Trained, split: int) -> dict:
    """구간 하나의 표본 전체와 날짜별 값. 표본은 학습과 같은 규칙(창·라벨 온전, sample_every 배수)으로 고른 행이다."""
    rows = prepared.rows(split, prepared.samples.labelled)
    # 표본 행은 창이 온전해서 전부 예측 행 안에 있다
    predicted = trained.probabilities[np.searchsorted(trained.prediction_rows, rows)].argmax(axis=1)
    truth, baseline, day_index = prepared.label[rows], prepared.baseline[rows], prepared.day_index[rows]
    whole = scores(truth, predicted, baseline)
    whole["days"] = {
        str(prepared.days[index]): scores(truth[day_index == index], predicted[day_index == index],
                                          baseline[day_index == index])
        for index in np.unique(day_index).tolist()
    }
    return whole


def write_run(prepared: Prepared, trained: Trained, settings: Settings, depth: int, seed: int, directory: Path,
              environment: dict) -> dict:
    """result.json, predictions.npz, model.pt 를 directory 에 쓰고 결과 dict 를 돌려준다."""
    directory.mkdir(parents=True, exist_ok=True)
    training_rows = prepared.rows(TRAIN, prepared.samples.labelled)
    truth = prepared.label[training_rows]
    result = {
        "depth": depth,
        "seed": seed,
        "parameters": parameter_count(trained.model),
        "settings": asdict(settings),
        "environment": environment,
        "stationary_width": prepared.stationary_width,
        "stationary_width_ticks": prepared.stationary_width_ticks,
        "baseline_threshold": prepared.baseline_threshold,
        # 학습 구간은 α 와 기준 문턱을 잰 곳이라 표본 수와 클래스 비율, 문턱을 고른 정확도만 싣는다
        "train": {
            "samples": len(training_rows), "class_shares": (np.bincount(truth, minlength=3) / len(truth)).tolist(),
            "baseline_accuracy": float(np.mean(prepared.baseline[training_rows] == truth)),
        },
        "epochs": trained.epochs,
        "best_epoch": trained.best_epoch,
        "normalization": {key: value.tolist() for key, value in asdict(prepared.normalization).items()},
    }
    for split, name in ((VALIDATION, "validation"), (EVALUATION, "evaluation")):
        if split in trained.splits:
            result[name] = split_scores(prepared, trained, split)
    (directory / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str))
    rows = trained.prediction_rows
    # 모의 매매는 날짜와 그 날짜 파일의 행 번호로 복원 배열의 호가를 찾는다. label 은 라벨 구간이 온전하지 않으면 -1 이다
    np.savez(
        directory / "predictions.npz",
        days=np.array([str(day) for day in prepared.days]), day_index=prepared.day_index[rows],
        day_row=prepared.day_row[rows], split=prepared.split[rows], probabilities=trained.probabilities,
        label=prepared.label[rows].astype(np.int8), labelled=prepared.samples.labelled[rows],
    )
    torch.save(trained.model.state_dict(), directory / "model.pt")
    return result