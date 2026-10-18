import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

# 모델용 의존성 그룹(model)을 안 깐 환경에서는 이 파일을 건너뛴다
pytest.importorskip("torch")
pytest.importorskip("sklearn")

from test_model_inputs import random_walk, write_day
from test_model_training import hide_gpu  # 이름을 가져오면 GPU 를 가리는 autouse fixture 가 이 파일에도 걸린다
from tickfold.model.settings import EVALUATION, Settings

SCRIPT = str(Path(__file__).parent.parent / "scripts" / "train_deeplob.py")


def test_without_evaluate_the_script_trains_each_seed_and_never_predicts_or_scores_the_evaluation_days(
    tmp_path, monkeypatch, capsys
):
    # 기본 설정의 20일을 450행씩. 창 100행과 horizon 30초가 들어가 날마다 표본이 몇 개씩 나온다
    random = np.random.default_rng(4)
    for day in Settings().days():
        write_day(tmp_path, day, random_walk(450, random), random)
    output = tmp_path / "runs"
    arguments = ["--days", str(tmp_path), "--output", str(output), "--depths", "10", "--seeds", "0", "1",
                 "--max-epochs", "1", "--device", "cpu"]
    monkeypatch.setattr(sys, "argv", [SCRIPT, *arguments])

    runpy.run_path(SCRIPT, run_name="__main__")

    results = [json.loads((output / f"depth10-seed{seed}" / "result.json").read_text()) for seed in (0, 1)]
    assert all((output / f"depth10-seed{seed}" / name).exists() for seed in (0, 1)
               for name in ("predictions.npz", "model.pt"))
    assert all("evaluation" not in result and "validation" in result for result in results)
    assert EVALUATION not in np.load(output / "depth10-seed0" / "predictions.npz")["split"]
    assert "평가" not in capsys.readouterr().out  # 평가 구간의 라벨 비율도 찍지 않는다
    summary = json.loads((output / "summary.json").read_text())
    accuracy = np.array([result["validation"]["accuracy"] for result in results])
    assert summary["part"] == "validation" and summary["10"]["seeds"] == [0, 1]
    assert np.isclose(summary["10"]["accuracy"]["mean"], accuracy.mean())
    assert list(summary["10"]["days"]) == [str(day) for day in Settings().days()[10:13]]
    # 작은 합성 데이터에서는 두 시드가 같은 클래스만 찍어 정확도가 같다. 퍼짐과 날짜별 평균은 값을 넣어 따로 본다
    runs = [{"depth": 10, "seed": seed, "parameters": 1, "validation": {
        "accuracy": value, "macro_f1": value, "majority_share": 0.5, "baseline_accuracy": 0.4,
        "days": {"2026-10-07": {"accuracy": value}}}} for seed, value in ((0, 0.4), (1, 0.6))]
    line = runpy.run_path(SCRIPT)["summarize"](runs, "validation")["10"]
    assert np.isclose(line["accuracy"]["standard_deviation"], np.std([0.4, 0.6], ddof=1))
    assert np.isclose(line["days"]["2026-10-07"], 0.5)