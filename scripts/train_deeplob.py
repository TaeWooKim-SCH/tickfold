"""
날짜별 복원 파일로 DeepLOB 깊이 20 과 10 을 시드마다 학습하고 평가한다. 깊이·시드마다 폴더 하나에 result.json,
predictions.npz, model.pt 를 쓰고, 끝에 깊이별 평균과 표준편차를 summary.json 으로 쓴다.

사용법 (프로젝트 루트에서):
    nice -n 19 ionice -c3 uv run --group model python scripts/train_deeplob.py --depths 20 10 --seeds 0 \
        --max-epochs 1 --output runs/check
    nice -n 19 ionice -c3 uv run --group model python scripts/train_deeplob.py --evaluate

평가 구간은 설정을 다 정한 뒤 --evaluate 를 준 실행에서 한 번만 돈다. 주지 않으면 평가 구간은 예측도 채점도 하지 않는다.

설정(라벨, 기간, 폭, 학습)은 tickfold/model/settings.py 한 곳에 있다. GPU 가 안 잡히면 멈춘다. CPU 로 돌릴 때만 --device cpu 를 준다.
"""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from tickfold.model.evaluation import write_run
from tickfold.model.inputs import read_days
from tickfold.model.settings import EVALUATION, TRAIN, VALIDATION, Settings
from tickfold.model.training import prepare, train


def summarize(results: list[dict], part: str) -> dict:
    """깊이마다 part 구간 점수의 시드 사이 평균과 표준편차. 다수 클래스와 모델 없는 기준은 시드와 상관없이 같아 하나만 싣는다."""
    summary = {"part": part}
    for depth in sorted({result["depth"] for result in results}, reverse=True):
        runs = [result for result in results if result["depth"] == depth]
        each_seed = [run[part] for run in runs]
        line = {"seeds": [run["seed"] for run in runs], "parameters": runs[0]["parameters"]}
        for key in ("accuracy", "macro_f1"):
            values = np.array([scores[key] for scores in each_seed])
            # 시드 셋의 표본 표준편차. 시드가 하나면 퍼짐을 모른다
            spread = float(values.std(ddof=1)) if len(values) > 1 else None
            line[key] = {"mean": float(values.mean()), "standard_deviation": spread}
        line["majority_share"] = each_seed[0]["majority_share"]
        line["baseline_accuracy"] = each_seed[0]["baseline_accuracy"]
        line["days"] = {day: float(np.mean([scores["days"][day]["accuracy"] for scores in each_seed]))
                        for day in each_seed[0]["days"]}
        summary[str(depth)] = line
    return summary


def main() -> None:
    defaults = Settings()
    parser = argparse.ArgumentParser(description="DeepLOB 깊이 20 vs 10 을 시드마다 학습하고 평가한다")
    parser.add_argument("--days", type=Path, default=Path("restored/BTCUSDT"), help="날짜별 .npz 폴더")
    parser.add_argument("--output", type=Path, default=Path("runs/deeplob"), help="깊이·시드마다 폴더를 만들 곳")
    parser.add_argument("--depths", type=int, nargs="+", default=list(defaults.depths), choices=range(1, 21))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(defaults.seeds))
    parser.add_argument("--max-epochs", type=int, default=defaults.max_epochs, help="짧게 확인할 때 줄인다")
    # 기본으로 평가 구간을 채점하면 설정을 고르는 중에 평가 숫자를 보게 된다. 설정을 다 정한 마지막 한 번만 준다
    parser.add_argument("--evaluate", action="store_true", help="평가 구간도 예측하고 채점한다. 설정을 다 정한 뒤 한 번만")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    arguments = parser.parse_args()
    # 재부팅 뒤 새 커널에 nvidia 모듈이 없으면 GPU 가 안 잡힌다. 말없이 CPU 로 돌면 AMP 가 꺼져 숫자가 달라지고 몇 배 오래 걸린다
    if arguments.device == "cuda" and not torch.cuda.is_available():
        parser.error("GPU 가 안 잡힌다. nvidia-smi 를 보고, CPU 로 돌릴 때만 --device cpu 를 준다")
    if arguments.max_epochs < 1:  # epoch 이 없으면 고를 모델도 없다
        parser.error("--max-epochs 는 1 이상이다")
    settings = replace(defaults, depths=tuple(arguments.depths), seeds=tuple(arguments.seeds),
                       max_epochs=arguments.max_epochs)
    device = torch.device(arguments.device)
    if arguments.evaluate:
        splits, part, part_name = (VALIDATION, EVALUATION), "evaluation", "평가"
    else:
        splits, part, part_name = (VALIDATION,), "validation", "검증"
    began = time.monotonic()
    joined = read_days(arguments.days, settings)
    prepared = prepare(joined, settings, device)
    del joined  # 장치에 올린 뒤에는 호스트의 열 5.4GB 를 놓는다
    print(f"읽기와 준비 {time.monotonic() - began:.0f}초, 장치 {device}. 보합 폭 α {prepared.stationary_width:.4e} "
          f"({prepared.stationary_width_ticks:.1f}틱), 모델 없는 기준 문턱 {prepared.baseline_threshold:.4e}", flush=True)
    for name, split in (("학습", TRAIN), ("검증", VALIDATION), ("평가", EVALUATION)):
        if split != TRAIN and split not in splits:  # 평가 구간의 라벨 비율도 평가 숫자라 --evaluate 실행에서만 찍는다
            continue
        truth = prepared.label[prepared.rows(split, prepared.samples.labelled)]
        shares = np.bincount(truth, minlength=3) / len(truth) * 100
        print(f"  {name} 표본 {len(truth):,}, 내림/보합/오름 {shares[0]:.1f}/{shares[1]:.1f}/{shares[2]:.1f}%")
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    environment = {"torch": torch.__version__, "device": device_name}
    results = []
    for depth in settings.depths:
        for seed in settings.seeds:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            trained = train(prepared, settings, depth, seed, device, splits)
            peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            directory = arguments.output / f"depth{depth}-seed{seed}"
            run_environment = {**environment, "peak_memory_bytes": peak}
            result = write_run(prepared, trained, settings, depth, seed, directory, run_environment)
            scores = result[part]
            print(f"깊이 {depth} 시드 {seed}: 파라미터 {result['parameters']:,}, 가장 좋은 epoch {trained.best_epoch}, "
                  f"{part_name} 정확도 {scores['accuracy']:.4f}, macro-F1 {scores['macro_f1']:.4f}, "
                  f"다수 클래스 {scores['majority_share']:.4f}, 모델 없는 기준 {scores['baseline_accuracy']:.4f}. {directory}",
                  flush=True)
            results.append(result)
    summary = summarize(results, part)
    (arguments.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"{arguments.output / 'summary.json'}, 모두 {time.monotonic() - began:.0f}초")


if __name__ == "__main__":
    main()