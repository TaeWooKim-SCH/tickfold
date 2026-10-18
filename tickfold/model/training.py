"""DeepLOB-N 을 한 깊이·한 시드로 학습하고, 검증 손실이 가장 낮던 모델로 검증·평가 구간 행마다 세 확률을 낸다."""

import math
import os
import time
from dataclasses import dataclass
from datetime import date

import numpy as np
import torch
from torch.nn import functional

from tickfold.model.deeplob import DeepLOB
from tickfold.model.inputs import JoinedDays, Normalization, fit_normalization, normalize
from tickfold.model.labels import Samples, baseline_threshold, classes, select_samples, stationary_width
from tickfold.model.settings import TRAIN, VALIDATION, Settings

# cuBLAS 는 이 값이 첫 CUDA 연산 전에 있어야 같은 시드에서 같은 숫자를 낸다. 진입점이 무엇을 하든 이 모듈을 먼저 읽는다
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
_PREDICTION_BATCH = 4096


@dataclass(frozen=True)
class Prepared:
    """깊이와 시드가 달라도 같은 입력·라벨·기준. 한 번 만들어 여섯 번의 학습이 같이 쓴다."""

    days: tuple[date, ...]
    day_index: np.ndarray  # (행,) int16
    day_row: np.ndarray  # (행,) int32
    split: np.ndarray  # (행,) int8
    samples: Samples
    label: np.ndarray  # (행,) int64. 라벨 구간이 온전하지 않은 행은 -1
    baseline: np.ndarray  # (행,) int64. 모델 없는 기준의 예측
    stationary_width: float  # 보합 폭 α, 상대값
    stationary_width_ticks: float  # 같은 α 를 학습 표본 평균 mid 에서 0.01 USDT 틱으로
    baseline_threshold: float
    normalization: Normalization
    columns: torch.Tensor  # (행, 4 × 20) float32. 학습할 장치에 올린 정규화한 열
    device_label: torch.Tensor  # label 을 같은 장치에 올린 것

    def rows(self, split: int, selection: np.ndarray) -> np.ndarray:
        """구간 하나에서 selection 이 참인 행 번호."""
        return np.flatnonzero(selection & (self.split == split))


@dataclass(frozen=True)
class Trained:
    """학습 한 번의 결과. 확률은 splits 구간에서 창이 온전한 행마다 있다."""

    model: DeepLOB
    epochs: list[dict]  # epoch 마다 학습 손실, 검증 손실, 검증 정확도, 걸린 초
    best_epoch: int
    splits: tuple[int, ...]  # 확률을 낸 구간
    prediction_rows: np.ndarray  # (예측,) int64. Prepared 의 행 번호
    probabilities: np.ndarray  # (예측, 3) float32. 내림, 보합, 오름


def prepare(joined: JoinedDays, settings: Settings, device: torch.device) -> Prepared:
    """정규화, 라벨, 보합 폭, 모델 없는 기준을 만든다. 통계와 α 와 문턱은 학습 구간에서만 잰다. joined.columns 를 그 자리에서 바꾼다."""
    normalization = fit_normalization(joined)
    normalize(joined.columns, normalization, settings.input_clip)
    samples = select_samples(joined, settings)
    training_rows = np.flatnonzero(samples.labelled & (joined.split == TRAIN))
    width = stationary_width(samples.change[training_rows], settings.stationary_quantile)
    label = np.where(np.isnan(samples.change), -1, classes(samples.change, width))
    threshold = baseline_threshold(samples.window_drift[training_rows], label[training_rows])
    # mid_half_cents 의 절반이 0.01 USDT 틱으로 센 mid 다
    ticks = width * float(joined.mid_half_cents[training_rows].mean()) / 2
    columns, device_label = torch.from_numpy(joined.columns).to(device), torch.from_numpy(label).to(device)
    return Prepared(
        joined.days, joined.day_index, joined.day_row, joined.split, samples, label,
        classes(samples.window_drift, threshold), width, ticks, threshold, normalization, columns, device_label,
    )


def train(prepared: Prepared, settings: Settings, depth: int, seed: int, device: torch.device,
          splits: tuple[int, ...] = (VALIDATION,), log=print) -> Trained:
    """학습 표본을 epoch 마다 섞어 돌고, 검증 손실이 patience 번 안 줄면 멈춘다. 같은 시드면 같은 숫자가 나온다.

    끝나면 검증 손실이 가장 낮던 모델로 splits 구간에서 창이 온전한 행마다 세 확률을 낸다. 평가 구간은 설정을 다 정한 뒤
    한 번만 돌므로 기본값에 넣지 않고, 부르는 쪽이 splits 에 넣을 때만 낸다.
    """
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = DeepLOB(depth, settings.conv_filters, settings.inception_filters, settings.lstm_units).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    mixed = device.type == "cuda"  # CPU 에서는 float16 이 빠르지 않아 시험이 쓰지 않는다
    scaler = torch.amp.GradScaler(device.type, enabled=mixed)
    generator = torch.Generator(device=device).manual_seed(seed)
    columns = prepared.columns[:, : 4 * depth]  # 10레벨 열은 20레벨 열의 앞 40개다
    train_rows, validation_rows = (
        torch.from_numpy(prepared.rows(split, prepared.samples.labelled)).to(device) for split in (TRAIN, VALIDATION)
    )
    epochs, best_loss, best_epoch, best_state = [], float("inf"), 0, None
    for epoch in range(1, settings.max_epochs + 1):
        began = time.monotonic()
        model.train()
        order = train_rows[torch.randperm(len(train_rows), generator=generator, device=device)]
        loss_sum = torch.zeros((), device=device)
        for batch in order.split(settings.batch_size):
            with torch.autocast(device.type, dtype=torch.float16, enabled=mixed):
                logits = model(_windows(columns, batch, settings))
                loss = functional.cross_entropy(logits, prepared.device_label[batch])
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.detach().float() * len(batch)  # .item() 은 배치마다 GPU 를 기다리게 해서 epoch 끝에 한 번만 읽는다
        logits = _logits(model, columns, validation_rows, settings, mixed)
        truth = prepared.device_label[validation_rows]
        validation_loss = functional.cross_entropy(logits, truth).item()
        if not math.isfinite(validation_loss):  # NaN 은 가장 낮은 손실과 견줘도 거짓이라 멈추지도 않고 고를 모델도 없다
            raise FloatingPointError(f"깊이 {depth} 시드 {seed} epoch {epoch} 검증 손실 {validation_loss}")
        accuracy = (logits.argmax(1) == truth).float().mean().item()
        epochs.append({
            "epoch": epoch, "train_loss": loss_sum.item() / len(train_rows), "validation_loss": validation_loss,
            "validation_accuracy": accuracy, "seconds": time.monotonic() - began,
        })
        log(f"  깊이 {depth} 시드 {seed} epoch {epoch}: 학습 손실 {epochs[-1]['train_loss']:.4f}, "
            f"검증 손실 {validation_loss:.4f}, 검증 정확도 {accuracy:.4f}, {epochs[-1]['seconds']:.0f}초")
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch
            best_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
        elif epoch - best_epoch >= settings.patience:
            break
    model.load_state_dict(best_state)
    prediction_rows = np.flatnonzero(prepared.samples.window_ok & np.isin(prepared.split, splits))
    logits = _logits(model, columns, torch.from_numpy(prediction_rows).to(device), settings, mixed)
    probabilities = torch.softmax(logits, dim=1).cpu().numpy()
    return Trained(model, epochs, best_epoch, tuple(splits), prediction_rows, probabilities)


def _windows(columns: torch.Tensor, rows: torch.Tensor, settings: Settings) -> torch.Tensor:
    # 행마다 그 행에서 끝나는 창을 인덱스로 뽑는다. 20일치 열을 장치에 한 번 올려 두면 DataLoader 없이 같은 속도가 난다
    offsets = torch.arange(1 - settings.window_rows, 1, device=rows.device)
    return columns[rows[:, None] + offsets].unsqueeze(1)


def _logits(model: DeepLOB, columns: torch.Tensor, rows: torch.Tensor, settings: Settings, mixed: bool):
    model.eval()
    with torch.no_grad(), torch.autocast(columns.device.type, dtype=torch.float16, enabled=mixed):
        parts = [model(_windows(columns, batch, settings)).float() for batch in rows.split(_PREDICTION_BATCH)]
    return torch.cat(parts) if parts else torch.empty((0, 3), device=columns.device)