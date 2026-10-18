"""깊이 20 vs 10 실험의 설정을 한 곳에 모은다. #3 의 결정(기간, 라벨, 보합 폭, 표본 간격, 폭, 시드)이 여기 값 하나로 바뀐다."""

from dataclasses import dataclass
from datetime import date, timedelta

# 행이 어느 구간인지
TRAIN, VALIDATION, EVALUATION = 0, 1, 2


@dataclass(frozen=True)
class Settings:
    """실험 한 번의 설정. 지금 값은 #3 에서 2026-10-17 에 정한 값이고, 결과 JSON 에 그대로 실어 어떤 값으로 돌았는지 남긴다."""

    # 시간순으로 자른다. 끝 날짜 포함. 학습과 평가에 주말이 다 들어가고, 평가 구간은 한 번만 돈다
    train_days: tuple[date, date] = (date(2026, 9, 27), date(2026, 10, 6))
    validation_days: tuple[date, date] = (date(2026, 10, 7), date(2026, 10, 9))
    evaluation_days: tuple[date, date] = (date(2026, 10, 10), date(2026, 10, 16))
    # 정규화한 입력을 이 표준편차에서 자른다. 스프레드가 벌어진 프레임에서 상대 가격이 1,700 표준편차까지 튄다
    input_clip: float = 10.0
    # 라벨은 식 3 이다. 거래소 시각 t 뒤 horizon 안(t < E ≤ t + horizon) 프레임 mid 평균이 지금 mid 에서 얼마나 떨어졌나
    horizon_milliseconds: int = 30_000
    stationary_quantile: float = 1 / 3  # 보합 폭 α 는 학습 표본 |l| 의 이 분위수. 세 클래스가 비슷해진다
    window_rows: int = 100  # 입력 창. 원 논문과 같다
    # 날짜 파일 행 번호가 이 배수인 행만 표본으로 쓴다. 이웃 표본이 창 100행 중 99행을 나누면 한 바퀴 안에 과적합한다
    sample_every: int = 10
    # 모델 폭은 논문 그림 3·4 의 16/32 와 LSTM 64. 이때 10레벨 파라미터가 논문이 적은 약 60,000 이다
    conv_filters: int = 16
    inception_filters: int = 32
    lstm_units: int = 64
    # 논문은 배치 32, Adam lr 0.01·epsilon 1, 검증 정확도가 20 epoch 안 늘면 멈춤이다. 학습 표본이 82만 개라 배치를 키웠다
    batch_size: int = 256
    learning_rate: float = 1e-3
    patience: int = 3  # 검증 손실이 가장 낮던 epoch 뒤로 이만큼 더 안 줄면 멈춘다
    max_epochs: int = 30
    depths: tuple[int, ...] = (20, 10)
    seeds: tuple[int, ...] = (0, 1, 2)  # 결과 표에는 시드 셋의 평균과 표준편차를 싣는다

    def days(self) -> list[date]:
        """학습 첫 날짜부터 평가 끝 날짜까지. 날짜 파일을 이 차례로 이어 붙인다."""
        first, last = self.train_days[0], self.evaluation_days[1]
        return [first + timedelta(days=offset) for offset in range((last - first).days + 1)]

    def split_of(self, day: date) -> int:
        """날짜가 든 구간. 세 구간 밖의 날짜는 받지 않는다."""
        for split, (first, last) in enumerate((self.train_days, self.validation_days, self.evaluation_days)):
            if first <= day <= last:
                return split
        raise ValueError(f"{day} 는 학습·검증·평가 어느 구간에도 없다")