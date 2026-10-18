"""거래 목록을 논문 결과 표의 숫자로 줄인다. 모델 예측과 기준선이 같은 셈을 탄다."""

from dataclasses import dataclass

import numpy as np

from tickfold.paper_trading.walk import HORIZON, OPPOSITE, RANGE_END, UNTRADABLE, Trades

TICK_SIZE = 0.01  # BTCUSDT 호가 단위. 단위가 바뀐 이력은 아직 안 모은다(1절)


@dataclass(frozen=True)
class Performance:
    """수수료 전과 뒤를 한 벌씩. 금액은 1 BTC 당 USDT 이고 bp 는 거래마다 들 때 가격에 견준 값의 평균이다."""

    trades: int
    gross_total: float
    net_total: float
    gross_mean_bp: float  # 누적 손익은 거래 수에 비례해 커지므로 거래당 값을 같이 싣는다
    net_mean_bp: float
    gross_max_drawdown: float  # 청산 차례로 쌓은 누적 손익이 앞선 최고점(0 포함)에서 가장 깊이 내려간 폭
    net_max_drawdown: float
    # 수수료 전 손익은 틱의 배수라 틱 정수로 반올림해 센다. 계산한 float 를 0 이나 -0.01 과 바로 견주면 틀린다
    win_rate: float  # 수수료 전 손익이 1틱 이상인 거래 비율
    net_win_rate: float  # 수수료 뒤 손익이 0 보다 큰 거래 비율. 수수료는 틱의 배수가 아니라 반올림하지 않는다
    one_tick_loss_rate: float  # 수수료 전 손익이 정확히 -1틱인 거래. 대개 가격이 그대로라 스프레드만 낸 거래다
    exits: dict[str, int]  # 끝난 까닭별 거래 수
    skipped_signals: int


def measure(trades: Trades) -> Performance:
    """거래 목록 하나를 Performance 로 줄인다. 거래가 없으면 평균과 비율은 NaN 이다."""
    gross, net = trades.gross, trades.net
    gross_ticks = np.rint(gross / TICK_SIZE)
    count = len(gross)
    mean = (lambda values: float(values.mean())) if count else (lambda values: float("nan"))
    names = {HORIZON: "horizon", OPPOSITE: "opposite", UNTRADABLE: "untradable", RANGE_END: "range_end"}
    return Performance(
        count,
        float(gross.sum()),
        float(net.sum()),
        mean(gross / trades.entry_price * 1e4),
        mean(net / trades.entry_price * 1e4),
        max_drawdown(gross),
        max_drawdown(net),
        mean(gross_ticks >= 1),
        mean(net > 0),
        mean(gross_ticks == -1),
        {name: int(np.sum(trades.exit_reason == reason)) for reason, name in names.items()},
        trades.skipped_signals,
    )


def max_drawdown(profits: np.ndarray) -> float:
    """거래 차례로 쌓은 누적 손익이 그때까지의 최고점에서 가장 깊이 내려간 폭. 시작점 0 도 최고점 후보다."""
    running = np.concatenate(([0.0], np.cumsum(profits)))
    return float((np.maximum.accumulate(running) - running).max())