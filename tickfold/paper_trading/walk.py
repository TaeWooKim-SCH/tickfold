"""신호를 받아 포지션 하나로 사고팔고 거래마다 체결 행과 가격을 남긴다. 모델 예측과 기준선이 같은 규칙을 탄다."""

from dataclasses import dataclass

import numpy as np

from tickfold.paper_trading.market import Market
from tickfold.replay.depth20 import ASK, BID, PRICE

# 거래가 끝난 까닭. 갭이나 못 쓰는 행에서 끝난 거래와 범위 끝에서 억지로 닫은 거래를 따로 센다
HORIZON, OPPOSITE, UNTRADABLE, RANGE_END = 0, 1, 2, 3


@dataclass(frozen=True)
class TradingRule:
    """모의 매매 규칙 한 벌. 계획서 5절 '착수 때 정할 것' 의 답(안)이 기본값이고, 바꿀 값은 여기 한 곳에서 바꾼다."""

    horizon_milliseconds: int = 30_000  # 라벨 horizon 과 같은 값(#3). 행 수가 아니라 거래소 시각으로 잰다
    delay_rows: int = 1  # 신호 행 뒤 몇 번째 행의 최우선 호가에 체결하는가. 1 이면 다음 프레임(대개 100ms 뒤)
    fee_rate: float = 0.001  # 테이커 일반 등급. 들 때와 날 때 체결 금액마다 곱한다
    allow_short: bool = True  # 거짓이면 팔자 신호는 든 매수를 닫기만 한다(현물 그대로)
    extend_on_same_signal: bool = False  # 든 쪽과 같은 신호가 또 오면 청산 시각을 그 신호부터 다시 잴지


@dataclass(frozen=True)
class Trades:
    """거래마다 한 칸. 수량은 1 BTC 로 쳐서 손익과 수수료는 1 BTC 당 USDT 다."""

    side: np.ndarray  # (거래,) int8. 1 은 사서 팔기, -1 은 공매도 뒤 되사기
    signal_row: np.ndarray  # (거래,) int64. 여기부터 셋은 Market 의 행
    entry_row: np.ndarray  # (거래,) int64
    exit_row: np.ndarray  # (거래,) int64
    entry_price: np.ndarray  # (거래,) float64. 살 때는 매도 1호가, 팔 때는 매수 1호가라 스프레드가 손익에 들어 있다
    exit_price: np.ndarray  # (거래,) float64
    exit_reason: np.ndarray  # (거래,) int8. HORIZON, OPPOSITE, UNTRADABLE, RANGE_END
    fee: np.ndarray  # (거래,) float64. 들 때와 날 때 수수료의 합
    skipped_signals: int  # 신호 행부터 체결 행까지 못 쓰는 행이나 갭이 끼어 안 든 신호

    @property
    def gross(self) -> np.ndarray:
        """수수료 전 손익."""
        return self.side * (self.exit_price - self.entry_price)

    @property
    def net(self) -> np.ndarray:
        """수수료 뒤 손익."""
        return self.gross - self.fee


def walk(market: Market, signal_rows: np.ndarray, sides: np.ndarray, rule: TradingRule) -> Trades:
    """오름차순 signal_rows 의 신호(sides 가 1 사자, -1 팔자)를 규칙대로 들고 닫는다.

    포지션이 하나라 다음 거래를 언제 들 수 있는지가 앞 거래의 청산 행에 달려서 신호 차례로 도는 루프 하나다. 행마다 돌지 않고
    청산 행은 정렬된 행 목록에서 찾으므로 루프는 거래 수와 건너뛴 신호 수만큼만 돈다.
    """
    rows, time = len(market.exchange_time), market.exchange_time
    usable = np.flatnonzero(market.tradable)
    broken = np.flatnonzero(~market.tradable | market.resumed_after_gap)
    same_side = {1: signal_rows[sides > 0], -1: signal_rows[sides < 0]}
    entries = np.flatnonzero(sides != 0) if rule.allow_short else np.flatnonzero(sides > 0)
    entry_rows = signal_rows[entries]
    taken, skipped, index = [], 0, 0
    while index < len(entry_rows):
        signal, side = int(entry_rows[index]), int(sides[entries[index]])
        entry = signal + rule.delay_rows
        # 신호 행부터 체결 행까지 하나라도 못 쓰는 행이거나 갭 뒤 행이면 그 사이 호가를 모르는 채 들게 된다
        if entry >= rows or _first_after(broken, signal - 1, rows) <= entry:
            skipped, index = skipped + 1, index + 1
            continue
        opposite = _first_after(same_side[-side], signal, rows)
        stop = _first_after(broken, entry, rows)
        anchor = signal
        while True:
            deadline = int(np.searchsorted(time, time[anchor] + rule.horizon_milliseconds, side="left"))
            trigger = min(deadline, opposite, stop)
            # 청산 행 앞에 온 같은 쪽 신호 가운데 마지막 것부터 다시 잰다. 그 앞 신호가 미룬 청산 시각은 그보다 이르다
            later = _last_before(same_side[side], trigger) if rule.extend_on_same_signal else -1
            if later <= anchor:
                break
            anchor = later
        # 체결 행이 갭이나 재동기화 대기에 걸리면 다시 쓸 수 있게 된 첫 행에서 닫는다. 그동안 가격 위험은 든 채로 진다
        exit_row = _first_after(usable, trigger + rule.delay_rows - 1, rows)
        # 반대 신호나 horizon 이 갭 바로 앞에 와도 닫는 체결은 갭 뒤라서, 갭에 걸렸는지는 청산 행까지 보고 정한다
        reason = UNTRADABLE if stop <= exit_row else OPPOSITE if trigger == opposite else HORIZON
        if exit_row >= rows:
            exit_row, reason = int(usable[-1]), RANGE_END
        taken.append((side, signal, entry, exit_row, reason))
        index = int(np.searchsorted(entry_rows, trigger, side="right"))
    return _priced(market, taken, skipped, rule.fee_rate)


def _first_after(sorted_rows: np.ndarray, row: int, missing: int) -> int:
    """sorted_rows 가운데 row 보다 큰 첫 값. 없으면 missing."""
    at = int(np.searchsorted(sorted_rows, row, side="right"))
    return int(sorted_rows[at]) if at < len(sorted_rows) else missing


def _last_before(sorted_rows: np.ndarray, row: int) -> int:
    """sorted_rows 가운데 row 보다 작은 마지막 값. 없으면 -1."""
    at = int(np.searchsorted(sorted_rows, row, side="left")) - 1
    return int(sorted_rows[at]) if at >= 0 else -1


def _priced(market: Market, taken: list[tuple[int, int, int, int, int]], skipped: int, fee_rate: float) -> Trades:
    # 거래가 하나도 없어도 칸마다 빈 배열이 되게 모양을 정해 둔다
    side, signal_row, entry_row, exit_row, reason = np.array(taken, dtype=np.int64).reshape(-1, 5).T
    buying = side > 0
    entry_price = np.where(buying, market.best[entry_row, ASK, PRICE], market.best[entry_row, BID, PRICE])
    exit_price = np.where(buying, market.best[exit_row, BID, PRICE], market.best[exit_row, ASK, PRICE])
    fee = fee_rate * (entry_price + exit_price)
    return Trades(side.astype(np.int8), signal_row, entry_row, exit_row, entry_price, exit_price,
                  reason.astype(np.int8), fee, skipped)