"""모델 예측 옆에 싣는 기준선 셋. 항상 보유, 최우선 호가 수량 불균형 규칙, 거래 수를 맞춘 무작위 신호."""

import numpy as np

from tickfold.paper_trading.market import Market
from tickfold.paper_trading.walk import RANGE_END, Trades, TradingRule, walk
from tickfold.replay.depth20 import ASK, BID, PRICE, QUANTITY


def always_hold(market: Market, fee_rate: float) -> Trades:
    """첫 쓸 수 있는 행의 매도 1호가에 사서 마지막 쓸 수 있는 행의 매수 1호가에 판다. 갭에서도 닫지 않는다."""
    usable = np.flatnonzero(market.tradable)
    first, last = usable[[0]], usable[[-1]]
    entry_price, exit_price = market.best[first, ASK, PRICE], market.best[last, BID, PRICE]
    side, reason = np.ones(1, dtype=np.int8), np.full(1, RANGE_END, dtype=np.int8)
    fee = fee_rate * (entry_price + exit_price)
    return Trades(side, first, first, last, entry_price, exit_price, reason, fee, 0)


def imbalance_signals(market: Market, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """최우선 호가의 (매수 수량 - 매도 수량) / 합이 문턱을 넘으면 사자, -문턱보다 작으면 팔자. 모델 없는 규칙(계획서 5절)."""
    bid_quantity, ask_quantity = market.best[:, BID, QUANTITY], market.best[:, ASK, QUANTITY]
    # 동기화 안 된 행은 NaN 이라 비교가 거짓이 되어 신호가 안 난다
    imbalance = (bid_quantity - ask_quantity) / (bid_quantity + ask_quantity)
    sides = np.where(imbalance > threshold, 1, np.where(imbalance < -threshold, -1, 0))
    rows = np.flatnonzero((sides != 0) & market.tradable)
    return rows, sides[rows]


def random_trades(
    market: Market, candidate_rows: np.ndarray, target_trades: int, rule: TradingRule, seed: int
) -> Trades:
    """오름차순 candidate_rows 가운데 무작위 행에 무작위 쪽 신호를 넣어 같은 규칙으로 돈다. 거래 수가 target_trades 에 가장 가깝게.

    신호 수와 거래 수는 같지 않다. 든 동안 온 신호는 무시되거나 청산만 하기 때문이다. 같은 무작위 차례의 앞 몇 개를 쓸지만
    바꿔 가며 맞춘다. 앞 몇 개만 바꾸면 신호 집합이 서로 포개져서 신호를 늘릴 때 거래 수가 거의 늘기만 해 범위를 좁혀 찾을 수 있다.
    """
    generator = np.random.default_rng(seed)
    order = generator.permutation(len(candidate_rows))
    sides = generator.choice(np.array([-1, 1]), len(candidate_rows))
    lowest, highest = 0, len(candidate_rows)
    count, closest = min(target_trades, highest), None
    # 비례로 짐작한 값이 범위 안이면 그리로 가고 아니면 반으로 가른다. 못 맞추면 가장 가까운 것을 돌려주고 거래 수가 결과에 찍힌다
    for _ in range(40):
        chosen = np.sort(order[:count])
        trades = walk(market, candidate_rows[chosen], sides[chosen], rule)
        made = len(trades.side)
        if closest is None or abs(made - target_trades) < abs(len(closest.side) - target_trades):
            closest = trades
        if made == target_trades:
            break
        lowest, highest = (count, highest) if made < target_trades else (lowest, count)
        if highest - lowest <= 1:
            break
        guess = round(count * target_trades / max(made, 1))
        count = guess if lowest < guess < highest else (lowest + highest) // 2
    return closest