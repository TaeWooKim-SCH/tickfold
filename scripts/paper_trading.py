"""
모의 매매 1단계. 예측을 규칙대로 사고팔아 수수료 전·후 손익과 기준선 셋을 찍고 누적 손익 그림을 쓴다.

사용법 (프로젝트 루트에서. 복원 파일을 열흘치 읽으므로 앞에 nice -n 19 ionice -c3 을 붙인다):
    uv run --group figures python scripts/paper_trading.py --predictions runs/deeplob/depth20-seed0/predictions.npz \
        --figure runs/deeplob/depth20-seed0/paper_trading.png
    uv run --group figures python scripts/paper_trading.py --random-predictions 0 \
        --figure runs/paper_trading_random.png

신호 문턱은 검증 구간에서 거래가 1,000건 이상인 문턱 가운데 거래당 수수료 전 bp 가 가장 큰 값으로 고르고, 평가 구간은
그 문턱으로 한 번만 돈다.
모델 예측이면 분할과 horizon 을 예측 옆 result.json 에서 읽는다. --random-predictions 는 모델 예측이 나오기 전에
끝까지 돌려 보는 무작위 예측이다.
"""

import argparse
import json
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 서버에 화면이 없다
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import StrMethodFormatter

from tickfold.paper_trading.baselines import always_hold, imbalance_signals, random_trades
from tickfold.paper_trading.market import Market, load_market
from tickfold.paper_trading.performance import TICK_SIZE, Performance, measure
from tickfold.paper_trading.predictions import (
    Predictions, load_predictions, prediction_rows, random_predictions, signals_of,
)
from tickfold.paper_trading.walk import Trades, TradingRule, walk
from tickfold.replay.depth20 import ASK, BID, PRICE, QUANTITY

# 무작위 예측의 분할. #3 에서 정한 값이고 평가 구간에 주말(10-10, 10-11)이 들어 있다
VALIDATION, EVALUATION = (date(2026, 10, 7), date(2026, 10, 9)), (date(2026, 10, 10), date(2026, 10, 16))
DELAYS = (1, 2, 5)  # 체결을 몇 행 뒤로 미룰지. 1 이 기본 결과이고 나머지는 지연에 손익이 얼마나 민감한지 본다
IMBALANCE_THRESHOLD, IMBALANCE_HOLD_MILLISECONDS = 0.8, 10_000  # 계획서 5절의 모델 없는 규칙 그대로
# 모델은 입력 창이 온전한 행마다 예측한다. 무작위 예측은 이웃 행끼리 이어지지 않아 어느 간격이든 모델의 거래 수를
# 미리 보여 주지 못한다. 행마다 두면 신호가 프레임마다 뒤집혀 거래만 수십만 건이 되므로 10행마다 둔다
SAMPLE_EVERY_ROWS = 10
# 거래가 몇 건뿐인 문턱은 거래당 값이 그 몇 건의 우연에 흔들린다. 검증 3일에서 문턱 0.40 ~ 0.65 가 1만 건을 넘는다
MINIMUM_VALIDATION_TRADES = 1_000


def line(name: str, result: Performance) -> str:
    """결과 표 한 줄. 논문 표에 옮겨 적는 값이다."""
    exits = ", ".join(f"{reason} {count:,}" for reason, count in result.exits.items())
    return (f"{name}: 거래 {result.trades:,}, 수수료 전 {result.gross_total:+,.2f}, 뒤 {result.net_total:+,.2f} USDT, "
            f"거래당 {result.gross_mean_bp:+.3f} / {result.net_mean_bp:+.3f}bp, "
            f"최대 낙폭 {result.gross_max_drawdown:,.2f} / {result.net_max_drawdown:,.2f}, "
            f"승률 {result.win_rate:.1%}(수수료 뒤 {result.net_win_rate:.2%}), -1틱 {result.one_tick_loss_rate:.1%}, "
            f"끝난 까닭 {exits}, 건너뛴 신호 {result.skipped_signals:,}")


def fill_line(market: Market, trades: Trades) -> str:
    """들 때 건너는 쪽 1호가 수량과 스프레드가 넓던 거래. 주문 수량을 정할 근거로 세기만 하고 손익에서 빼지 않는다."""
    buying, best = trades.side > 0, market.best
    quantity = np.where(buying, best[trades.entry_row, ASK, QUANTITY], best[trades.entry_row, BID, QUANTITY])
    spread_ticks = [np.rint((best[rows, ASK, PRICE] - best[rows, BID, PRICE]) / TICK_SIZE)
                    for rows in (trades.entry_row, trades.exit_row)]
    wide = (spread_ticks[0] > 1) | (spread_ticks[1] > 1)
    return (f"  들 때 건너는 쪽 1호가 수량 중앙 {np.median(quantity):.3f} BTC, 0.01 미만 {np.mean(quantity < 0.01):.2%}, "
            f"1 미만 {np.mean(quantity < 1):.2%}. 들거나 날 때 스프레드가 1틱보다 넓던 거래 {int(wide.sum()):,}")


def model_settings(predictions: Path) -> tuple[list[date], list[date], int]:
    """모델 갈래가 예측 옆에 쓴 result.json 의 분할과 horizon. 모의 매매가 #3 설정을 모델이 학습한 값 그대로 따른다."""
    settings = json.loads((predictions.parent / "result.json").read_text())["settings"]
    validation, evaluation = ([date.fromisoformat(day) for day in settings[f"{name}_days"]]
                              for name in ("validation", "evaluation"))
    return validation, evaluation, settings["horizon_milliseconds"]


def choose_threshold(market: Market, predictions: Predictions, rule: TradingRule, thresholds: list[float],
                     minimum_trades: int = MINIMUM_VALIDATION_TRADES) -> float:
    """검증 구간에서 문턱마다 돌려, 거래가 minimum_trades 건 이상인 문턱 가운데 거래당 수수료 전 bp 가 가장 큰 문턱을 고른다.

    총손익으로 고르면 문턱을 내릴수록 거래가 늘어 늘 후보의 끝이 뽑힌다. 수수료 뒤로 고르면 거래가 적을수록 나아 보인다.
    깊이 20 과 10 을 견주는 값이 거래당 예측력이라서 그것으로 고른다(2026-10-18 결정).
    """
    per_trade = {}
    for threshold in thresholds:
        result = measure(walk(market, *signals_of(market, predictions, threshold), rule))
        print(line(f"  검증 문턱 {threshold:.2f}", result), flush=True)
        if result.trades >= minimum_trades:
            per_trade[threshold] = result.gross_mean_bp
    if not per_trade:
        raise ValueError(f"검증 구간에서 거래가 {minimum_trades:,}건 이상 난 문턱이 없다")
    chosen = max(per_trade, key=per_trade.get)
    if chosen in (min(thresholds), max(thresholds)):
        print(f"  고른 문턱 {chosen:.2f} 이 견준 문턱의 끝이다. 범위 밖에 더 나은 값이 있을 수 있다", flush=True)
    return chosen


def draw(market: Market, trades: Trades, title: str, path: Path) -> None:
    """수수료 전과 뒤 누적 손익을 한 그림에. 수수료 전 곡선만 있으면 수익으로 읽힌다.

    서버에 한글 글꼴이 없어(fc-list :lang=ko 가 0개) 그림 글자는 영어로 둔다. 한글로 두면 네모로 찍힌다.
    """
    when = market.exchange_time[trades.exit_row].astype("datetime64[ms]")
    figure, axes = plt.subplots(figsize=(9, 4.5), dpi=150)
    for profits, name, color in ((trades.gross, "before fees", "#2a78d6"), (trades.net, "after fees", "#eb6834")):
        running = np.cumsum(profits)
        axes.plot(when, running, color=color, linewidth=1.5, label=name)
        if len(running):
            # 0 선이 축 끝까지 가서 끝 글자를 선 위로 올린다
            axes.annotate(f"{name} {running[-1]:+,.0f}", (when[-1], running[-1]), xytext=(4, 2),
                          textcoords="offset points", va="bottom", fontsize=8, color="#52514e")
    axes.axhline(0, color="#9a9994", linewidth=0.8, zorder=0)
    axes.grid(axis="y", color="#e4e3df", linewidth=0.6)
    # 수수료 뒤 손익은 백만 단위라 기본 축은 1e6 을 따로 적는다. 표의 숫자와 같은 꼴로 찍는다
    axes.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    axes.set(title=title, xlabel="exit time (UTC, exchange time)", ylabel="cumulative profit (USDT per 1 BTC)")
    axes.legend(frameon=False, loc="lower left")
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    figure.autofmt_xdate()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    # tee 로 넘기면 출력이 블록 단위로 모여 평가 구간 결과가 실행이 끝날 때까지 안 보인다. 줄마다 내보낸다
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description="예측으로 모의 매매를 돌려 손익과 기준선을 찍는다")
    parser.add_argument("--days-directory", type=Path, default=Path("restored/BTCUSDT"), help="날짜별 복원 .npz 폴더")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--predictions", type=Path, help="모델 갈래가 쓴 예측 .npz. 검증과 평가 날짜가 다 들어 있다")
    source.add_argument("--random-predictions", type=int, metavar="SEED", help="모델 대신 무작위 예측. 시드")
    for name in ("validation", "evaluation"):
        parser.add_argument(f"--{name}", nargs=2, type=date.fromisoformat, metavar=("FIRST", "LAST"),
                            help="무작위 예측의 UTC 날짜, 둘 다 포함. 모델 예측은 옆 result.json 의 분할을 쓴다")
    thresholds = [round(0.4 + 0.05 * step, 2) for step in range(11)]
    parser.add_argument("--thresholds", nargs="+", type=float, default=thresholds, help="검증 구간에서 견줄 신호 문턱")
    parser.add_argument("--random-draws", type=int, default=100, help="거래 수를 맞춘 무작위 기준선을 몇 번 돌릴지")
    parser.add_argument("--minimum-trades", type=int, default=MINIMUM_VALIDATION_TRADES,
                        help="문턱을 견줄 검증 구간 거래 수 하한. 작은 시험 데이터에서만 줄인다")
    parser.add_argument("--figure", type=Path, required=True, help="누적 손익 그림 .png")
    arguments = parser.parse_args()
    if not arguments.figure.parent.is_dir():
        parser.error(f"--figure 의 폴더 {arguments.figure.parent} 가 없다. 다 돈 뒤에야 그림을 쓰므로 먼저 멈춘다")
    rule = TradingRule()
    validation_days, evaluation_days = arguments.validation or VALIDATION, arguments.evaluation or EVALUATION
    if arguments.predictions:
        if arguments.validation or arguments.evaluation:
            parser.error("모델 예측은 옆 result.json 의 분할을 쓴다. --validation, --evaluation 은 무작위 예측에만 준다")
        validation_days, evaluation_days, horizon = model_settings(arguments.predictions)
        rule = TradingRule(horizon_milliseconds=horizon)

    def predictions_for(market: Market, offset: int) -> Predictions:
        if arguments.predictions:
            return load_predictions(arguments.predictions)
        # 검증과 평가에 같은 시드를 쓰면 두 구간 예측이 같은 난수 열에서 나온다
        return random_predictions(market, SAMPLE_EVERY_ROWS, arguments.random_predictions + offset)

    validation = load_market(arguments.days_directory, *validation_days)
    print(f"검증 {' ~ '.join(map(str, validation_days))}, 행 {len(validation.exchange_time):,}", flush=True)
    threshold = choose_threshold(validation, predictions_for(validation, 0), rule, arguments.thresholds,
                                 arguments.minimum_trades)
    del validation  # 하루치도 평가 구간을 읽기 전에 놓는다

    market = load_market(arguments.days_directory, *evaluation_days)
    predictions = predictions_for(market, 1)
    first, last = evaluation_days
    seed = arguments.random_predictions
    source = arguments.predictions or f"무작위(시드 검증 {seed}, 평가 {seed + 1})"
    print(f"평가 {first} ~ {last}, 예측 {source}, 행 {len(market.exchange_time):,}, "
          f"쓸 수 있는 행 {market.tradable.mean():.4%}, "
          f"고른 문턱 {threshold:.2f}, horizon {rule.horizon_milliseconds / 1000:g}초, 수수료 {rule.fee_rate:.3%}, "
          f"공매도 {'허용' if rule.allow_short else '없음'}, 같은 신호에 연장 {'함' if rule.extend_on_same_signal else '안 함'}")
    signal_rows, sides = signals_of(market, predictions, threshold)
    model = {delay: walk(market, signal_rows, sides, replace(rule, delay_rows=delay)) for delay in DELAYS}
    for delay, trades in model.items():
        print(line(f"모델, {delay}행 뒤 체결", measure(trades)))
    print(fill_line(market, model[1]))
    print(line("항상 보유", measure(always_hold(market, rule.fee_rate))))
    imbalance_rule = replace(rule, horizon_milliseconds=IMBALANCE_HOLD_MILLISECONDS)
    imbalance = walk(market, *imbalance_signals(market, IMBALANCE_THRESHOLD), imbalance_rule)
    print(line(f"수량 불균형 {IMBALANCE_THRESHOLD} 규칙, {IMBALANCE_HOLD_MILLISECONDS / 1000:g}초", measure(imbalance)))
    print(fill_line(market, imbalance))

    # 무작위 신호는 모델이 예측을 낸 행에만 넣는다. 모델이 고를 수 있던 시점 가운데에서 우연히 고른 것과 견준다
    candidate_rows, _ = prediction_rows(market, predictions)
    target, gross = len(model[1].side), measure(model[1]).gross_total
    draws = [measure(random_trades(market, candidate_rows, target, rule, seed))
             for seed in range(arguments.random_draws)]
    counts = np.array([result.trades for result in draws])
    totals, nets = (np.array([getattr(result, name) for result in draws]) for name in ("gross_total", "net_total"))
    print(f"무작위 {len(draws)}번(시드 0 ~ {len(draws) - 1}, 거래 {counts.min():,} ~ {counts.max():,}, 목표 {target:,}): "
          f"수수료 전 평균 {totals.mean():+,.2f} 표준편차 {totals.std():,.2f} 5% ~ 95% {np.percentile(totals, 5):+,.2f} ~ "
          f"{np.percentile(totals, 95):+,.2f}, 수수료 뒤 평균 {nets.mean():+,.2f}. "
          f"모델 수수료 전 {gross:+,.2f} 보다 큰 무작위 {np.mean(totals > gross):.0%}")
    draw(market, model[1], f"paper trading {first} ~ {last}, threshold {threshold:.2f}", arguments.figure)
    print(f"그림 {arguments.figure}")


if __name__ == "__main__":
    main()