from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from test_replay_comparison import store_across_midnight
from tickfold.paper_trading.baselines import always_hold, imbalance_signals, random_trades
from tickfold.paper_trading.market import Market, load_market
from tickfold.paper_trading.performance import max_drawdown, measure
from tickfold.paper_trading.walk import HORIZON, OPPOSITE, RANGE_END, UNTRADABLE, Trades, TradingRule, walk
from tickfold.replay.depth20 import ASK, BID, PRICE, QUANTITY
from tickfold.replay.restored_day import load_restored_day, restore_day, save_restored_day

BUY, SELL = 1, -1


def market(rows: int, times=None, untradable=(), resumed=()) -> Market:
    """r 행의 매도 1호가가 100 + r, 매수 1호가가 그보다 1틱 낮은 시장. 거래소 시각은 100ms 간격."""
    best = np.ones((rows, 2, 2))
    best[:, ASK, PRICE] = 100 + np.arange(rows)
    best[:, BID, PRICE] = best[:, ASK, PRICE] - 0.01
    tradable, resumed_after_gap = np.ones(rows, dtype=bool), np.zeros(rows, dtype=bool)
    tradable[list(untradable)], resumed_after_gap[list(resumed)] = False, True
    best[~tradable] = np.nan  # 동기화 안 된 행은 복원 배열에서 NaN 이다
    exchange_time = np.arange(rows, dtype=np.int64) * 100 if times is None else np.array(times, dtype=np.int64)
    return Market(np.array(["2026-10-10"], dtype="datetime64[D]"), np.array([0, rows]), exchange_time, best,
                  tradable, resumed_after_gap)


def walked(market: Market, signals: list[tuple[int, int]], **rule) -> Trades:
    rows, sides = (np.array(column, dtype=np.int64) for column in zip(*signals))
    return walk(market, rows, sides, TradingRule(**{"horizon_milliseconds": 300, **rule}))


def columns(trades: Trades) -> list[tuple[int, int, int, int]]:
    picked = (trades.side, trades.entry_row, trades.exit_row, trades.exit_reason)
    return list(zip(*(column.tolist() for column in picked)))


def test_buy_fills_at_the_ask_delay_rows_after_the_signal_and_sells_at_the_bid_delay_rows_after_the_horizon():
    trades = walked(market(10), [(2, BUY)])

    # 신호 2행(200ms) 에서 300ms 뒤인 5행이 청산 시점이고 그다음 행에서 판다
    assert columns(trades) == [(BUY, 3, 6, HORIZON)]
    assert (trades.entry_price[0], trades.exit_price[0]) == (103.0, 105.99)
    assert trades.fee[0] == pytest.approx(0.001 * (103.0 + 105.99))
    assert columns(walked(market(10), [(2, BUY)], delay_rows=2)) == [(BUY, 4, 7, HORIZON)]


def test_horizon_is_counted_in_exchange_time_so_an_uneven_frame_gap_moves_the_exit_row():
    trades = walked(market(8, times=[0, 100, 200, 250, 600, 700, 800, 900]), [(0, BUY)])

    # 행 수로 세면 3행이 300ms 뒤지만 거래소 시각으로는 250ms 라서 그다음 행이 청산 시점이다
    assert columns(trades) == [(BUY, 1, 5, HORIZON)]


def test_opposite_signal_only_closes_and_a_sell_opens_a_short_only_when_short_is_allowed():
    signals = [(2, BUY), (4, SELL), (6, SELL), (9, BUY)]

    shorting = walked(market(12), signals, horizon_milliseconds=10_000)
    spot = walked(market(12), signals, horizon_milliseconds=10_000, allow_short=False)

    assert columns(shorting) == [(BUY, 3, 5, OPPOSITE), (SELL, 7, 10, OPPOSITE)]
    # 공매도는 매수 1호가에 팔고 매도 1호가에 되산다
    assert shorting.gross.tolist() == pytest.approx([104.99 - 103.0, 106.99 - 110.0])
    # 현물이면 팔자 신호는 든 매수를 닫기만 하고, 끝까지 든 매수는 범위 끝의 마지막 쓸 수 있는 행에서 닫는다
    assert columns(spot) == [(BUY, 3, 5, OPPOSITE), (BUY, 10, 11, RANGE_END)]


def test_holding_into_a_gap_closes_at_the_first_tradable_row_after_it_and_signals_on_untradable_rows_are_skipped():
    gapped = market(14, untradable=[5, 6], resumed=[5])

    trades = walked(gapped, [(2, BUY), (4, BUY), (6, BUY), (8, SELL)], horizon_milliseconds=10_000)

    # 5행 앞에서 순번이 끊겼다. 끊긴 걸 본 5행 다음 행(6)도 못 쓰는 행이라 다시 쓸 수 있게 된 7행에서 판다.
    # 4행 사자는 든 동안 온 같은 쪽 신호라 안 세고, 6행 사자는 못 쓰는 행이라 건너뛴다
    assert columns(trades) == [(BUY, 3, 7, UNTRADABLE), (SELL, 9, 13, RANGE_END)]
    assert trades.skipped_signals == 1
    # 체결 행이 못 쓰는 행이어도 안 든다
    assert walked(gapped, [(4, BUY)]).skipped_signals == 1
    # 반대 신호가 갭 바로 앞에 와도 닫는 체결은 갭 뒤라 갭에 걸린 거래로 센다
    assert columns(walked(gapped, [(2, BUY), (4, SELL)], horizon_milliseconds=10_000)) == [(BUY, 3, 7, UNTRADABLE)]
    # 갭 바로 뒤에 스냅샷이 끼어 동기화된 행도 갭 자리라서, 그 행을 지나는 포지션은 갭에 걸린 거래다
    resynced = market(10, resumed=[5])
    assert columns(walked(resynced, [(2, BUY)], horizon_milliseconds=10_000)) == [(BUY, 3, 6, UNTRADABLE)]


def test_a_same_side_signal_restarts_the_horizon_only_when_extension_is_on():
    signals = [(2, BUY), (4, BUY)]

    assert columns(walked(market(10), signals)) == [(BUY, 3, 6, HORIZON)]
    assert columns(walked(market(10), signals, extend_on_same_signal=True)) == [(BUY, 3, 8, HORIZON)]


def test_performance_counts_ticks_after_rounding_and_draws_down_from_the_best_point_including_the_start():
    side = np.array([BUY, BUY, SELL, BUY], dtype=np.int8)
    entry_price = np.array([100.0, 100.0, 100.0, 100.0])
    # 100.01 - 100.0 같은 계산 값은 0.01 과 정확히 같지 않다. 틱으로 반올림해 세야 1틱 이익과 -1틱 손실이 맞게 잡힌다
    exit_price = np.array([100.01, 99.99, 100.03, 100.0])
    rows = np.arange(4)
    trades = Trades(side, rows, rows, rows, entry_price, exit_price, np.array([0, 1, 1, 3], dtype=np.int8),
                    np.full(4, 0.02), 7)

    result = measure(trades)

    assert (result.trades, result.skipped_signals) == (4, 7)
    assert result.gross_total == pytest.approx(0.01 - 0.01 - 0.03 + 0.0)
    assert result.net_total == pytest.approx(-0.03 - 4 * 0.02)
    assert result.gross_mean_bp == pytest.approx(-0.03 / 4 / 100 * 1e4)
    # 누적 수수료 전 손익 0, 0.01, 0, -0.03, -0.03 에서 최고점 0.01 부터 0.04 내려갔다
    assert result.gross_max_drawdown == pytest.approx(0.04)
    assert (result.net_mean_bp, result.net_max_drawdown) == pytest.approx((-0.11 / 4 / 100 * 1e4, 0.11))
    # 첫 거래부터 잃으면 시작점 0 이 최고점이다. 뒤 최고점 0.03 에서 내려간 0.01 보다 깊다
    assert max_drawdown(np.array([-0.02, 0.05, -0.01])) == pytest.approx(0.02)
    assert (result.win_rate, result.net_win_rate, result.one_tick_loss_rate) == (0.25, 0.0, 0.25)
    assert result.exits == {"horizon": 1, "opposite": 2, "untradable": 0, "range_end": 1}


def saved_fixture_days(tmp_path) -> Path:
    """픽스처를 28일 자정에 걸쳐 옮겨 27일과 28일을 복원해 쓴 폴더."""
    symbol_directory, _ = store_across_midnight(tmp_path / "raw")
    days_directory = tmp_path / "restored"
    days_directory.mkdir()
    for day in (27, 28):
        save_restored_day(restore_day(symbol_directory, date(2026, 9, day)), days_directory / f"2026-09-{day}.npz")
    return days_directory


def test_market_joins_saved_days_in_row_order_and_keeps_only_the_best_level_of_rows_valid_on_both_columns(tmp_path):
    days_directory = saved_fixture_days(tmp_path)
    # 픽스처에는 동기화됐지만 범위 밖인 행이 없어서 28일 홀수 행을 범위 밖으로 바꿔 쓴다
    day = load_restored_day(days_directory / "2026-09-28.npz")
    even = np.arange(len(day.valid.synced)) % 2 == 0
    narrowed = replace(day.valid, inside_snapshot_range=day.valid.inside_snapshot_range & even)
    save_restored_day(replace(day, valid=narrowed), days_directory / "2026-09-28.npz")
    saved = [load_restored_day(days_directory / f"2026-09-{day}.npz") for day in (27, 28)]

    joined = load_market(days_directory, date(2026, 9, 27), date(2026, 9, 28))

    books = np.concatenate([day.top.book for day in saved])
    assert joined.day_first_rows.tolist() == [0, len(saved[0].top.book), len(books)]
    assert joined.days.tolist() == [date(2026, 9, 27), date(2026, 9, 28)]
    assert np.array_equal(joined.best[:, ASK, QUANTITY], books[:, ASK, 0, QUANTITY], equal_nan=True)
    assert np.array_equal(joined.best[:, BID, PRICE], books[:, BID, 0, PRICE], equal_nan=True)
    synced_inside = np.concatenate([day.valid.synced & day.valid.inside_snapshot_range for day in saved])
    assert np.array_equal(joined.tradable, synced_inside) and 0 < joined.tradable.sum() < len(books)
    with pytest.raises(FileNotFoundError):
        load_market(days_directory, date(2026, 9, 27), date(2026, 9, 29))


def test_random_baseline_reaches_the_target_trade_count_and_repeats_with_the_same_seed():
    candidates = np.arange(0, 2000, 10)
    # 거래 하나가 후보 세 개를 덮어서 신호 수와 거래 수가 다르다
    rule = TradingRule(horizon_milliseconds=3_000)

    trades = random_trades(market(2000), candidates, 40, rule, seed=3)

    assert len(trades.side) == 40 and set(trades.side.tolist()) == {BUY, SELL}
    assert np.array_equal(random_trades(market(2000), candidates, 40, rule, seed=3).entry_row, trades.entry_row)
    assert not np.array_equal(random_trades(market(2000), candidates, 40, rule, seed=4).entry_row, trades.entry_row)


def test_always_hold_keeps_its_position_through_a_gap_and_imbalance_signals_come_only_from_tradable_rows():
    gapped = market(10, untradable=[4], resumed=[4])
    gapped.best[[2, 4, 6], BID, QUANTITY] = [19.0, 19.0, 1.0]
    gapped.best[[2, 4, 6], ASK, QUANTITY] = [1.0, 1.0, 19.0]

    rows, sides = imbalance_signals(gapped, 0.8)
    hold = always_hold(gapped, 0.001)

    assert (rows.tolist(), sides.tolist()) == ([2, 6], [BUY, SELL])
    assert columns(hold) == [(BUY, 0, 9, RANGE_END)]
    assert (hold.entry_price[0], hold.exit_price[0]) == (100.0, 108.99)