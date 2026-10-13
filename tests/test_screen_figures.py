import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

# 화면용 의존성 그룹(screen)을 안 깐 환경에서는 이 파일을 건너뛴다
pytest.importorskip("plotly")

from test_replay_segment import exchange_time_of, fixture_lines, write_day

from tickfold.replay.depth20 import ASK, BID, QUANTITY
from tickfold.replay.segment import ReplaySegment, read_segment
from tickfold.screen.figures import SegmentSummary, book_figure, find_quantity_ceiling, mid_figure, summarize


def read_with_a_gap(symbol_directory: Path, first: int, last: int) -> tuple[ReplaySegment, list[bytes]]:
    """픽스처의 depth 100 ~ 102번 줄을 갭으로 만들고, first 번 줄의 시각부터 last 번 줄의 시각 앞까지를 읽는다."""
    depth, depth20 = fixture_lines(b"depth"), fixture_lines(b"depth20")
    # 갭에서는 두 스트림이 같이 빈다. depth 셋과 같은 번호의 depth20 을 같이 뺀다
    lost = {json.loads(line)["m"]["data"]["u"] for line in depth[100:103]}
    kept_depth20 = [line for line in depth20 if json.loads(line)["m"]["data"]["lastUpdateId"] not in lost]
    write_day(symbol_directory, "2026-09-27", "06", depth[:100] + depth[103:], kept_depth20)
    return read_segment(symbol_directory, exchange_time_of(depth[first]), exchange_time_of(depth[last])), depth


def raw_depth20(segment: ReplaySegment) -> list[dict]:
    """프레임마다 거래소가 보낸 원본 depth20 을 찾는다. 그림을 배열이 아니라 원본 문자열과 견주려는 것이다."""
    records = (json.loads(line)["m"]["data"] for line in fixture_lines(b"depth20"))
    by_identifier = {record["lastUpdateId"]: record for record in records}
    return [by_identifier[identifier] for identifier in segment.last_update_id.tolist()]


def on_the_time_axis(time: datetime) -> str:
    return time.replace(tzinfo=None).isoformat(timespec="milliseconds")


def test_book_figure_draws_forty_bars_with_the_prices_and_quantities_of_that_frame(tmp_path):
    segment, _ = read_with_a_gap(tmp_path, 50, 400)
    frame = 123
    raw = raw_depth20(segment)[frame]
    nothing_is_cut = segment.book[..., QUANTITY].max()

    figure = book_figure(segment, frame, nothing_is_cut)

    asks, bids = figure.data
    assert asks.marker.color != bids.marker.color
    for bars, name, levels in ((asks, "매도", raw["asks"]), (bids, "매수", raw["bids"])):
        assert bars.orientation == "h"
        assert list(bars.y) == [f"{name} {level}" for level in range(1, 21)]
        assert bars.x.tolist() == [float(quantity) for _, quantity in levels]
        # 글자는 원본 문자열과 같은 수여야 한다. 자릿수가 모자라 반올림되면 여기서 걸린다
        shown = [[Decimal(number) for number in text.replace(",", "").split()] for text in bars.text]
        assert shown == [[Decimal(price), Decimal(quantity)] for price, quantity in levels]
    top_to_bottom = list(figure.layout.yaxis.categoryarray)[::-1]
    assert top_to_bottom == [f"매도 {level}" for level in range(20, 0, -1)] + [f"매수 {level}" for level in range(1, 21)]
    time = datetime.fromtimestamp(segment.exchange_time[frame] / 1000, tz=timezone.utc)
    best_bid, best_ask = Decimal(raw["bids"][0][0]), Decimal(raw["asks"][0][0])
    assert figure.layout.title.text == (
        f"{on_the_time_axis(time).replace('T', ' ')} UTC"
        f" · mid {(best_bid + best_ask) / 2:,.3f} · 스프레드 {best_ask - best_bid:.2f}"
    )


def test_bar_longer_than_the_ceiling_is_cut_while_its_text_keeps_the_quantity(tmp_path):
    segment, _ = read_with_a_gap(tmp_path, 50, 400)
    quantities = segment.book[..., QUANTITY]
    ceiling = find_quantity_ceiling(segment)
    # 상한은 구간 전체의 99 백분위수라서 넘는 막대가 백에 하나쯤이다
    assert 0.009 < (quantities > ceiling).mean() < 0.011
    frame, side, level = np.argwhere(quantities > ceiling)[0].tolist()
    raw_levels = raw_depth20(segment)[frame]["asks" if side == ASK else "bids"]

    figure = book_figure(segment, frame, ceiling)

    bars = {ASK: figure.data[0], BID: figure.data[1]}[side]
    assert bars.x.tolist() == [min(float(quantity), ceiling) for _, quantity in raw_levels]
    assert bars.x[level] == ceiling < float(raw_levels[level][1])
    assert Decimal(bars.text[level].split()[-1]) == Decimal(raw_levels[level][1])
    # 축의 절반을 넘는 막대만 글자를 막대 안에 적는다. 잘린 막대는 축 끝까지 차므로 안에 있어야 보인다
    wanted = ["inside" if float(quantity) > ceiling / 2 else "outside" for _, quantity in raw_levels]
    assert bars.textposition.tolist() == wanted and wanted[level] == "inside"
    assert tuple(figure.layout.xaxis.range) == (0, ceiling)


def test_mid_figure_shades_the_gap_breaks_the_line_there_and_marks_the_chosen_frame(tmp_path):
    segment, depth = read_with_a_gap(tmp_path, 50, 400)
    frame = 200

    figure = mid_figure(segment, frame)

    (line,) = figure.data
    gap, chosen = figure.layout.shapes
    drawn = ~np.isnan(line.y)
    best_prices = [(float(raw["bids"][0][0]), float(raw["asks"][0][0])) for raw in raw_depth20(segment)]
    assert line.y[drawn].tolist() == [(best_bid + best_ask) / 2 for best_bid, best_ask in best_prices]
    # 시간 축은 거래소 시각이고 UTC 다. 이 서버의 시간대(KST)로 바뀌어 나오면 여기서 걸린다
    times = [datetime.fromtimestamp(time / 1000, tz=timezone.utc) for time in segment.exchange_time.tolist()]
    assert line.x[drawn].tolist() == [on_the_time_axis(time) for time in times]
    before_the_gap, after_the_gap = (on_the_time_axis(exchange_time_of(depth[index])) for index in (99, 103))
    (line_break,) = np.flatnonzero(~drawn).tolist()
    assert (line.x[line_break - 1], line.x[line_break + 1]) == (before_the_gap, after_the_gap)
    assert (gap.type, gap.x0, gap.x1) == ("rect", before_the_gap, after_the_gap)
    assert (chosen.type, chosen.x0, chosen.x1) == ("line", line.x[drawn][frame], line.x[drawn][frame])


def test_gap_reaching_outside_the_segment_is_shaded_only_up_to_the_edge_frame(tmp_path):
    # 갭 안에서 시작하는 구간이라 갭 직전 프레임이 구간 밖에 있다
    segment, depth = read_with_a_gap(tmp_path, 101, 400)
    assert segment.gap_last_exchange_time[0] < segment.exchange_time[0]

    figure = mid_figure(segment, 0)

    first_frame = on_the_time_axis(exchange_time_of(depth[103]))
    gap = figure.layout.shapes[0]
    assert (gap.x0, gap.x1) == (first_frame, first_frame)
    assert min(figure.data[0].x) == first_frame

    # 갭 안에서 끝나는 구간은 갭 직후 프레임이 구간 밖에 있다
    segment, _ = read_with_a_gap(tmp_path, 50, 102)
    figure = mid_figure(segment, 0)
    last_frame = on_the_time_axis(exchange_time_of(depth[99]))
    assert (figure.layout.shapes[0].x0, figure.layout.shapes[0].x1) == (last_frame, last_frame)
    assert max(figure.data[0].x) == last_frame


def test_summary_holds_the_frame_count_the_gap_and_the_longest_delivery_delay(tmp_path):
    segment, depth = read_with_a_gap(tmp_path, 50, 400)
    before_the_gap, after_the_gap = (json.loads(depth[index])["m"]["data"] for index in (99, 103))
    received_and_exchange_times = zip(segment.received_at.tolist(), segment.exchange_time.tolist())

    assert summarize(segment) == SegmentSummary(
        first_exchange_time=json.loads(depth[50])["m"]["data"]["E"],
        last_exchange_time=json.loads(depth[399])["m"]["data"]["E"],
        frame_count=347,
        gap_count=1,
        gap_seconds=(after_the_gap["E"] - before_the_gap["E"]) / 1000,
        missed_updates=after_the_gap["U"] - before_the_gap["u"] - 1,
        longest_delivery_delay_seconds=max(
            (received - exchange * 1_000_000) / 1e9 for received, exchange in received_and_exchange_times
        ),
    )