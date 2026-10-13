"""재생 화면의 그림과 숫자 띠 값을 만든다. Streamlit 을 모르는 순수 함수라서 화면 없이 시험하고 미리보기 파일로도 쓴다."""

from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go

from tickfold.replay.depth20 import ASK, BID, LEVELS, PRICE, QUANTITY
from tickfold.replay.segment import ReplaySegment

# 매도 파랑, 매수 빨강은 국내 거래소 호가창의 색이다. 초록·빨강 짝은 적록 색각에서 구별이 어려워서 안 쓴다
ASK_COLOR, BID_COLOR = "#2a78d6", "#e34948"
MID_COLOR, SELECTED_FRAME_COLOR = "#008300", "#9085e9"
GAP_EDGE_COLOR, GAP_FILL_COLOR = "#898781", "rgba(137, 135, 129, 0.35)"
# 가장 큰 주문에 축을 맞추면 나머지 막대가 더 짧아진다. 이 백분위수를 넘는 수량은 막대를 잘라 그린다
QUANTITY_CEILING_PERCENTILE = 99
# 막대 40개에 글자가 한 줄씩 붙는다. 위아래 여백을 빼고 한 줄에 16픽셀이 돌아가는 높이다
BOOK_HEIGHT, MID_HEIGHT = 760, 380
MARGIN = dict(l=70, r=20, t=60, b=50)
SIDE_NAMES = {ASK: "매도", BID: "매수"}


@dataclass(frozen=True)
class SegmentSummary:
    """화면 맨 위 숫자 띠에 올리는 값. 글자로 바꾸는 것은 화면이 한다."""

    first_exchange_time: int  # 첫 프레임의 거래소 시각, 밀리초
    last_exchange_time: int  # 끝 프레임의 거래소 시각, 밀리초
    frame_count: int
    gap_count: int
    gap_seconds: float  # 갭 길이의 합. 구간 밖으로 나간 부분도 센다
    missed_updates: int
    longest_delivery_delay_seconds: float  # 수신 시각 - 거래소 시각 가운데 가장 큰 값


def summarize(segment: ReplaySegment) -> SegmentSummary:
    """구간 하나를 숫자 띠에 올릴 값으로 줄인다."""
    gap_lengths = segment.gap_next_exchange_time - segment.gap_last_exchange_time
    delivery_delay = segment.received_at - segment.exchange_time * 1_000_000
    return SegmentSummary(
        first_exchange_time=int(segment.exchange_time[0]),
        last_exchange_time=int(segment.exchange_time[-1]),
        frame_count=len(segment.exchange_time),
        gap_count=len(gap_lengths),
        gap_seconds=float(gap_lengths.sum() / 1e3),
        missed_updates=int(segment.gap_missed_updates.sum()),
        longest_delivery_delay_seconds=float(delivery_delay.max() / 1e9),
    )


def utc_text(exchange_time: np.ndarray | int) -> np.ndarray | str:
    """거래소 시각(밀리초)을 UTC 문자열로 바꾼다. 그림의 시간 축과 화면의 글자가 같은 값을 쓴다."""
    # 숫자로 넘기면 plotly 가 브라우저의 시간대로 옮겨 그린다. 시간대 없는 문자열은 적힌 그대로 그린다
    return np.datetime_as_string(np.asarray(exchange_time).astype("datetime64[ms]"), unit="ms")


def find_quantity_ceiling(segment: ReplaySegment) -> float:
    """수량 축의 상한을 구간 전체에서 잡는다. 프레임마다 잡으면 프레임을 넘길 때 축이 출렁인다."""
    return float(np.percentile(segment.book[..., QUANTITY], QUANTITY_CEILING_PERCENTILE))


def book_figure(segment: ReplaySegment, frame: int, quantity_ceiling: float) -> go.Figure:
    """한 프레임의 상위 20레벨을 가로 막대로 그린다. 위에서부터 매도 20 … 매도 1, 매수 1 … 매수 20 이다."""
    book = segment.book[frame]
    level_names = {side: [f"{name} {level}" for level in range(1, LEVELS + 1)] for side, name in SIDE_NAMES.items()}
    figure = go.Figure()
    for side, color in ((ASK, ASK_COLOR), (BID, BID_COLOR)):
        prices, quantities = book[side, :, PRICE], book[side, :, QUANTITY]
        figure.add_bar(
            x=np.minimum(quantities, quantity_ceiling),
            y=level_names[side],
            orientation="h",
            marker_color=color,
            # 잘린 막대도 글자와 마우스를 올렸을 때의 값은 원래 수량이다
            text=[f"{price:,.2f}  {quantity:.5f}" for price, quantity in zip(prices, quantities)],
            hoverinfo="y+text",
            # 축의 절반을 넘는 막대는 오른쪽에 글자 자리가 모자라서 막대 안에 적는다
            textposition=np.where(quantities > quantity_ceiling / 2, "inside", "outside"),
            # 그냥 두면 plotly 가 글자를 막대 굵기에 맞춰 줄여 그린다. 40줄이라 막대가 글자보다 가늘다
            constraintext="none",
        )
    best_bid, best_ask = book[BID, 0, PRICE], book[ASK, 0, PRICE]
    time = str(utc_text(segment.exchange_time[frame])).replace("T", " ")
    figure.update_layout(
        title_text=f"{time} UTC · mid {(best_bid + best_ask) / 2:,.3f} · 스프레드 {best_ask - best_bid:.2f}",
        xaxis=dict(title_text=f"수량 (막대는 {quantity_ceiling:.2f} 에서 자름)", range=[0, quantity_ceiling]),
        yaxis=dict(
            # 범주 축은 아래에서 위로 쌓인다. 맨 아래가 매수 20, 맨 위가 매도 20 이다
            categoryarray=level_names[BID][::-1] + level_names[ASK],
            categoryorder="array",
            # 눈금을 plotly 에 맡기면 한 줄이 14픽셀보다 낮아질 때 레벨 이름을 건너뛰며 적는다
            tickmode="linear",
        ),
        showlegend=False,
        bargap=0.15,
        height=BOOK_HEIGHT,
        margin=MARGIN,
    )
    return figure


def mid_figure(segment: ReplaySegment, frame: int) -> go.Figure:
    """구간 전체의 mid 를 거래소 시각 위에 그린다. 갭은 띠로 칠하고 고른 프레임에는 세로선을 긋는다."""
    exchange_time = segment.exchange_time
    mid = (segment.book[:, BID, 0, PRICE] + segment.book[:, ASK, 0, PRICE]) / 2
    # 갭 안에서 자른 구간은 갭이 첫·끝 프레임 밖으로 나가 있다. 그대로 칠하면 시간 축이 프레임 없는 데까지 늘어난다
    gap_starts = utc_text(np.clip(segment.gap_last_exchange_time, exchange_time[0], exchange_time[-1]))
    gap_ends = utc_text(np.clip(segment.gap_next_exchange_time, exchange_time[0], exchange_time[-1]))
    # 갭을 건너 선을 이으면 받지 못한 동안의 가격을 아는 것처럼 보인다. 갭 직후 프레임 앞에 빈 점을 끼워 선을 끊는다
    after_gap = np.searchsorted(exchange_time, segment.gap_next_exchange_time)
    figure = go.Figure(
        go.Scatter(
            x=np.insert(utc_text(exchange_time), after_gap, gap_ends),
            y=np.insert(mid, after_gap, np.nan),
            mode="lines",
            line=dict(color=MID_COLOR, width=2),
            hovertemplate="%{x|%H:%M:%S.%L} UTC<br>mid %{y:,.3f}<extra></extra>",
        )
    )
    for gap_start, gap_end in zip(gap_starts, gap_ends):
        # 테두리가 없으면 한 시간짜리 구간에서 1초 갭은 한 픽셀이 안 돼 안 보인다
        edge = dict(color=GAP_EDGE_COLOR, width=1)
        figure.add_vrect(x0=gap_start, x1=gap_end, fillcolor=GAP_FILL_COLOR, line=edge, layer="below")
    figure.add_vline(x=utc_text(exchange_time[frame]), line=dict(color=SELECTED_FRAME_COLOR, width=2))
    figure.update_layout(
        title_text="mid · 회색 띠는 갭, 세로선은 고른 프레임",
        xaxis_title_text="거래소 시각 (UTC)",
        yaxis_tickformat=",.2f",
        showlegend=False,
        height=MID_HEIGHT,
        margin=MARGIN,
    )
    return figure