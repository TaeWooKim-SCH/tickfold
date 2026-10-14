"""재생 화면의 숫자 띠 값과 그림에 넘길 값을 만든다. Streamlit 을 모르는 순수 함수라서 화면 없이 시험한다."""

from dataclasses import dataclass

import numpy as np

from tickfold.replay.depth20 import ASK, BID, PRICE, QUANTITY
from tickfold.replay.segment import ReplaySegment

# 가장 큰 주문에 축을 맞추면 나머지 막대가 더 짧아진다. 이 백분위수를 넘는 수량은 막대를 잘라 그린다
QUANTITY_CEILING_PERCENTILE = 99
# mid 선은 구간마다 한 번 브라우저로 간다. 솎지 않으면 한 시간 구간이 3만 5천 점이라 보내는 양과 그리는 시간이 는다
MID_LINE_POINTS = 2000


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


@dataclass(frozen=True)
class MidLine:
    """mid 그림에서 재생 중에 안 바뀌는 것. 구간마다 한 번 만들어 브라우저에 넘기고, 재생 중에는 브라우저가 세로선만 옮긴다."""

    time: np.ndarray  # 선의 거래소 시각, UTC 문자열. 갭 직후 프레임 앞에는 선을 끊는 빈 점이 끼어 있다
    mid: np.ndarray  # 빈 점은 nan
    gap_starts: np.ndarray  # 갭을 칠할 띠의 양 끝, UTC 문자열
    gap_ends: np.ndarray


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


def thin_mid_line(segment: ReplaySegment) -> MidLine:
    """구간 전체의 mid 선을 MID_LINE_POINTS 개 안팎으로 솎고 갭에서 끊는다. 갭을 칠할 띠도 같이 잡는다."""
    exchange_time = segment.exchange_time
    mid = (segment.book[:, BID, 0, PRICE] + segment.book[:, ASK, 0, PRICE]) / 2
    # 갭 안에서 자른 구간은 갭이 첫·끝 프레임 밖으로 나가 있다. 그대로 칠하면 시간 축이 프레임 없는 데까지 늘어난다
    gap_starts = utc_text(np.clip(segment.gap_last_exchange_time, exchange_time[0], exchange_time[-1]))
    gap_ends = utc_text(np.clip(segment.gap_next_exchange_time, exchange_time[0], exchange_time[-1]))
    after_gap = np.searchsorted(exchange_time, segment.gap_next_exchange_time)
    kept = np.arange(len(mid))
    if len(mid) > MID_LINE_POINTS:
        # 몇 번째마다 하나씩 뽑으면 한두 프레임짜리 급등락이 사라진다. 묶음마다 가장 낮은 점과 가장 높은 점을 남긴다
        bunch = -(-len(mid) // (MID_LINE_POINTS // 2))
        # 끝 묶음의 빈 자리는 끝 값으로 채운다. argmin 과 argmax 는 같은 값 가운데 앞의 것을 골라서 채운 자리는 안 뽑힌다
        bunches = np.pad(mid, (0, -len(mid) % bunch), mode="edge").reshape(-1, bunch)
        first_of_bunch = np.arange(len(bunches)) * bunch
        # 갭 양쪽 프레임과 구간의 첫·끝 프레임은 안 뽑혀도 남긴다. 빠지면 선이 갭보다 넓게 끊기고 시간 축이 줄어든다
        edges = np.concatenate([after_gap - 1, after_gap, [0, len(mid) - 1]]).clip(0, len(mid) - 1)
        lowest, highest = first_of_bunch + bunches.argmin(axis=1), first_of_bunch + bunches.argmax(axis=1)
        kept = np.unique(np.concatenate([lowest, highest, edges]))
    # 갭을 건너 선을 이으면 받지 못한 동안의 가격을 아는 것처럼 보인다. 갭 직후 프레임 앞에 빈 점을 끼워 선을 끊는다
    breaks = np.searchsorted(kept, after_gap)
    time = np.insert(utc_text(exchange_time[kept]), breaks, gap_ends)
    return MidLine(time, np.insert(mid[kept], breaks, np.nan), gap_starts, gap_ends)