"""재생 시계를 계산한다. 재생은 프레임 번호가 아니라 거래소 시각을 따라간다. 시계를 직접 재지 않는 순수 함수라서 화면 없이 시험한다."""

from dataclasses import dataclass

import numpy as np

from tickfold.replay.segment import ReplaySegment


@dataclass(frozen=True)
class GapPassage:
    """재생 시계가 지나고 있는 갭. 화면이 멈춘 것이 고장이 아니라 받은 프레임이 없어서라는 안내에 쓴다."""

    gap_seconds: float  # 갭의 길이
    elapsed_seconds: float  # 갭 직전 프레임에서 재생 시계까지. 거래소 시각으로 잰다


def clock_after(anchor_clock: int, elapsed_seconds: float, speed: float) -> int:
    """기준점에서 실제 시간이 elapsed_seconds 만큼 흐른 뒤의 재생 시계(거래소 시각, 밀리초)를 돌려준다."""
    # 프레임 수를 세어 넘기면 프레임 간격이 고르지 않아 시간이 틀어지고, 갭이 한 걸음에 지나가 버린다
    return anchor_clock + round(elapsed_seconds * 1000 * speed)


def frame_at(exchange_time: np.ndarray, clock: int) -> int:
    """거래소 시각이 재생 시계 이하인 마지막 프레임을 찾는다. 다시 그리는 것이 늦으면 그 사이 프레임은 건너뛴다."""
    return max(int(np.searchsorted(exchange_time, clock, side="right")) - 1, 0)


def gap_at(segment: ReplaySegment, clock: int) -> GapPassage | None:
    """재생 시계가 갭 안에 있으면 그 갭의 길이와 지금 몇 초째인지를 돌려준다."""
    # 갭 직전 프레임의 시각은 그 프레임을 받은 때라서 갭 안이 아니다
    inside = np.flatnonzero((segment.gap_last_exchange_time < clock) & (clock < segment.gap_next_exchange_time))
    if len(inside) == 0:
        return None
    last_exchange_time = int(segment.gap_last_exchange_time[inside[0]])
    next_exchange_time = int(segment.gap_next_exchange_time[inside[0]])
    return GapPassage((next_exchange_time - last_exchange_time) / 1000, (clock - last_exchange_time) / 1000)