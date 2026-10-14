import numpy as np

from tickfold.replay.segment import ReplaySegment
from tickfold.screen.playback import GapPassage, clock_after, frame_at, gap_at


def test_frame_follows_the_exchange_time_at_any_speed_and_stays_on_the_last_frame_past_the_end():
    # 프레임 간격이 고르지 않다. 프레임 수를 세어 넘기면 0.45초 뒤에 4번이 아니라 다른 프레임이 나온다
    exchange_time = np.array([10_000, 10_100, 10_200, 10_300, 10_450, 10_500])

    def frame_after(start_frame: int, elapsed_seconds: float, speed: int) -> int:
        return frame_at(exchange_time, clock_after(int(exchange_time[start_frame]), elapsed_seconds, speed))

    assert [frame_after(0, elapsed, 1) for elapsed in (0, 0.099, 0.1, 0.25, 0.449, 0.45)] == [0, 0, 1, 2, 3, 4]
    assert [frame_after(0, elapsed, 10) for elapsed in (0.0099, 0.01, 0.025, 0.045)] == [0, 1, 2, 4]
    assert [frame_after(2, elapsed, 5) for elapsed in (0.019, 0.02, 0.05)] == [2, 3, 4]
    # 끝 프레임을 지난 재생 시계는 끝 프레임에 머문다. 화면은 이것을 보고 재생을 멈춘다
    assert [frame_after(0, elapsed, 1) for elapsed in (0.5, 0.6, 3600)] == [5, 5, 5]
    assert frame_after(4, 100, 30) == 5
    # 첫 프레임보다 앞선 시계에 맞는 프레임은 없다. 끝 프레임(-1)이 아니라 첫 프레임을 보인다
    assert frame_at(exchange_time, 9_999) == 0


def test_clock_inside_a_gap_stays_on_the_frame_before_it_and_tells_how_far_into_the_gap_it_is():
    exchange_time = np.array([10_000, 10_100, 10_200, 15_200, 15_300, 15_400, 16_700, 16_800])
    nothing = np.array([], dtype=np.int64)
    gap_edges = np.array([10_200, 15_400]), np.array([15_200, 16_700])
    segment = ReplaySegment(exchange_time, nothing, nothing, nothing, *gap_edges, np.array([7, 3]))

    def position_after(elapsed_seconds: float, speed: int) -> tuple[int, GapPassage | None]:
        clock = clock_after(10_000, elapsed_seconds, speed)
        return frame_at(exchange_time, clock), gap_at(segment, clock)

    # 갭 직전 프레임의 시각과 갭 직후 프레임의 시각은 프레임을 받은 때라서 갭 안이 아니다
    assert position_after(0.2, 1) == (2, None)
    assert position_after(0.201, 1) == (2, GapPassage(gap_seconds=5.0, elapsed_seconds=0.001))
    assert position_after(2.7, 1) == (2, GapPassage(gap_seconds=5.0, elapsed_seconds=2.5))
    assert position_after(5.199, 1) == (2, GapPassage(gap_seconds=5.0, elapsed_seconds=4.999))
    assert position_after(5.2, 1) == (3, None)
    # 배속을 올려도 갭의 몇 초째인지는 거래소 시각으로 센다
    assert position_after(0.27, 10) == (2, GapPassage(gap_seconds=5.0, elapsed_seconds=2.5))
    assert position_after(0.6, 10) == (5, GapPassage(gap_seconds=1.3, elapsed_seconds=0.6))
    assert position_after(0.67, 10) == (6, None)