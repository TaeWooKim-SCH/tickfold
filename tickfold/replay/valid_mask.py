"""상위 N 배열의 행마다 믿을 수 있는지를 컬럼으로 두고, 거래소 시각 창 하나의 시간을 어디에 썼는지 나눈다. 대조와 커버리지가 읽는다."""

from dataclasses import astuple, dataclass

import numpy as np

from tickfold.replay.book_replay import BookReplay, BookStatus
from tickfold.replay.depth20 import ASK, BID, PRICE
from tickfold.replay.top_levels import TopLevels


@dataclass(frozen=True)
class ValidColumns:
    """TopLevels 와 같은 행. 커버리지는 이 네 컬럼만 읽으므로 날짜별로 저장한 뒤에도 호가창 배열 없이 다시 낸다."""

    exchange_time: np.ndarray  # (프레임,) int64. TopLevels.exchange_time 그대로
    # 갭 바로 뒤 프레임 앞에 스냅샷이 끼면 그 행은 SYNCED 라서 status 로는 갭 자리를 못 찾는다.
    # 갭 앞 행이 아니라 뒤 행에 두는 것은 날짜 하나의 재생이 그날 행 앞의 갭은 늘 보지만 끝 행 뒤의 갭은 못 볼 수 있어서다
    resumed_after_gap: np.ndarray  # (프레임,) bool. 이 프레임 바로 앞에서 순번이 끊겼다
    synced: np.ndarray  # (프레임,) bool. status 가 SYNCED
    inside_snapshot_range: np.ndarray  # (프레임,) bool. 매수·매도 N번째 레벨이 둘 다 마지막 스냅샷이 아는 범위 안


@dataclass(frozen=True)
class Coverage:
    """거래소 시각 창 하나를 넷으로 나눈 시간, 밀리초. 판정은 재동기화까지 뺀 비율로 하지만 다른 두 비율도 표에 쓰여서 나눠 담는다."""

    window_milliseconds: int
    gap_milliseconds: int  # 순번이 끊겨 프레임이 없는 시간
    unsynced_milliseconds: int  # 갭 밖에서 동기화 안 된 시간. 갭 뒤 재동기화 대기와 수집 첫 스냅샷 전
    outside_range_milliseconds: int  # 동기화됐지만 N번째 레벨이 마지막 스냅샷 범위 밖인 시간

    def __add__(self, other: "Coverage") -> "Coverage":
        """두 창을 하나로 친다. 맞닿은 창끼리 더하면 둘을 한 창으로 잰 값과 같다."""
        return Coverage(*(mine + theirs for mine, theirs in zip(astuple(self), astuple(other))))

    @property
    def gap_free_ratio(self) -> float:
        """갭만 뺀 비율."""
        return (self.window_milliseconds - self.gap_milliseconds) / self.window_milliseconds

    @property
    def synced_ratio(self) -> float:
        """갭과 갭 뒤 재동기화가 끝날 때까지를 뺀 비율. 72시간 판정 기준의 믿을 수 있는 시간 비율이다."""
        excluded = self.gap_milliseconds + self.unsynced_milliseconds
        return (self.window_milliseconds - excluded) / self.window_milliseconds

    @property
    def valid_ratio(self) -> float:
        """범위 밖 시간까지 뺀 비율. valid 두 컬럼이 다 참인 시간이다."""
        excluded = self.gap_milliseconds + self.unsynced_milliseconds + self.outside_range_milliseconds
        return (self.window_milliseconds - excluded) / self.window_milliseconds


def mark_valid(top: TopLevels, replay: BookReplay) -> ValidColumns:
    """top 의 행마다 갭 자리와 valid 두 컬럼을 만든다. replay 는 top 을 뽑은 재생이고 갭 목록을 거기서 읽는다."""
    frames = len(top.exchange_time)
    if frames != replay.end_row - replay.first_row:
        raise ValueError(f"top 은 {frames}행인데 replay 범위는 {replay.end_row - replay.first_row}행이다")
    # 갭 목록의 행 번호는 데우기 시작 행보다도 앞에서 시작하는 replay.sequence 의 것이라 범위 첫 행만큼 당긴다
    resumed_rows = replay.gaps.resumed_row - replay.first_row
    resumed_after_gap = np.zeros(frames, dtype=bool)
    resumed_after_gap[resumed_rows[(resumed_rows >= 0) & (resumed_rows < frames)]] = True
    # NaN 과 견주면 거짓이라 동기화 안 된 행, 스냅샷이 범위를 모르는 행, N번째 칸이 빈 행은 범위 밖이 된다.
    # N번째 칸이 비었으면 우리가 모르는 범위 밖 레벨이 거래소의 상위 N 에 들 수 있어서 범위 안이라고 못 한다
    bid_inside = top.book[:, BID, -1, PRICE] >= top.lowest_known_bid
    ask_inside = top.book[:, ASK, -1, PRICE] <= top.highest_known_ask
    synced = top.status == BookStatus.SYNCED
    return ValidColumns(top.exchange_time, resumed_after_gap, synced, bid_inside & ask_inside)


def measure_coverage(valid: ValidColumns, start: int, end: int) -> Coverage:
    """거래소 시각 start 이상 end 미만(밀리초)을 갭, 동기화 안 됨, 범위 밖, 나머지로 나눈다. 행 수가 아니라 시각 간격으로 잰다.

    시점마다 거래소 시각이 그 시점 이하인 마지막 프레임의 상태로 센다. 재생 화면의 시계와 같은 규칙이다. 프레임 하나가 다음
    프레임까지를 맡고, 그 사이에서 순번이 끊겼으면 그 시간은 갭이다. 창 시작에 걸친 프레임과 창 끝 이후의 프레임이 valid 에
    있어야 한다. 날짜 하나를 잴 때는 앞 날짜 끝 행과 뒤 날짜 첫 행을 이어 붙여 넘긴다.
    """
    if start >= end:
        raise ValueError(f"창 시작 {start} 이 끝 {end} 보다 앞이어야 한다")
    time = valid.exchange_time
    # 창 앞 조각은 창 앞 프레임 몫이고, 창 끝 조각은 다음 프레임이 갭 뒤에 왔는지 봐야 나눈다. 없으면 모르는 시간을 아는 것처럼 센다
    owner = int(np.searchsorted(time, start, side="right")) - 1
    after = int(np.searchsorted(time, end, side="left"))
    if owner < 0:
        raise ValueError(f"창 시작 {start} 앞 프레임이 없다. 앞 날짜의 끝 행을 이어 붙이거나 창을 첫 프레임부터 잡는다")
    if after == len(time):
        raise ValueError(f"창 끝 {end} 뒤 프레임이 없다. 뒤 날짜의 첫 행을 이어 붙이거나 창을 끝 프레임까지로 잡는다")
    owned = np.diff(np.concatenate(([start], time[owner + 1 : after], [end])))
    gap = valid.resumed_after_gap[owner + 1 : after + 1]
    synced = valid.synced[owner:after]
    unsynced = ~gap & ~synced
    outside = ~gap & synced & ~valid.inside_snapshot_range[owner:after]
    return Coverage(end - start, int(owned[gap].sum()), int(owned[unsynced].sum()), int(owned[outside].sum()))