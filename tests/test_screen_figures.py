import json
from datetime import datetime
from pathlib import Path

import numpy as np
from test_replay_segment import exchange_time_of, fixture_lines, write_day

from tickfold.replay.depth20 import ASK, BID, PRICE
from tickfold.replay.segment import ReplaySegment, read_segment
from tickfold.screen.figures import MID_LINE_POINTS, SegmentSummary, summarize, thin_mid_line, utc_text


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


def test_thinned_mid_line_keeps_the_lowest_and_highest_of_every_bunch_and_breaks_only_at_the_gap():
    # 끝 묶음이 덜 차는 길이다. mid 가 계속 오르므로 묶음의 가장 낮은 점과 높은 점은 그 묶음의 첫 프레임과 끝 프레임이다
    frames, before_the_gap = 8_995, 4_504
    mid = 80_000 + np.arange(frames) * 0.5
    # 몇 번째마다 하나씩 뽑으면 사라지는 한 프레임짜리 급등과 급락
    mid[1_234] += 700
    mid[5_678] -= 700
    # 구간의 첫 프레임과 끝 프레임이 제 묶음에서 가장 낮지도 높지도 않게 한다
    mid[0], mid[-1] = mid[2], mid[-3]
    # 값이 전부 같은 묶음은 가장 낮은 점과 높은 점이 같은 프레임이다. 두 번 남기면 보내는 점만 는다
    mid[18:27] = mid[18]
    exchange_time = 1_791_358_000_014 + np.arange(frames) * 100
    exchange_time[before_the_gap + 1 :] += 24_800
    book = np.zeros((frames, 2, 20, 2))
    book[:, BID, 0, PRICE], book[:, ASK, 0, PRICE] = mid - 0.25, mid + 0.25
    nothing = np.array([], dtype=np.int64)
    gap_edges = exchange_time[[before_the_gap]], exchange_time[[before_the_gap + 1]]
    segment = ReplaySegment(exchange_time, nothing, nothing, book, *gap_edges, np.array([5]))

    line = thin_mid_line(segment)

    times = utc_text(exchange_time).tolist()
    drawn = ~np.isnan(line.mid)
    # 갭 양쪽 프레임과 구간의 첫·끝 프레임은 가장 낮지도 높지도 않다. 그래도 남아야 선이 갭에서만 끊기고 구간을 다 덮는다
    assert drawn.sum() <= MID_LINE_POINTS + 4
    kept = dict(zip(line.time[drawn].tolist(), line.mid[drawn].tolist()))
    assert len(kept) == drawn.sum()
    assert kept.items() <= dict(zip(times, mid.tolist())).items()
    bunch = -(-frames // (MID_LINE_POINTS // 2))
    for first in range(0, frames, bunch):
        kept_here = {kept[time] for time in times[first : first + bunch] if time in kept}
        assert {mid[first : first + bunch].min(), mid[first : first + bunch].max()} <= kept_here, first
    (line_break,) = np.flatnonzero(~drawn).tolist()
    around_the_break = line.time[line_break - 1], line.time[line_break + 1]
    assert around_the_break == (times[before_the_gap], times[before_the_gap + 1])
    assert (line.time[0], line.time[-1]) == (times[0], times[-1])


def test_gap_reaching_outside_the_segment_is_shaded_only_up_to_the_edge_frame(tmp_path):
    # 갭 안에서 시작하는 구간이라 갭 직전 프레임이 구간 밖에 있다
    segment, depth = read_with_a_gap(tmp_path, 101, 400)
    assert segment.gap_last_exchange_time[0] < segment.exchange_time[0]

    line = thin_mid_line(segment)

    first_frame = on_the_time_axis(exchange_time_of(depth[103]))
    assert (line.gap_starts.tolist(), line.gap_ends.tolist()) == ([first_frame], [first_frame])
    assert min(line.time) == first_frame

    # 갭 안에서 끝나는 구간은 갭 직후 프레임이 구간 밖에 있다
    segment, _ = read_with_a_gap(tmp_path, 50, 102)
    line = thin_mid_line(segment)
    last_frame = on_the_time_axis(exchange_time_of(depth[99]))
    assert (line.gap_starts.tolist(), line.gap_ends.tolist()) == ([last_frame], [last_frame])
    assert max(line.time) == last_frame


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