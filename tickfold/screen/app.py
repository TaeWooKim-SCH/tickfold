"""
학술제 시연용 재생 화면. 내보낸 구간 파일 하나를 골라 호가창과 구간 전체의 mid 를 보이고 거래소 시각을 따라 재생한다.

실행 (프로젝트 루트에서. 서버 밖에서는 SSH 터널로 8501 포트를 연다):
    nice -n 19 uv run --group screen streamlit run tickfold/screen/app.py

127.0.0.1 로만 받게 하는 설정은 .streamlit/config.toml 에 있다.
구간 파일만 읽는다. data/raw 와 수집 서비스에는 붙지 않는다.
"""

import os
import time
from pathlib import Path
from typing import NamedTuple

import streamlit as st

from tickfold.replay.segment import ReplaySegment, load_segment
from tickfold.screen.figures import (
    MidLine,
    SegmentSummary,
    book_figure,
    find_quantity_ceiling,
    mid_figure,
    summarize,
    thin_mid_line,
    utc_text,
)
from tickfold.screen.playback import clock_after, frame_at, gap_at

EXPORT_COMMAND = (
    "nice -n 19 uv run python scripts/export_replay_segment.py"
    " 2026-10-07T07:25:00 2026-10-07T07:40:00 segments/gap-0732.npz"
)
SPEEDS = (1, 2, 5, 10, 30)
# 재생 중에 화면 아래쪽을 다시 그리는 주기(초). 한 번 다시 그려 보내는 데 36ms 가 든다(같은 서버 안 웹소켓 왕복).
# 주기가 그보다 짧으면 Streamlit 이 앞 실행의 그림을 보내기 전에 버려서 화면이 멈춘다. 브라우저가 그리는 시간은 못 재서 넉넉히 잡았다
REDRAW_SECONDS = 0.25


class Playback(NamedTuple):
    """재생 시계의 기준점. 재생 중에만 있다. 배속을 바꾸면 그 순간의 재생 시계로 다시 잡아서 자리가 안 튄다."""

    clock: int  # 기준점을 잡은 순간의 재생 시계. 거래소 시각, 밀리초
    moment: float  # 기준점을 잡은 순간의 time.monotonic()
    speed: int


# cache_data 는 부를 때마다 사본을 만든다. 구간 배열은 읽기만 하므로 사본 없이 같은 배열을 돌려주는 쪽을 쓴다
@st.cache_resource
def read_once(path: Path, modified_at: int) -> tuple[ReplaySegment, SegmentSummary, float, MidLine]:
    """다시 그릴 때마다 파일을 읽고 mid 선을 솎지 않게 한다. 같은 이름으로 다시 내보내면 수정 시각이 달라져 새로 읽는다."""
    segment = load_segment(path)
    return segment, summarize(segment), find_quantity_ceiling(segment), thin_mid_line(segment)


def stop_playback() -> None:
    """재생을 멈추고 슬라이더를 새로 만든다. 서버가 자리를 정할 때(멈춤, 끝 프레임, 구간 바뀜) 부른다."""
    st.session_state["playback"] = None
    # 슬라이더는 멈춘 동안 입력이다. 같은 위젯을 계속 쓰면 브라우저가 앞서 들고 있던 값을 늦게 실어 보냈을 때 사용자가 옮긴 것과
    # 구별이 안 된다. 키를 바꾸면 새 위젯이라 브라우저가 들고 있는 옛 값이 없다
    st.session_state["slider_generation"] = st.session_state.get("slider_generation", 0) + 1


def toggle_playback(first_exchange_time: int, last_frame: int) -> None:
    """재생 단추를 누르면 마지막에 그린 자리에서 재생을 걸고, 재생 중이면 그 자리에 멈춘다."""
    position = st.session_state.get("position")
    if position is None:
        # 구간이 바뀌어 자리를 지운 뒤 새 구간을 그리기 전에 누름이 왔다. 앞 구간의 화면에서 누른 것이라 버린다
        return
    if st.session_state.get("playback") is not None:
        stop_playback()
        return
    frame, clock = position
    if frame == last_frame:
        # 끝 프레임에서는 더 갈 데가 없어 눌러도 바로 멈춘다. 첫 프레임부터 다시 재생한다
        clock = first_exchange_time
    st.session_state["playback"] = Playback(clock, time.monotonic(), st.session_state.get("speed", SPEEDS[0]))


st.set_page_config(page_title="Tickfold 호가창 재생", layout="wide")
segment_directory = Path(os.environ.get("TICKFOLD_SEGMENTS", "segments"))
segment_paths = sorted(segment_directory.glob("*.npz"))
if not segment_paths:
    st.info(f"`{segment_directory}` 폴더에 구간 파일(*.npz)이 없다. 프로젝트 루트에서 먼저 내보낸다.")
    st.code(EXPORT_COMMAND, language="bash")
    st.stop()

segment_path = st.selectbox("구간", segment_paths, format_func=lambda path: path.stem)
modified_at = segment_path.stat().st_mtime_ns
segment, summary, quantity_ceiling, mid_line = read_once(segment_path, modified_at)

first_date, first_clock = str(utc_text(summary.first_exchange_time)).split("T")
last_date, last_clock = str(utc_text(summary.last_exchange_time)).split("T")
# Streamlit 은 칸보다 긴 값을 잘라 보인다. 시각이 다른 숫자보다 길어서 칸을 넓게 준다
strip = st.columns([3, 3, 2, 1, 2, 2, 2])
strip[0].metric(f"시작 {first_date} UTC", first_clock)
strip[1].metric(f"끝 {last_date} UTC", last_clock)
strip[2].metric("프레임", f"{summary.frame_count:,}")
strip[3].metric("갭", f"{summary.gap_count}")
strip[4].metric("갭 길이 합", f"{summary.gap_seconds:.1f}초")
strip[5].metric("놓친 변경", f"{summary.missed_updates:,}")
strip[6].metric("가장 긴 전달 지연", f"{summary.longest_delivery_delay_seconds:.1f}초")

# 재생 기준점과 마지막에 그린 자리는 이 구간의 거래소 시각이라 구간이 바뀌면 지운다.
# 구간 고르기가 바뀔 때만 지우면 고르기를 안 거치고 구간이 바뀔 때 남는다. 폴더에 파일이 늘면 고르기가 첫 파일로 돌아가고,
# 같은 이름으로 다시 내보내면 이름은 그대로인데 시각이 바뀐다
shown_segment = (segment_path.name, modified_at)
if st.session_state.get("shown_segment") != shown_segment:
    st.session_state["shown_segment"] = shown_segment
    st.session_state.pop("position", None)
    # 자리를 지웠으니 새 슬라이더는 첫 프레임에서 시작한다. 처음 뜰 때도 이 길로 첫 슬라이더가 생긴다
    stop_playback()
is_playing = st.session_state.get("playback") is not None
play_column, speed_column = st.columns([1, 9], vertical_alignment="bottom")
play_arguments = (summary.first_exchange_time, summary.frame_count - 1)
play_column.button("멈춤" if is_playing else "재생", key="play", on_click=toggle_playback, args=play_arguments)
speed = speed_column.segmented_control(
    "배속", SPEEDS, default=SPEEDS[0], format_func=lambda choice: f"{choice}배", key="speed", required=True
)


def draw_position() -> None:
    """슬라이더, 그림 둘, 갭 안내를 그린다. 슬라이더는 멈춘 동안에만 입력이고 재생 중에는 재생 위치를 보이기만 한다."""
    frame_key = f"frame:{st.session_state['slider_generation']}"
    shown_frame, shown_clock = st.session_state.get("position", (None, None))
    playback: Playback | None = st.session_state.get("playback")
    if playback is not None:
        moment = time.monotonic()
        clock = clock_after(playback.clock, moment - playback.moment, playback.speed)
        if playback.speed != speed:
            # 배속을 바꿨다. 지금까지는 앞 배속으로 흐른 것이므로 지금의 재생 시계에서 새 배속으로 이어 간다
            st.session_state["playback"] = Playback(clock, moment, speed)
        frame = frame_at(segment.exchange_time, clock)
        if frame == summary.frame_count - 1:
            # 재생 시계는 끝 프레임을 지나쳐 있다. 멈춘 자리는 끝 프레임의 시각으로 적는다
            st.session_state["position"] = (frame, summary.last_exchange_time)
            stop_playback()
            st.rerun()
        # 재생 중에는 슬라이더 값을 읽지 않고 주기마다 적기만 한다. 브라우저가 돌려보내는 값은 몇 주기 늦을 수 있어서 읽으면
        # 재생이 그 자리에 묶인다. 적을 때는 기본값을 같이 주지 않는다. 둘 다 주면 Streamlit 이 서버 로그에 경고를 남긴다
        st.session_state[frame_key] = frame
        known_frame = None
    else:
        # 멈춘 동안에는 서버가 아는 값을 슬라이더의 기본값으로 실어 보낸다. 새 슬라이더라 아는 값이 없으면 마지막에 그린 프레임이다.
        # 세션에 적어 주면 그 값은 적은 실행의 메시지에 한 번만 실린다. 그 메시지가 브라우저에 가기 전에 다음 실행이 시작되면
        # Streamlit 이 버리고(멈춤을 누르고 곧바로 배속을 바꿀 때), 값을 못 받은 브라우저는 새 슬라이더를 기본값에 둔다
        known_frame = st.session_state.get(frame_key, shown_frame or 0)
    book_column, mid_column = st.columns(2)
    frame = mid_column.slider(
        "프레임", 0, summary.frame_count - 1, known_frame, key=frame_key, disabled=playback is not None
    )
    if playback is None:
        # 멈춘 동안에는 슬라이더가 자리를 정한다. 슬라이더가 마지막에 그린 프레임 그대로면 재생 시계도 그대로 둔다.
        # 프레임의 시각으로 되돌리면 갭 안에서 멈췄을 때 세로선이 갭 앞으로 튀고, 다시 재생하면 갭을 처음부터 다시 지난다
        clock = shown_clock if shown_frame == frame else int(segment.exchange_time[frame])
    st.session_state["position"] = (frame, clock)
    book_column.plotly_chart(book_figure(segment, frame, quantity_ceiling), key="book")
    mid_column.plotly_chart(mid_figure(mid_line, clock), key="mid")
    gap = gap_at(segment, clock)
    if gap is not None:
        # 이 안내가 없으면 갭을 지나는 동안 호가창이 고장 나 멈춘 것처럼 보인다
        mid_column.warning(
            f"받은 프레임이 없는 구간이다. 갭 {gap.gap_seconds:.1f}초 가운데 {gap.elapsed_seconds:.1f}초째."
            " 호가창은 갭 직전 프레임에 멈춰 있다."
        )


if is_playing:
    # 재생 중에는 이 아래만 주기마다 다시 돈다. 구간 고르기와 숫자 띠까지 다시 돌면 주기마다 화면 전체를 새로 보낸다.
    # 주기 실행은 화면 전체가 다시 돌 때만 켜지고 꺼진다. 그래서 끝 프레임에 닿아 멈출 때는 st.rerun 으로 전체를 다시 돌린다
    st.fragment(run_every=REDRAW_SECONDS)(draw_position)()
else:
    # 멈춘 동안에는 fragment 로 감싸지 않는다. 브라우저는 멈춘 화면을 받을 때까지 주기 신호를 보낸다. 감싸 두면 그 신호가 멈춘 화면을
    # 다시 돌리고, 그 실행이 멈춤 누름의 실행에 바로 붙으면 Streamlit 이 앞 실행이 보내던 메시지를 버린다(아래쪽, 드물게는 단추까지).
    # 감싸지 않으면 그 신호는 돌릴 fragment 가 없어 버려진다. fragment 는 속을 container 한 겹에 담아 그리므로 멈춘 화면도 한 겹에
    # 담는다. 깊이가 다르면 재생과 멈춤이 바뀔 때마다 아래쪽 요소의 자리가 달라진다
    with st.container():
        draw_position()