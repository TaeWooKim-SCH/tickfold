"""
학술제 시연용 재생 화면. 내보낸 구간 파일 하나를 골라 호가창과 구간 전체의 mid 를 보이고 거래소 시각을 따라 재생한다.

실행 (프로젝트 루트에서. 서버 밖에서는 SSH 터널로 8501 포트를 연다):
    nice -n 19 uv run --group screen streamlit run tickfold/screen/app.py

127.0.0.1 로만 받게 하는 설정은 .streamlit/config.toml 에 있다.
구간 파일만 읽는다. data/raw 와 수집 서비스에는 붙지 않는다.
파이썬은 구간을 고르고 숫자 띠를 보이고 구간을 브라우저에 한 번 넘긴다. 재생과 조작은 브라우저가 한다(player.js).
"""

import os
from pathlib import Path

import streamlit as st

from tickfold.replay.segment import load_segment
from tickfold.screen.figures import SegmentSummary, summarize, utc_text
from tickfold.screen.player import player_data, show_player

EXPORT_COMMAND = (
    "nice -n 19 uv run python scripts/export_replay_segment.py"
    " 2026-10-07T07:25:00 2026-10-07T07:40:00 segments/gap-0732.npz"
)


# cache_data 는 부를 때마다 사본을 만든다. 넘길 값은 읽기만 하므로 사본 없이 같은 값을 돌려주는 쪽을 쓴다
@st.cache_resource
def read_once(path: Path, modified_at: int) -> tuple[SegmentSummary, dict]:
    """구간을 고를 때만 파일을 읽고 브라우저에 넘길 값을 만든다. 같은 이름으로 다시 내보내면 수정 시각이 달라져 새로 읽는다."""
    segment = load_segment(path)
    return summarize(segment), player_data(segment, f"{path.name} {modified_at}")


st.set_page_config(page_title="Tickfold 호가창 재생", layout="wide")
segment_directory = Path(os.environ.get("TICKFOLD_SEGMENTS", "segments"))
segment_paths = sorted(segment_directory.glob("*.npz"))
if not segment_paths:
    st.info(f"`{segment_directory}` 폴더에 구간 파일(*.npz)이 없다. 프로젝트 루트에서 먼저 내보낸다.")
    st.code(EXPORT_COMMAND, language="bash")
    st.stop()

segment_path = st.selectbox("구간", segment_paths, format_func=lambda path: path.stem)
summary, data = read_once(segment_path, segment_path.stat().st_mtime_ns)

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

show_player(data)