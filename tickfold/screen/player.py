"""브라우저가 재생하는 호가창 컴포넌트. 구간을 고를 때 배열과 고정값을 한 번 넘기고, 재생과 조작은 브라우저 안에서만 돈다."""

import base64
from pathlib import Path

import numpy as np
import streamlit as st
from plotly.offline import get_plotlyjs

from tickfold.replay.depth20 import ASK, BID, PRICE, QUANTITY
from tickfold.replay.segment import ReplaySegment
from tickfold.screen.figures import find_quantity_ceiling, thin_mid_line

SCREEN_DIRECTORY = Path(__file__).parent
PLAYER_FILES = ("book.js", "playback.js", "player.js")


def player_script() -> str:
    """컴포넌트의 자바스크립트 모듈 하나를 만든다. plotly.js 뒤에 화면 파일 셋을 이어 붙인다."""
    # 발표장에 인터넷이 없을 수 있다. CDN 대신 plotly 파이썬 패키지에 든 plotly.min.js 를 모듈에 통째로 싣는다.
    # 컴포넌트는 js 문자열 하나만 받고 import 는 못 해서 화면 파일 셋도 이어 붙인다
    return "\n".join([get_plotlyjs(), *((SCREEN_DIRECTORY / name).read_text() for name in PLAYER_FILES)])


def encoded(values: np.ndarray) -> str:
    """float64 배열을 바이트 그대로 base64 글자로 바꾼다. 브라우저는 이것을 Float64Array 로 푼다."""
    # 숫자를 JSON 으로 적으면 15분 구간 배열이 7.5MB 에서 더 커지고, 글자를 다시 수로 읽을 때 값이 같은지 따로 봐야 한다
    return base64.b64encode(np.ascontiguousarray(values, dtype="<f8").tobytes()).decode("ascii")


def player_data(segment: ReplaySegment, name: str) -> dict:
    """구간 하나를 브라우저에 한 번 넘길 값으로 바꾼다. 이 값이 그대로면 브라우저는 재생을 이어 가고 바뀌면 처음부터 멈춘 채로 그린다."""
    line = thin_mid_line(segment)
    gap_edges = zip(segment.gap_last_exchange_time.tolist(), segment.gap_next_exchange_time.tolist())
    return {
        # 같은 이름으로 다시 내보낸 구간은 내용이 같아도 새 구간으로 친다
        "segment": name,
        "exchangeTime": encoded(segment.exchange_time),
        "askPrice": encoded(segment.book[:, ASK, :, PRICE]),
        "askQuantity": encoded(segment.book[:, ASK, :, QUANTITY]),
        "bidPrice": encoded(segment.book[:, BID, :, PRICE]),
        "bidQuantity": encoded(segment.book[:, BID, :, QUANTITY]),
        "quantityCeiling": find_quantity_ceiling(segment),
        "gaps": [
            {"lastExchangeTime": last, "nextExchangeTime": following, "shadeStart": start, "shadeEnd": end}
            for (last, following), start, end in zip(gap_edges, line.gap_starts.tolist(), line.gap_ends.tolist())
        ],
        # JSON 에는 nan 이 없다. 선을 끊는 빈 점은 null 로 보낸다
        "midLine": {"time": line.time.tolist(), "mid": np.where(np.isnan(line.mid), None, line.mid).tolist()},
    }


def show_player(data: dict) -> None:
    """컴포넌트를 띄운다. 키를 고정해서 구간이 바뀌어도 같은 자리의 컴포넌트에 새 값만 간다."""
    # 그림자 DOM 으로 감싸면 plotly 가 문서 머리에 넣는 스타일(도구 막대, 마우스 안내)이 그림에 안 닿는다
    player = st.components.v2.component("tickfold_player", js=player_script(), isolate_styles=False)
    player(key="player", data=data)