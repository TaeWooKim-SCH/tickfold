import base64
import json
from datetime import datetime, timezone

import numpy as np
import pytest

# 화면용 의존성 그룹(screen)을 안 깐 환경에서는 이 파일을 건너뛴다
pytest.importorskip("plotly")
pytest.importorskip("streamlit")

from test_replay_segment import exchange_time_of
from test_screen_figures import on_the_time_axis, raw_depth20, read_with_a_gap

from tickfold.screen.figures import find_quantity_ceiling
from tickfold.screen.player import player_data


def as_the_browser_gets_it(data: dict) -> dict:
    # JSON 에 nan 이 끼면 파이썬은 그대로 적지만 브라우저의 JSON.parse 가 못 읽는다
    return json.loads(json.dumps(data, allow_nan=False))


def decoded(text: str) -> list[float]:
    return np.frombuffer(base64.b64decode(text), dtype="<f8").tolist()


def test_player_data_carries_every_frame_with_the_raw_prices_quantities_and_exchange_times(tmp_path):
    segment, depth = read_with_a_gap(tmp_path, 50, 400)

    data = as_the_browser_gets_it(player_data(segment, "fixture.npz 1"))

    raw = raw_depth20(segment)
    for side in ("ask", "bid"):
        # 쪽이 바뀌거나 레벨이 밀리면 여기서 걸린다. 값은 원본 문자열을 읽은 float64 와 비트까지 같아야 한다
        levels = [record[f"{side}s"] for record in raw]
        assert decoded(data[f"{side}Price"]) == [float(price) for frame in levels for price, _ in frame]
        assert decoded(data[f"{side}Quantity"]) == [float(quantity) for frame in levels for _, quantity in frame]
    assert decoded(data["exchangeTime"]) == segment.exchange_time.tolist()
    assert data["quantityCeiling"] == find_quantity_ceiling(segment)
    before_the_gap, after_the_gap = (exchange_time_of(depth[index]) for index in (99, 103))
    assert data["gaps"] == [
        {
            "lastExchangeTime": int(before_the_gap.timestamp() * 1000),
            "nextExchangeTime": int(after_the_gap.timestamp() * 1000),
            "shadeStart": on_the_time_axis(before_the_gap),
            "shadeEnd": on_the_time_axis(after_the_gap),
        }
    ]
    assert data["segment"] == "fixture.npz 1"


def test_mid_line_is_cut_by_a_null_at_the_gap_on_the_utc_time_axis(tmp_path):
    segment, depth = read_with_a_gap(tmp_path, 50, 400)

    line = as_the_browser_gets_it(player_data(segment, "fixture.npz 1"))["midLine"]

    # 프레임이 MID_LINE_POINTS 보다 적은 구간이라 솎지 않고 전부 보낸다. 빈 점 하나만 null 이다
    (line_break,) = [index for index, mid in enumerate(line["mid"]) if mid is None]
    drawn = [index for index in range(len(line["mid"])) if index != line_break]
    best_prices = [(float(record["bids"][0][0]), float(record["asks"][0][0])) for record in raw_depth20(segment)]
    assert [line["mid"][index] for index in drawn] == [(bid + ask) / 2 for bid, ask in best_prices]
    # 시간 축은 거래소 시각이고 UTC 다. 이 서버의 시간대(KST)로 바뀌어 나오면 여기서 걸린다
    times = [datetime.fromtimestamp(time / 1000, tz=timezone.utc) for time in segment.exchange_time.tolist()]
    assert [line["time"][index] for index in drawn] == [on_the_time_axis(time) for time in times]
    around_the_break = [line["time"][line_break - 1], line["time"][line_break + 1]]
    assert around_the_break == [on_the_time_axis(exchange_time_of(depth[index])) for index in (99, 103)]