import json

import pytest

# 화면용 의존성 그룹(screen)을 안 깐 환경에서는 이 파일을 건너뛴다
pytest.importorskip("plotly")
pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest
from test_replay_segment import PROJECT
from test_screen_figures import read_with_a_gap

from tickfold.replay.segment import save_segment
from tickfold.screen.figures import book_figure, find_quantity_ceiling, mid_figure

APP = str(PROJECT / "tickfold" / "screen" / "app.py")


def test_screen_shows_the_segment_and_redraws_both_figures_when_the_slider_moves(tmp_path, monkeypatch, caplog):
    # 갭이 없는 구간이면 숫자 띠의 갭 칸 셋이 전부 0 이라 서로 바뀌어도 모른다
    segment, _ = read_with_a_gap(tmp_path / "raw", 0, -1)
    (tmp_path / "segments").mkdir()
    save_segment(segment, tmp_path / "segments" / "fixture.npz")
    monkeypatch.setenv("TICKFOLD_SEGMENTS", str(tmp_path / "segments"))

    def expected_figures(frame: int) -> list[dict]:
        figures = book_figure(segment, frame, find_quantity_ceiling(segment)), mid_figure(segment, frame)
        return [json.loads(figure.to_json()) for figure in figures]

    screen = AppTest.from_file(APP, default_timeout=30).run()

    assert not screen.exception
    assert [(metric.label, metric.value) for metric in screen.metric] == [
        ("시작 2026-09-27 UTC", "06:14:07.414"),
        ("끝 2026-09-27 UTC", "06:15:07.214"),
        ("프레임", "595"),
        ("갭", "1"),
        ("갭 길이 합", "0.4초"),
        ("놓친 변경", "32"),
        ("가장 긴 전달 지연", "0.1초"),
    ]
    assert (screen.slider[0].min, screen.slider[0].max) == (0, 594)
    assert [json.loads(chart.proto.spec) for chart in screen.get("plotly_chart")] == expected_figures(0)

    screen.slider[0].set_value(321).run()

    assert not screen.exception
    assert [json.loads(chart.proto.spec) for chart in screen.get("plotly_chart")] == expected_figures(321)
    # 없어질 예정인 호출을 쓰면 Streamlit 이 이 이름으로 경고를 남긴다
    assert not [record for record in caplog.records if record.name == "streamlit.deprecation_util"]


def test_screen_without_segment_files_shows_only_how_to_export_one(tmp_path, monkeypatch):
    monkeypatch.setenv("TICKFOLD_SEGMENTS", str(tmp_path))

    screen = AppTest.from_file(APP, default_timeout=30).run()

    assert not screen.exception
    assert str(tmp_path) in screen.info[0].value
    assert "scripts/export_replay_segment.py" in screen.code[0].value
    assert not (screen.selectbox or screen.metric or screen.slider or screen.get("plotly_chart"))