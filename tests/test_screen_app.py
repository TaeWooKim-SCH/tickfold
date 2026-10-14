import dataclasses
import json
from pathlib import Path

import pytest

# 화면용 의존성 그룹(screen)을 안 깐 환경에서는 이 파일을 건너뛴다
pytest.importorskip("plotly")
pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest
from test_replay_segment import PROJECT
from test_screen_figures import read_with_a_gap

from tickfold.replay.segment import save_segment
from tickfold.screen.player import PLAYER_FILES, SCREEN_DIRECTORY, player_data

APP = str(PROJECT / "tickfold" / "screen" / "app.py")


def open_screen(directory: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[AppTest, Path]:
    """픽스처 구간 하나를 내보낸 폴더로 화면을 띄운다."""
    # 갭이 없는 구간이면 숫자 띠의 갭 칸 셋이 전부 0 이라 서로 바뀌어도 모른다
    segment, _ = read_with_a_gap(directory / "raw", 0, -1)
    (directory / "segments").mkdir()
    path = directory / "segments" / "fixture.npz"
    save_segment(segment, path)
    monkeypatch.setenv("TICKFOLD_SEGMENTS", str(directory / "segments"))
    return AppTest.from_file(APP, default_timeout=30).run(), path


def handed_over(app: AppTest) -> tuple[str, str, dict]:
    """화면이 띄운 재생 컴포넌트 하나의 이름, 위젯 ID, 넘긴 값."""
    assert not app.exception
    (player,) = app.get("bidi_component")
    return player.proto.component_name, player.proto.id, json.loads(player.proto.json)


def test_screen_shows_the_summary_strip_and_hands_the_segment_to_the_browser_player(tmp_path, monkeypatch, caplog):
    app, path = open_screen(tmp_path, monkeypatch)

    assert [(metric.label, metric.value) for metric in app.metric] == [
        ("시작 2026-09-27 UTC", "06:14:07.414"),
        ("끝 2026-09-27 UTC", "06:15:07.214"),
        ("프레임", "595"),
        ("갭", "1"),
        ("갭 길이 합", "0.4초"),
        ("놓친 변경", "32"),
        ("가장 긴 전달 지연", "0.1초"),
    ]
    segment, _ = read_with_a_gap(tmp_path / "again", 0, -1)
    name, _, data = handed_over(app)
    assert name == "tickfold_player"
    assert data == json.loads(json.dumps(player_data(segment, f"fixture.npz {path.stat().st_mtime_ns}")))
    # plotly.js 를 인터넷에서 받지 않고 모듈에 싣는다. 화면 파일 셋도 같이 실린다
    (player,) = app.get("bidi_component")
    assert player.proto.js_content.startswith("/**\n* plotly.js v")
    assert all((SCREEN_DIRECTORY / name).read_text() in player.proto.js_content for name in PLAYER_FILES)
    # 재생 단추, 배속, 슬라이더는 브라우저가 그린다. 파이썬 쪽에 남으면 누를 때마다 화면 전체가 다시 돈다
    assert not (app.button or app.button_group or app.slider or app.get("plotly_chart"))
    # 없어질 예정인 호출(components.v1 같은)을 쓰면 Streamlit 이 이 이름으로 경고를 남긴다
    assert not [record for record in caplog.records if record.name == "streamlit.deprecation_util"]


def test_rerun_hands_over_the_same_value_and_a_new_segment_or_export_changes_it(tmp_path, monkeypatch):
    app, path = open_screen(tmp_path, monkeypatch)
    _, player_id, first = handed_over(app)

    # 값이 그대로면 브라우저는 재생을 이어 간다. 위젯 ID 가 바뀌면 컴포넌트가 새로 만들어져 처음부터 멈춘다
    assert handed_over(app.run())[1:] == (player_id, first)

    # 같은 구간을 다른 이름으로 하나 더 내보내고 고른다. 내용이 같아도 새 구간이라 새 값이 간다. 위젯 ID 는 그대로라
    # 같은 컴포넌트에 새 값이 가고, 화면 함수가 앞 화면을 치운 뒤 첫 프레임에 멈춘 채로 다시 그린다
    segment, _ = read_with_a_gap(tmp_path / "again", 0, -1)
    save_segment(segment, path.with_name("second.npz"))
    app.run().selectbox[0].select_index(1).run()
    _, same_id, second = handed_over(app)
    assert same_id == player_id and second["segment"].startswith("second.npz ") and second != first

    # 한 시간 뒤의 구간을 같은 이름으로 다시 내보낸다. 고르기는 그대로여도 새 값이 간다
    names = ("exchange_time", "gap_last_exchange_time", "gap_next_exchange_time")
    later = dataclasses.replace(segment, **{name: getattr(segment, name) + 3_600_000 for name in names})
    save_segment(later, path.with_name("second.npz"))
    _, _, third = handed_over(app.run())
    assert third["exchangeTime"] != second["exchangeTime"]


def test_screen_without_segment_files_shows_only_how_to_export_one(tmp_path, monkeypatch):
    monkeypatch.setenv("TICKFOLD_SEGMENTS", str(tmp_path))

    screen = AppTest.from_file(APP, default_timeout=30).run()

    assert not screen.exception
    assert str(tmp_path) in screen.info[0].value
    assert "scripts/export_replay_segment.py" in screen.code[0].value
    assert not (screen.selectbox or screen.metric or screen.get("bidi_component"))