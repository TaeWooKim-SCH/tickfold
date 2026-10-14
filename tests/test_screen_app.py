import dataclasses
import json
import time
from pathlib import Path

import pytest

# 화면용 의존성 그룹(screen)을 안 깐 환경에서는 이 파일을 건너뛴다
pytest.importorskip("plotly")
pytest.importorskip("streamlit")

import streamlit
from streamlit.testing.v1 import AppTest
from test_replay_segment import PROJECT
from test_screen_figures import read_with_a_gap

from tickfold.replay.segment import save_segment
from tickfold.screen.figures import book_figure, find_quantity_ceiling, mid_figure, thin_mid_line

APP = str(PROJECT / "tickfold" / "screen" / "app.py")
GAP_NOTICE = "받은 프레임이 없는 구간이다. 갭 0.4초 가운데 {:.1f}초째. 호가창은 갭 직전 프레임에 멈춰 있다."


class Screen:
    """픽스처 구간 하나를 띄운 화면. 화면이 재는 시계를 손으로 돌린다. AppTest 는 주기 실행을 스스로 돌리지 않는다."""

    def __init__(self, directory: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # 갭이 없는 구간이면 숫자 띠의 갭 칸 셋이 전부 0 이라 서로 바뀌어도 모른다
        self.segment, _ = read_with_a_gap(directory / "raw", 0, -1)
        (directory / "segments").mkdir()
        save_segment(self.segment, directory / "segments" / "fixture.npz")
        monkeypatch.setenv("TICKFOLD_SEGMENTS", str(directory / "segments"))
        self.times = self.segment.exchange_time.tolist()
        self.started_at = self.now = 1000.0
        monkeypatch.setattr(time, "monotonic", lambda: self.now)
        # 주기 실행이 걸렸는지는 화면 요소로 안 보인다. 화면이 아래쪽을 fragment 로 감쌀 때 넘긴 주기를 실행마다 적어 둔다.
        # 화면은 실행마다 set_page_config 를 맨 먼저 부르므로 거기서 앞 실행의 것을 지운다
        self.redraw_intervals: list[float | None] = []
        fragment, set_page_config = streamlit.fragment, streamlit.set_page_config

        def noting_fragment(run_every: float | None):
            self.redraw_intervals.append(run_every)
            return fragment(run_every=run_every)

        def forgetting_set_page_config(**settings) -> None:
            self.redraw_intervals.clear()
            set_page_config(**settings)

        monkeypatch.setattr(streamlit, "fragment", noting_fragment)
        monkeypatch.setattr(streamlit, "set_page_config", forgetting_set_page_config)
        self.app = AppTest.from_file(APP, default_timeout=30).run()

    def run_at(self, seconds: float) -> None:
        """시계를 처음에서 seconds 만큼 지난 때로 돌리고 주기 실행 한 번처럼 다시 돌린다."""
        self.now = self.started_at + seconds
        self.app.run()

    def browser_sends_slider_value(self, frame: int) -> None:
        """다음 실행에 브라우저가 실어 보낼 슬라이더 값을 정한다. 브라우저는 자기가 아는 값을 실행을 청할 때마다 같이 보낸다."""
        # 잠긴 슬라이더는 사람이 못 옮기므로 AppTest 가 set_value 를 막는다. 브라우저가 보내는 값은 잠겨 있어도 온다.
        # AppTest 는 위젯마다 세션에 있는 값을 실어 보내므로 지금 화면에 있는 슬라이더의 키에 적어 둔다
        self.app.session_state[self.app.slider[0].key] = frame

    def expected(self, play_button: str, clock: int, *notices: str) -> tuple:
        """재생 시계가 clock 일 때 화면에 보여야 하는 것. 프레임은 거래소 시각이 clock 이하인 마지막 프레임이다."""
        frame = max(index for index, frame_time in enumerate(self.times) if frame_time <= clock)
        ceiling, line = find_quantity_ceiling(self.segment), thin_mid_line(self.segment)
        figures = book_figure(self.segment, frame, ceiling), mid_figure(line, clock)
        return play_button, frame, [json.loads(figure.to_json()) for figure in figures], list(notices)

    def shown(self) -> tuple:
        assert not self.app.exception
        playing = self.app.button[0].label == "멈춤"
        # 재생 중에만 아래쪽을 fragment 로 감싸 주기 실행을 건다. 안 걸리면 재생해도 화면이 안 움직인다. 멈춘 동안에도 감싸면
        # 멈춘 뒤에 늦게 닿은 주기 신호가 멈춘 화면을 다시 돌린다
        assert [bool(interval) for interval in self.redraw_intervals] == ([True] if playing else [])
        # 재생 중에는 슬라이더가 표시만 한다. 잠겨 있지 않으면 사용자가 옮긴 것이 다음 주기에 말없이 덮인다
        assert self.app.slider[0].disabled == playing
        if not playing:
            # 멈춘 동안에는 슬라이더의 기본값이 서버가 아는 값이다. 브라우저는 슬라이더를 새로 만들 때 기본값에서 시작한다.
            # 값을 따로 한 번만 보내면 Streamlit 이 그 메시지를 버렸을 때(멈춤을 누르고 곧바로 배속을 바꾸면) 0 에 선다
            assert list(self.app.slider[0].proto.default) == [self.app.slider[0].value]
        figures = [json.loads(chart.proto.spec) for chart in self.app.get("plotly_chart")]
        notices = [notice.value for notice in self.app.warning]
        return self.app.button[0].label, self.app.slider[0].value, figures, notices


def test_screen_shows_the_segment_and_redraws_both_figures_when_the_slider_moves(tmp_path, monkeypatch, caplog):
    screen = Screen(tmp_path, monkeypatch)

    assert not screen.app.exception
    assert [(metric.label, metric.value) for metric in screen.app.metric] == [
        ("시작 2026-09-27 UTC", "06:14:07.414"),
        ("끝 2026-09-27 UTC", "06:15:07.214"),
        ("프레임", "595"),
        ("갭", "1"),
        ("갭 길이 합", "0.4초"),
        ("놓친 변경", "32"),
        ("가장 긴 전달 지연", "0.1초"),
    ]
    assert (screen.app.slider[0].min, screen.app.slider[0].max) == (0, 594)
    assert screen.app.button_group[0].options == ["1배", "2배", "5배", "10배", "30배"]
    assert screen.shown() == screen.expected("재생", screen.times[0])

    screen.app.slider[0].set_value(321).run()

    assert screen.shown() == screen.expected("재생", screen.times[321])
    # 없어질 예정인 호출을 쓰면 Streamlit 이 이 이름으로 경고를 남긴다
    assert not [record for record in caplog.records if record.name == "streamlit.deprecation_util"]


def test_playing_follows_exchange_time_waits_inside_the_gap_and_stays_put_when_paused(tmp_path, monkeypatch, caplog):
    screen = Screen(tmp_path, monkeypatch)
    first, before_the_gap, after_the_gap = (screen.times[frame] for frame in (0, 99, 100))
    assert after_the_gap - before_the_gap == 399

    screen.app.button[0].click().run()
    assert screen.shown() == screen.expected("멈춤", first)

    # 프레임 사이의 시각이다. 세로선은 재생 시계에 긋고 호가창은 그 앞 프레임을 보인다
    screen.run_at(2.03)
    assert screen.shown() == screen.expected("멈춤", first + 2_030)
    assert first + 2_030 not in screen.times

    # 갭 안에서는 호가창이 갭 직전 프레임에 멈추고 세로선만 간다. 안내가 없으면 고장으로 보인다
    screen.run_at((before_the_gap - first + 200) / 1000)
    assert screen.shown() == screen.expected("멈춤", before_the_gap + 200, GAP_NOTICE.format(0.2))
    assert screen.app.slider[0].value == 99

    # 멈추면 화면이 그대로 선다. 세로선이 갭 앞으로 돌아가지 않고, 다시 재생하면 선 자리에서 이어 간다.
    # 누르는 순간 브라우저의 슬라이더가 마지막에 그린 프레임보다 옛 값이어도 사용자가 옮긴 것으로 읽지 않는다.
    # 누름은 마지막 주기 실행보다 늦게 닿는다. 그 순간의 시계로 다시 계산하지 않고 마지막에 그린 자리에 선다
    screen.browser_sends_slider_value(97)
    screen.now += 0.04
    screen.app.button[0].click().run()
    assert screen.shown() == screen.expected("재생", before_the_gap + 200, GAP_NOTICE.format(0.2))
    screen.run_at(30)
    assert screen.shown() == screen.expected("재생", before_the_gap + 200, GAP_NOTICE.format(0.2))
    screen.app.button[0].click().run()
    screen.run_at(30.1)
    assert screen.shown() == screen.expected("멈춤", before_the_gap + 300, GAP_NOTICE.format(0.3))
    screen.run_at(30.199)
    assert screen.shown() == screen.expected("멈춤", after_the_gap)
    assert screen.app.slider[0].value == 100

    # 멈춘 채 슬라이더로 갭 직전 프레임을 고르면 안내가 뜨지 않는다. 재생 시계가 갭 안이 아니라 그 프레임의 시각이다
    screen.app.button[0].click().run()
    screen.app.slider[0].set_value(98).run()
    screen.app.slider[0].set_value(99).run()
    assert screen.shown() == screen.expected("재생", before_the_gap)
    # 자리를 옮기려면 멈추고 옮긴다. 다시 재생하면 멈췄던 자리가 아니라 옮긴 자리에서 이어 간다
    screen.app.button[0].click().run()
    screen.run_at(30.299)
    assert screen.shown() == screen.expected("멈춤", before_the_gap + 100, GAP_NOTICE.format(0.1))
    assert not [record for record in caplog.records if record.name == "streamlit.deprecation_util"]


def test_playing_keeps_its_place_at_a_new_speed_ignores_the_slider_and_restarts_after_the_end(tmp_path, monkeypatch):
    screen = Screen(tmp_path, monkeypatch)
    first = screen.times[0]
    screen.app.button[0].click().run()
    screen.run_at(3)
    three_seconds_in = screen.app.slider[0].value

    # 배속을 바꾼 순간에는 자리가 그대로다. 그 뒤부터 새 배속으로 간다
    screen.app.button_group[0].set_value(10).run()
    assert screen.shown() == screen.expected("멈춤", first + 3_000)
    screen.run_at(3.5)
    assert screen.shown() == screen.expected("멈춤", first + 8_000)

    # 재생 중에는 슬라이더 값을 읽지 않는다. 브라우저가 여러 주기 전에 받은 값을 돌려보내도 재생 시계대로 간다
    screen.browser_sends_slider_value(three_seconds_in)
    screen.run_at(3.6)
    assert screen.shown() == screen.expected("멈춤", first + 9_000)
    screen.browser_sends_slider_value(three_seconds_in)
    screen.run_at(3.9)
    assert screen.shown() == screen.expected("멈춤", first + 12_000)

    # 끝 프레임에 닿으면 멈춘다. 거기서 다시 누르면 첫 프레임부터 재생한다
    screen.run_at(60)
    assert screen.shown() == screen.expected("재생", screen.times[-1])
    screen.run_at(90)
    assert screen.shown() == screen.expected("재생", screen.times[-1])
    screen.app.button[0].click().run()
    assert screen.shown() == screen.expected("멈춤", first)
    screen.run_at(90.5)
    assert screen.shown() == screen.expected("멈춤", first + 5_000)


def test_switching_the_segment_while_playing_shows_the_new_one_paused_at_its_first_frame(tmp_path, monkeypatch):
    screen = Screen(tmp_path, monkeypatch)
    save_segment(screen.segment, tmp_path / "segments" / "second.npz")
    screen.app.slider[0].set_value(50).run()
    screen.app.button[0].click().run()
    screen.run_at(2)
    assert screen.shown() == screen.expected("멈춤", screen.times[50] + 2_000)

    screen.app.selectbox[0].select_index(1).run()

    assert screen.shown() == screen.expected("재생", screen.times[0])
    screen.run_at(4)
    assert screen.shown() == screen.expected("재생", screen.times[0])

    # 구간이 바뀌어 자리를 지운 실행이 새 구간을 그리기 전에 끊기면 자리가 없는 채로 누름이 온다. 앞 구간의 화면에서 누른 것이라 버린다
    del screen.app.session_state["position"]
    screen.app.button[0].click().run()
    assert screen.shown() == screen.expected("재생", screen.times[0])


def test_segment_exported_again_under_the_same_name_while_playing_is_shown_paused(tmp_path, monkeypatch):
    screen = Screen(tmp_path, monkeypatch)
    screen.app.button[0].click().run()
    screen.run_at(0.05)
    assert screen.shown() == screen.expected("멈춤", screen.times[0] + 50)

    # 한 시간 뒤의 구간을 같은 이름으로 다시 내보낸다. 구간 고르기는 그대로라서 고르기가 바뀔 때만 멈추면 앞 구간의 재생 시계가 남는다.
    # 아직 첫 프레임이라 슬라이더 번호는 새 구간의 첫 프레임과 같다. 번호만 보고 그대로라고 여기면 앞 구간의 시각에 세로선을 긋는다
    names = ("exchange_time", "gap_last_exchange_time", "gap_next_exchange_time")
    an_hour_later = {name: getattr(screen.segment, name) + 3_600_000 for name in names}
    later = dataclasses.replace(screen.segment, **an_hour_later)
    save_segment(later, tmp_path / "segments" / "fixture.npz")
    screen.segment, screen.times = later, later.exchange_time.tolist()
    screen.run_at(3)

    assert screen.shown() == screen.expected("재생", screen.times[0])
    screen.app.button[0].click().run()
    screen.run_at(4)
    assert screen.shown() == screen.expected("멈춤", screen.times[0] + 1_000)

    # 재생 중에 더 짧은 구간을 또 내보내고 멈춤을 누른다. 앞 구간에서 마지막에 그린 프레임 번호는 새 구간에 없다
    shorter, _ = read_with_a_gap(tmp_path / "shorter", 0, 5)
    save_segment(shorter, tmp_path / "segments" / "fixture.npz")
    screen.segment, screen.times = shorter, shorter.exchange_time.tolist()
    screen.app.button[0].click().run()
    assert screen.shown() == screen.expected("재생", screen.times[0])


def test_screen_without_segment_files_shows_only_how_to_export_one(tmp_path, monkeypatch):
    monkeypatch.setenv("TICKFOLD_SEGMENTS", str(tmp_path))

    screen = AppTest.from_file(APP, default_timeout=30).run()

    assert not screen.exception
    assert str(tmp_path) in screen.info[0].value
    assert "scripts/export_replay_segment.py" in screen.code[0].value
    assert not (screen.selectbox or screen.metric or screen.slider or screen.get("plotly_chart"))