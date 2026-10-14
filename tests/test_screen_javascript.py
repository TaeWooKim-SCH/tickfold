import shutil
import subprocess
from pathlib import Path

import pytest

TESTS = Path(__file__).parent
NODE = shutil.which("node")
# 브라우저 쪽 코드는 node 의 기본 시험기로 시험한다. node 가 없는 환경에서는 건너뛴다(npm 으로 깔 것은 없다)
pytestmark = pytest.mark.skipif(NODE is None, reason="node 가 없다")


def run_node(*arguments: str) -> None:
    result = subprocess.run([NODE, *arguments], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_playback_and_book_functions_pass_their_node_tests():
    run_node("--test", str(TESTS / "playback.test.js"), str(TESTS / "book.test.js"))



def test_player_script_with_plotly_parses_as_one_module(tmp_path):
    # 화면용 의존성 그룹(screen)을 안 깐 환경에서는 이 시험만 건너뛴다
    pytest.importorskip("plotly")
    pytest.importorskip("streamlit")
    from tickfold.screen.player import player_script

    # 이어 붙인 파일끼리 이름이 겹치거나, 모듈에서만 막히는 문법이 plotly.js 에 있으면 브라우저에서야 알게 된다
    module = tmp_path / "player.mjs"
    module.write_text(player_script())

    run_node("--check", str(module))