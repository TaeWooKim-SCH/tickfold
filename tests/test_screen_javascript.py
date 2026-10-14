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