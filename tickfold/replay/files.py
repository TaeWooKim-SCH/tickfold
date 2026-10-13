"""저장된 원본 파일을 쓰인 순서대로 찾아 읽는다. 파일 이름 순서가 시간 순서가 아니라서 따로 둔다."""

import re
from pathlib import Path

import zstandard

# 08.depth.ndjson.zst, 08.depth.ndjson.1.zst, 08.depth.ndjson, snapshots.ndjson.zst. 압축 중에 생기는 .zst.tmp 는 안 맞는다
_FILE_NAME = re.compile(r"((?P<hour>\d\d)\.)?(?P<stream>[a-z0-9]+)\.ndjson(?P<compressed>(\.(?P<part>\d+))?\.zst)?")

# 압축 전 .ndjson 은 그 시간대에 마지막으로 연 파일이다. 압축되면 가장 큰 파트 번호를 받는다
_NOT_COMPRESSED_YET = float("inf")


def stream_files(day_directory: Path, stream: str) -> list[Path]:
    """한 날짜 폴더에서 한 스트림의 파일을 쓰인 순서대로 돌려준다."""
    found = []
    for path in day_directory.iterdir():
        name = _FILE_NAME.fullmatch(path.name)
        if name is None or name["stream"] != stream:
            continue
        # 수집기는 같은 시간대의 첫 파일을 .zst 로, 재시작 뒤의 파일을 .1.zst, .2.zst 로 압축한다.
        # 이름으로 정렬하면 '1' 이 'z' 보다 앞이라 나중 파일이 먼저 나온다
        part = int(name["part"] or 0) if name["compressed"] else _NOT_COMPRESSED_YET
        found.append((name["hour"] or "", part, path))
    return [path for _, _, path in sorted(found)]


def read_lines(path: Path) -> list[bytes]:
    """파일 하나의 온전한 줄을 돌려준다. 쓰는 중이라 덜 쓴 마지막 줄은 버린다."""
    content = path.read_bytes()
    if path.suffix == ".zst":
        content = zstandard.ZstdDecompressor().decompress(content)
    # 마지막 조각은 개행 뒤의 빈 바이트이거나 덜 쓴 줄이다
    return content.split(b"\n")[:-1]