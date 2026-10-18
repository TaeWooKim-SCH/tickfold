"""저장된 원본 파일을 쓰인 순서대로 찾아 읽는다. 파일 이름 순서가 시간 순서가 아니라서 따로 둔다."""

import re
from datetime import datetime, timedelta
from pathlib import Path

import zstandard

# 08.depth.ndjson.zst, 08.depth.ndjson.1.zst, 08.depth.ndjson, snapshots.ndjson.zst. 압축 중에 생기는 .zst.tmp 는 안 맞는다
_FILE_NAME = re.compile(r"((?P<hour>\d\d)\.)?(?P<stream>[a-z0-9]+)\.ndjson(?P<compressed>(\.(?P<part>\d+))?\.zst)?")

# 압축 전 .ndjson 은 그 시간대에 마지막으로 연 파일이다. 압축되면 가장 큰 파트 번호를 받는다
_NOT_COMPRESSED_YET = float("inf")

# 거래소 시각 범위를 읽을 때 수신 시각으로는 이만큼 넓게 읽는다. 복원기, 대조, 화면 구간 내보내기가 같이 쓴다.
# 폴더는 수신 시각의 날짜로 나뉘고 구간은 거래소 시각으로 자른다. 전달이 밀리면 프레임이 다음 날 폴더에 들어간다.
# 연결은 핑 20초와 응답 대기 20초보다 오래 밀린 채로는 못 버티므로 60초면 넉넉하다.
# 앞쪽으로도 같은 여유를 두는 것은 우리 시계가 거래소보다 늦을 때 수신 시각이 거래소 시각보다 앞서 찍히기 때문이다
DELIVERY_MARGIN = timedelta(seconds=60)

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


def stream_files_between(symbol_directory: Path, stream: str, first: datetime, last: datetime) -> list[Path]:
    """수신 시각이 first 의 시간대부터 last 의 시간대까지인 한 스트림의 파일을 날짜 폴더를 넘어 쓰인 순서대로 돌려준다."""
    first_hour, last_hour = f"{first:%Y-%m-%d %H}", f"{last:%Y-%m-%d %H}"
    paths = []
    day = first.date()
    while day <= last.date():
        if (symbol_directory / str(day)).is_dir():
            # 파일 이름의 앞 두 글자가 수신 시각의 시다
            day_paths = stream_files(symbol_directory / str(day), stream)
            paths += [path for path in day_paths if first_hour <= f"{day} {path.name[:2]}" <= last_hour]
        day += timedelta(days=1)
    if not paths:
        # 빈 목록을 돌려주면 폴더를 잘못 짚은 것이 빈 결과가 되어, 날짜별로 더할 때 그날이 소리 없이 빠진다
        raise FileNotFoundError(f"{symbol_directory} 에 {first} ~ {last} 의 {stream} 파일이 없다")
    return paths


def read_lines(path: Path) -> list[bytes]:
    """파일 하나의 온전한 줄을 돌려준다. 쓰는 중이라 덜 쓴 마지막 줄은 버린다."""
    content = path.read_bytes()
    if path.suffix == ".zst":
        content = zstandard.ZstdDecompressor().decompress(content)
    # 마지막 조각은 개행 뒤의 빈 바이트이거나 덜 쓴 줄이다
    return content.split(b"\n")[:-1]