"""
재생 화면에 쓸 구간 하나를 npz 파일로 내보낸다. 시각은 UTC 거래소 시각이다.

사용법 (프로젝트 루트에서):
    uv run python scripts/export_replay_segment.py 2026-10-07T07:25:00 2026-10-07T07:40:00 segments/gap-0732.npz

하루를 통째로 읽어서 자르므로 30초쯤 걸리고 메모리를 2GB 쯤 쓴다. 수집기와 같은 서버라서 nice 를 걸어 돌린다.
"""

import argparse
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tickfold.replay.segment import read_segment, save_segment


def utc_time(text: str) -> datetime:
    """명령줄의 시각을 읽는다. 시간대가 없으면 UTC 로 본다."""
    time = datetime.fromisoformat(text)
    return time if time.tzinfo else time.replace(tzinfo=timezone.utc)


def main() -> None:
    parser = argparse.ArgumentParser(description="재생 화면에 쓸 구간 하나를 npz 파일로 내보낸다")
    parser.add_argument("start", type=utc_time, help="구간 시작(포함). UTC, 예: 2026-10-07T07:25:00")
    parser.add_argument("end", type=utc_time, help="구간 끝(제외). UTC")
    parser.add_argument("output", type=Path, help="내보낼 .npz 파일")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("TICKFOLD_DATA_ROOT", "data/raw")))
    arguments = parser.parse_args()
    if arguments.start >= arguments.end:
        parser.error(f"시작 {arguments.start} 이 끝 {arguments.end} 보다 앞이어야 한다")
    # 화면은 구간 전체를 웹소켓 메시지 하나로 받는다. 한 시간이 36MB, 하루가 750MB 이고 Streamlit 한도 200MB 를 넘으면 영어 에러만 뜬다
    if arguments.end - arguments.start > timedelta(hours=2):
        parser.error("재생 화면용 구간은 2시간 이하로 자른다")
    # numpy 는 .npz 로 안 끝나는 이름에 .npz 를 붙여 쓴다. 그러면 아래 검사와 크기 출력이 실제로 쓰인 파일이 아닌 이름을 본다
    if arguments.output.suffix != ".npz":
        parser.error(f"{arguments.output} 은 .npz 로 끝나야 한다")
    # 원본 폴더에는 수집기만 쓴다
    if arguments.data_root.resolve() in arguments.output.resolve().parents:
        parser.error(f"{arguments.output} 은 원본 폴더 {arguments.data_root} 아래다. 다른 곳에 쓴다")

    segment = read_segment(arguments.data_root / arguments.symbol, arguments.start, arguments.end)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    save_segment(segment, arguments.output)
    # 끝이 아직 안 온 시각이면 받은 데까지만 담긴다. 실제로 담긴 범위를 같이 찍어야 요청한 구간과 견줄 수 있다
    edges = segment.exchange_time[[0, -1]].tolist()
    first, last = (datetime.fromtimestamp(edge / 1000, tz=timezone.utc).isoformat() for edge in edges)
    gap_seconds = ((segment.gap_next_exchange_time - segment.gap_last_exchange_time) / 1000).tolist()
    print(f"{arguments.output}: 프레임 {len(segment.exchange_time):,}개 ({first} ~ {last})")
    print(f"갭 {len(gap_seconds)}개 {gap_seconds}초, {arguments.output.stat().st_size / 1e6:.2f}MB")


if __name__ == "__main__":
    main()