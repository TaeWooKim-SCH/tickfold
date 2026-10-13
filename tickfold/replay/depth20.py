"""거래소가 따로 보내 주는 상위 20레벨(depth20) 프레임을 배열로 읽는다. 화면이 그리고 대조가 견주는 기준이다."""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tickfold.replay.files import read_lines, stream_files
from tickfold.replay.repeats import first_time_read

LEVELS = 20

# book 배열의 축 번호. 화면, 대조, 모델이 같은 배열을 주고받으므로 여기 한 곳에만 적는다
BID, ASK = 0, 1
PRICE, QUANTITY = 0, 1

_LEVEL = rb'\["\d+\.\d+","\d+\.\d+"\],?'
_SIDE = rb"\[(?:" + _LEVEL + rb"){%d}\]" % LEVELS
# 줄 전체가 이 모양일 때만 프레임으로 센다. 레벨이 하나라도 모자라면 숫자가 옆 칸으로 밀려서 틀린 줄도 조용히 읽히기 때문이다
_FRAME = re.compile(
    rb'^\{"rx":(\d+),"m":\{"stream":"[a-z0-9]+@depth20@100ms","data":\{"lastUpdateId":(\d+),'
    rb'"bids":' + _SIDE + rb',"asks":' + _SIDE + rb"\}\}\}$",
    re.MULTILINE,
)
_PRICE_OR_QUANTITY = re.compile(rb'"(\d+\.\d+)"')


@dataclass(frozen=True)
class Depth20Frames:
    """프레임마다 한 행. 거래소 시각은 depth20 에 없어서 여기 안 들어 있다."""

    last_update_id: np.ndarray  # (프레임,) int64
    received_at: np.ndarray  # (프레임,) int64. 수신 시각 나노초
    book: np.ndarray  # (프레임, 매수·매도, 레벨, 가격·수량) float64. 레벨 0 이 최우선 호가


def parse_depth20(lines: list[bytes]) -> Depth20Frames:
    """저장된 depth20 줄들을 배열 하나로 바꾼다."""
    content = b"\n".join(lines)
    frames = _FRAME.findall(content)
    if len(frames) != len(lines):
        raise ValueError(f"depth20 {len(lines)}줄 중 {len(lines) - len(frames)}줄이 20레벨 프레임 모양이 아니다")
    received_at = np.array([frame[0] for frame in frames], dtype="S").astype(np.int64)
    last_update_id = np.array([frame[1] for frame in frames], dtype="S").astype(np.int64)
    # 줄 안에서 따옴표로 싸인 소수는 가격과 수량뿐이고, 매수 20레벨 다음에 매도 20레벨이 (가격, 수량) 순서로 온다
    numbers = np.array(_PRICE_OR_QUANTITY.findall(content), dtype="S").astype(np.float64)
    return Depth20Frames(last_update_id, received_at, numbers.reshape(len(lines), 2, LEVELS, 2))


def read_depth20_day(day_directory: Path) -> Depth20Frames:
    """한 날짜 폴더의 depth20 을 쓰인 순서대로 이어 붙인다."""
    paths = stream_files(day_directory, "depth20")
    if not paths:
        # 빈 배열을 돌려주면 폴더를 잘못 짚은 것과 그날 수집이 없던 것이 뒤에서 구별되지 않는다
        raise FileNotFoundError(f"{day_directory} 에 depth20 파일이 없다")
    parts = [parse_depth20(read_lines(path)) for path in paths]
    last_update_id = np.concatenate([part.last_update_id for part in parts])
    # 압축되는 찰나나 크래시 뒤에는 같은 내용이 두 파일로 보인다. 식별자는 늘기만 하므로 앞선 값 이하면 이미 읽은 프레임이다
    is_new = first_time_read(last_update_id, day_directory)
    
    return Depth20Frames(
        last_update_id[is_new],
        np.concatenate([part.received_at for part in parts])[is_new],
        np.concatenate([part.book for part in parts])[is_new],
    )