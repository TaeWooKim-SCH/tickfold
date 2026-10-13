"""depth 프레임의 순번과 거래소 시각만 읽는다. 갭이 어디서 얼마나 났는지와 depth20 프레임의 거래소 시각이 여기서 나온다."""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tickfold.replay.files import read_lines, stream_files
from tickfold.replay.repeats import first_time_read

# 순번과 시각은 줄 앞머리에 다 있다. 레벨 목록까지 풀면 하루치가 몇 배 느려진다
_FRAME_HEAD = re.compile(
    rb'^\{"rx":(\d+),"m":\{"stream":"[a-z0-9]+@depth@100ms","data":'
    rb'\{"e":"depthUpdate","E":(\d+),"s":"[A-Z0-9]+","U":(\d+),"u":(\d+),"b":\[',
    re.MULTILINE,
)


@dataclass(frozen=True)
class DepthSequence:
    """depth 프레임마다 한 행. 한 프레임은 first_update_id 부터 final_update_id 까지의 변경을 묶은 것이다."""

    received_at: np.ndarray  # (프레임,) int64. 수신 시각 나노초
    exchange_time: np.ndarray  # (프레임,) int64. 거래소 시각 E, 밀리초
    first_update_id: np.ndarray  # (프레임,) int64. 거래소 필드 U
    final_update_id: np.ndarray  # (프레임,) int64. 거래소 필드 u. depth20 의 lastUpdateId 와 같은 번호다


@dataclass(frozen=True)
class SequenceGaps:
    """순번이 끊긴 곳마다 한 행. 길이는 수신 시각이 아니라 거래소 시각으로 잰다."""

    resumed_row: np.ndarray  # (갭,) 갭 뒤 첫 프레임의 행 번호
    last_exchange_time: np.ndarray  # (갭,) 갭 직전 프레임의 거래소 시각, 밀리초
    next_exchange_time: np.ndarray  # (갭,) 갭 직후 프레임의 거래소 시각, 밀리초
    missed_updates: np.ndarray  # (갭,) 놓친 변경 수


def parse_depth_sequence(lines: list[bytes]) -> DepthSequence:
    """저장된 depth 줄들에서 순번과 시각을 배열로 뽑는다."""
    content = b"\n".join(lines)
    heads = _FRAME_HEAD.findall(content)
    if len(heads) != len(lines):
        raise ValueError(f"depth {len(lines)}줄 중 {len(lines) - len(heads)}줄이 depth 프레임 모양이 아니다")
    # 앞머리만 보므로 덜 쓴 줄 뒤에 다음 줄이 붙은 줄도 한 프레임으로 읽힌다. 정전으로 줄이 잘린 파일에 수집기가 이어 쓰면 그렇게 된다
    line_starts = content.count(b'{"rx":')
    if line_starts != len(lines):
        raise ValueError(f"depth {len(lines)}줄에 줄 머리가 {line_starts}번 나온다. 덜 쓴 줄 뒤에 다음 줄이 붙어 있다")
    columns = np.array(heads, dtype="S").astype(np.int64).reshape(len(lines), 4)
    return DepthSequence(columns[:, 0], columns[:, 1], columns[:, 2], columns[:, 3])


def read_depth_sequence_day(day_directory: Path) -> DepthSequence:
    """한 날짜 폴더의 depth 순번과 시각을 쓰인 순서대로 이어 붙인다."""
    paths = stream_files(day_directory, "depth")
    if not paths:
        # 빈 배열을 돌려주면 폴더를 잘못 짚은 것과 그날 수집이 없던 것이 뒤에서 구별되지 않는다
        raise FileNotFoundError(f"{day_directory} 에 depth 파일이 없다")
    parts = [parse_depth_sequence(read_lines(path)) for path in paths]
    final_update_id = np.concatenate([part.final_update_id for part in parts])
    is_new = first_time_read(final_update_id, day_directory)
    return DepthSequence(
        np.concatenate([part.received_at for part in parts])[is_new],
        np.concatenate([part.exchange_time for part in parts])[is_new],
        np.concatenate([part.first_update_id for part in parts])[is_new],
        final_update_id[is_new],
    )


def find_gaps(sequence: DepthSequence) -> SequenceGaps:
    """순번 사슬이 끊긴 곳을 찾는다. 갭 파일이 없던 때의 갭도 여기서는 나온다.

    넘겨받은 프레임 사이만 본다. 하루치씩 넘기면 자정에 걸친 갭은 빠지므로, 며칠을 볼 때는 이어 붙여서 넘긴다.
    """
    missed = sequence.first_update_id[1:] - sequence.final_update_id[:-1] - 1
    if (missed < 0).any():
        raise ValueError(f"depth 프레임 {(missed < 0).sum()}개가 앞 프레임과 순번이 겹친다")
    resumed_row = np.flatnonzero(missed > 0) + 1
    return SequenceGaps(
        resumed_row,
        sequence.exchange_time[resumed_row - 1],
        sequence.exchange_time[resumed_row],
        missed[resumed_row - 1],
    )


def exchange_time_at(sequence: DepthSequence, last_update_id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """depth20 프레임의 식별자마다 같은 번호로 끝나는 depth 프레임의 거래소 시각을 찾는다.

    (찾은 프레임의 거래소 시각, 찾았는지) 를 돌려준다. 못 찾은 프레임에는 시각이 없으므로 앞의 배열은 찾은 수만큼만 있다.
    호출자는 같은 마스크로 depth20 배열을 줄여야 행이 맞는다. sequence 의 끝 순번은 늘기만 해야 한다.
    """
    position = np.searchsorted(sequence.final_update_id, last_update_id)
    found = np.zeros(len(last_update_id), dtype=bool)
    inside = position < len(sequence.final_update_id)
    found[inside] = sequence.final_update_id[position[inside]] == last_update_id[inside]
    return sequence.exchange_time[position[found]], found