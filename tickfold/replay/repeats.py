"""파일을 이어 읽을 때 두 번 나온 프레임을 가려낸다. 같은 내용이 두 파일로 보이는 때가 있어서 스트림마다 같은 검사가 필요하다."""

from pathlib import Path

import numpy as np


def first_time_read(identifiers: np.ndarray, source: Path) -> np.ndarray:
    """이어 붙인 식별자에서 처음 읽는 프레임만 참인 마스크를 돌려준다."""
    # 압축되는 찰나나 크래시 뒤에는 같은 내용이 두 파일로 보인다. 식별자는 늘기만 하므로 앞선 값 이하면 이미 읽은 프레임이다
    is_new = np.ones(len(identifiers), dtype=bool)
    is_new[1:] = identifiers[1:] > np.maximum.accumulate(identifiers)[:-1]
    # 버려도 되는 것은 이미 읽은 프레임뿐이다. 처음 보는 식별자가 뒤로 갔는데 버리면 그 뒤 프레임까지 조용히 사라진다
    never_read = ~np.isin(identifiers[~is_new], identifiers[is_new])
    if never_read.any():
        raise ValueError(f"{source}: 식별자가 뒤로 간 프레임 {never_read.sum()}개는 앞에서 읽은 적이 없다")
    return is_new