import json
import re
from pathlib import Path

import numpy as np
import pytest
import zstandard

from tickfold.replay.depth20 import ASK, BID, PRICE, QUANTITY, parse_depth20, read_depth20_day

FIXTURE = Path(__file__).parent / "fixtures" / "btcusdt_combined.ndjson"


def fixture_depth20_lines() -> list[bytes]:
    return [line for line in FIXTURE.read_bytes().splitlines() if b"@depth20@" in line]


def test_fixture_frames_become_an_array_equal_to_the_raw_strings():
    lines = fixture_depth20_lines()

    frames = parse_depth20(lines)

    records = [json.loads(line) for line in lines]
    assert frames.book.shape == (598, 2, 20, 2)
    assert frames.book.dtype == np.float64
    assert frames.last_update_id.dtype == frames.received_at.dtype == np.int64
    # 축 번호는 화면, 대조, 모델이 같이 쓰는 약속이라 픽스처 첫 줄에서 눈으로 읽은 값으로 고정한다
    assert (frames.book[0, BID, 0, PRICE], frames.book[0, ASK, 0, QUANTITY]) == (84600.0, 1.40574)
    assert frames.book[:, BID].tolist() == [
        [[float(price), float(quantity)] for price, quantity in record["m"]["data"]["bids"]] for record in records
    ]
    assert frames.book[:, ASK].tolist() == [
        [[float(price), float(quantity)] for price, quantity in record["m"]["data"]["asks"]] for record in records
    ]
    assert frames.last_update_id.tolist() == [record["m"]["data"]["lastUpdateId"] for record in records]
    assert frames.received_at.tolist() == [record["rx"] for record in records]


def test_line_without_twenty_levels_on_each_side_is_rejected():
    lines = fixture_depth20_lines()[:3]
    first_bid = re.search(rb'\["\d+\.\d+","\d+\.\d+"\],', lines[1])
    without_first_bid = lines[1][: first_bid.start()] + lines[1][first_bid.end() :]
    # 매수 19 + 매도 21 이면 숫자는 그대로 80개라서, 줄 모양을 안 보면 예외 없이 한 칸씩 밀려 읽힌다
    moved_to_asks = without_first_bid.replace(b'"asks":[', b'"asks":[' + first_bid.group(), 1)

    for broken_line in (without_first_bid, moved_to_asks):
        with pytest.raises(ValueError, match="20레벨"):
            parse_depth20([lines[0], broken_line, lines[2]])


def test_day_is_read_in_written_order_and_repeated_frames_appear_once(tmp_path):
    lines = fixture_depth20_lines()
    compressor = zstandard.ZstdCompressor()
    (tmp_path / "08.depth20.ndjson.zst").write_bytes(compressor.compress(b"\n".join(lines[:200]) + b"\n"))
    (tmp_path / "08.depth20.ndjson.1.zst").write_bytes(compressor.compress(b"\n".join(lines[200:300]) + b"\n"))
    # 압축을 끝내고 원본을 지우기 전에 죽으면, 다시 뜬 수집기가 남은 원본에 이어 써서 다음 파일이 앞 파일 내용으로 시작한다
    (tmp_path / "08.depth20.ndjson.2.zst").write_bytes(compressor.compress(b"\n".join(lines[200:400]) + b"\n"))
    # 지금 쓰고 있는 시간대는 압축 전 파일뿐이다
    (tmp_path / "09.depth20.ndjson").write_bytes(b"\n".join(lines[400:]) + b"\n")

    frames = read_depth20_day(tmp_path)

    expected = parse_depth20(lines)
    assert frames.last_update_id.tolist() == expected.last_update_id.tolist()
    assert np.array_equal(frames.book, expected.book)
    assert np.array_equal(frames.received_at, expected.received_at)


def test_frame_going_backward_to_an_identifier_never_read_is_rejected(tmp_path):
    lines = fixture_depth20_lines()[:5]
    lines[3] = re.sub(rb'"lastUpdateId":\d+', b'"lastUpdateId":1', lines[3])
    (tmp_path / "08.depth20.ndjson").write_bytes(b"\n".join(lines) + b"\n")

    with pytest.raises(ValueError, match="읽은 적이 없다"):
        read_depth20_day(tmp_path)