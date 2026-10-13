import json
from pathlib import Path

import numpy as np
import pytest
import zstandard

from tickfold.replay.depth20 import parse_depth20
from tickfold.replay.depth_sequence import exchange_time_at, find_gaps, parse_depth_sequence, read_depth_sequence_day

FIXTURE = Path(__file__).parent / "fixtures" / "btcusdt_combined.ndjson"


def fixture_lines(stream: bytes) -> list[bytes]:
    return [line for line in FIXTURE.read_bytes().splitlines() if b"@" + stream + b"@" in line]


def test_fixture_depth_lines_become_columns_equal_to_the_raw_values():
    lines = fixture_lines(b"depth")

    sequence = parse_depth_sequence(lines)

    records = [json.loads(line) for line in lines]
    assert sequence.final_update_id.dtype == np.int64
    assert sequence.received_at.tolist() == [record["rx"] for record in records]
    assert sequence.exchange_time.tolist() == [record["m"]["data"]["E"] for record in records]
    assert sequence.first_update_id.tolist() == [record["m"]["data"]["U"] for record in records]
    assert sequence.final_update_id.tolist() == [record["m"]["data"]["u"] for record in records]


def test_line_that_is_not_one_whole_depth_frame_is_rejected():
    lines = fixture_lines(b"depth")[:4]
    other_stream = [lines[0], fixture_lines(b"depth20")[0], lines[3]]
    # 정전으로 줄이 잘린 파일에 다시 뜬 수집기가 이어 쓰면 두 줄이 한 줄로 붙는다. 앞머리는 온전해서 앞머리만 보면 통과한다
    cut_then_next = [lines[0], lines[1][: lines[1].index(b'"a":[')] + lines[2], lines[3]]

    for broken_lines, message in ((other_stream, "프레임 모양"), (cut_then_next, "붙어 있다")):
        with pytest.raises(ValueError, match=message):
            parse_depth_sequence(broken_lines)


def test_gap_is_reported_with_the_exchange_times_around_it_and_the_updates_missed():
    lines = fixture_lines(b"depth")
    whole = parse_depth_sequence(lines)

    with_hole = find_gaps(parse_depth_sequence(lines[:100] + lines[103:]))

    assert find_gaps(whole).resumed_row.tolist() == []
    assert with_hole.resumed_row.tolist() == [100]
    assert with_hole.last_exchange_time.tolist() == [whole.exchange_time[99]]
    assert with_hole.next_exchange_time.tolist() == [whole.exchange_time[103]]
    assert with_hole.missed_updates.tolist() == [whole.first_update_id[103] - whole.final_update_id[99] - 1]


def test_frames_out_of_order_are_rejected_instead_of_reported_as_a_gap():
    lines = fixture_lines(b"depth")

    # 파일을 이름 순으로 읽으면 나중 파일이 먼저 온다. 그대로 세면 순서가 되돌아간 자리 뒤에 없는 갭이 하나 생긴다
    with pytest.raises(ValueError, match="겹친다"):
        find_gaps(parse_depth_sequence(lines[200:300] + lines[:200] + lines[300:]))


def test_depth20_frames_get_the_exchange_time_of_the_depth_frame_ending_at_the_same_identifier():
    depth_lines = fixture_lines(b"depth")
    frames = parse_depth20(fixture_lines(b"depth20"))
    exchange_time_by_identifier = {
        record["m"]["data"]["u"]: record["m"]["data"]["E"] for record in map(json.loads, depth_lines)
    }

    exchange_time, found = exchange_time_at(parse_depth_sequence(depth_lines), frames.last_update_id)

    expected = [exchange_time_by_identifier[identifier] for identifier in frames.last_update_id.tolist()]
    assert found.all()
    assert exchange_time.tolist() == expected

    # depth 프레임 하나가 없으면 그 번호의 depth20 프레임만 시각을 못 얻는다.
    # 마지막 depth 프레임의 번호는 찾고, 그보다 뒤의 번호(depth 를 먼저 읽은 사이에 더 쓰인 depth20)는 못 찾는다
    missing = [line for line in depth_lines if json.loads(line)["m"]["data"]["u"] != frames.last_update_id[10]]
    last_depth_identifier = json.loads(depth_lines[-1])["m"]["data"]["u"]
    asked = np.append(frames.last_update_id, [last_depth_identifier, last_depth_identifier + 1])
    exchange_time, found = exchange_time_at(parse_depth_sequence(missing), asked)

    assert np.flatnonzero(~found).tolist() == [10, 599]
    assert len(exchange_time) == 598


def test_day_is_read_in_written_order_and_repeated_frames_appear_once(tmp_path):
    lines = fixture_lines(b"depth")
    compressor = zstandard.ZstdCompressor()
    (tmp_path / "08.depth.ndjson.zst").write_bytes(compressor.compress(b"\n".join(lines[:200]) + b"\n"))
    (tmp_path / "08.depth.ndjson.1.zst").write_bytes(compressor.compress(b"\n".join(lines[200:300]) + b"\n"))
    # 압축을 끝내고 원본을 지우기 전에 죽으면, 다시 뜬 수집기가 남은 원본에 이어 써서 다음 파일이 앞 파일 내용으로 시작한다
    (tmp_path / "08.depth.ndjson.2.zst").write_bytes(compressor.compress(b"\n".join(lines[200:400]) + b"\n"))
    (tmp_path / "09.depth.ndjson").write_bytes(b"\n".join(lines[400:]) + b"\n")
    # 같은 폴더의 depth20 파일은 읽지 않는다
    (tmp_path / "08.depth20.ndjson").write_bytes(b"\n".join(fixture_lines(b"depth20")[:5]) + b"\n")

    sequence = read_depth_sequence_day(tmp_path)

    expected = parse_depth_sequence(lines)
    assert sequence.received_at.tolist() == expected.received_at.tolist()
    assert sequence.exchange_time.tolist() == expected.exchange_time.tolist()
    assert sequence.first_update_id.tolist() == expected.first_update_id.tolist()
    assert sequence.final_update_id.tolist() == expected.final_update_id.tolist()