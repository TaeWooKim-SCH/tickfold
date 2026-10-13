import zstandard

from tickfold.replay.files import read_lines, stream_files


def test_files_of_one_stream_come_out_in_the_order_they_were_written(tmp_path):
    for name in (
        "08.depth.ndjson", "08.depth.ndjson.zst", "08.depth.ndjson.1.zst", "08.depth.ndjson.2.zst",
        "08.depth.ndjson.10.zst", "07.depth.ndjson.zst", "07.depth.ndjson.1.zst",
        "08.depth20.ndjson.zst", "08.depth.ndjson.zst.tmp", "snapshots.ndjson.zst",
    ):
        (tmp_path / name).touch()

    def names(stream):
        return [path.name for path in stream_files(tmp_path, stream)]

    assert names("depth") == [
        "07.depth.ndjson.zst", "07.depth.ndjson.1.zst",
        "08.depth.ndjson.zst", "08.depth.ndjson.1.zst", "08.depth.ndjson.2.zst", "08.depth.ndjson.10.zst",
        "08.depth.ndjson",
    ]
    assert names("depth20") == ["08.depth20.ndjson.zst"]
    assert names("snapshots") == ["snapshots.ndjson.zst"]


def test_compressed_and_plain_files_give_the_same_lines(tmp_path):
    content = b'{"rx":1,"m":{}}\n{"rx":2,"m":{}}\n'
    (tmp_path / "08.gap.ndjson").write_bytes(content)
    (tmp_path / "09.gap.ndjson.zst").write_bytes(zstandard.ZstdCompressor().compress(content))

    lines = [b'{"rx":1,"m":{}}', b'{"rx":2,"m":{}}']
    assert read_lines(tmp_path / "08.gap.ndjson") == lines
    assert read_lines(tmp_path / "09.gap.ndjson.zst") == lines


def test_half_written_last_line_is_dropped(tmp_path):
    (tmp_path / "08.trade.ndjson").write_bytes(b'{"rx":1,"m":{}}\n{"rx":2,"m"')
    # 시간이 바뀐 직후라 아직 한 줄도 다 안 쓰인 파일
    (tmp_path / "09.trade.ndjson").write_bytes(b'{"rx":3,"m"')

    assert read_lines(tmp_path / "08.trade.ndjson") == [b'{"rx":1,"m":{}}']
    assert read_lines(tmp_path / "09.trade.ndjson") == []