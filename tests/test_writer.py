import json
from datetime import datetime, timezone
from pathlib import Path
 
import zstandard
 
from tickfold.collector.writer import RawWriter

def ns(y, m, d, h, mi=0):
    return int(datetime(y, m, d, h, mi, tzinfo=timezone.utc).timestamp() * 1_000_000_000)

def read_zst(path: Path) -> bytes:
    with open(path, "rb") as f:
        return zstandard.ZstdDecompressor().stream_reader(f).read()

RAW1 = b'{"e":"depthUpdate","U":157,"u":160,"b":[["50000.00","2.5"]],"a":[]}'
RAW2 = b'{"e":"depthUpdate","U":161,"u":161,"b":[],"a":[["50001.00","0.00000000"]]}'
T10 = ns(2026, 9, 10, 10)
T11 = ns(2026, 9, 10, 11)

def test_line_format_keeps_raw_bytes_intact(tmp_path):
    w = RawWriter(tmp_path)
    w.write("BTCUSDT", "depth", T10, RAW1)
    w.write("BTCUSDT", "depth", T10 + 1, RAW2)
    w.flush()
 
    lines = (tmp_path / "BTCUSDT" / "2026-09-10" / "10.depth.ndjson").read_bytes().splitlines()
    assert lines[0] == b'{"rx":%d,"m":' % T10 + RAW1 + b"}"
    assert json.loads(lines[1]) == {"rx": T10 + 1, "m": json.loads(RAW2)}

def test_hour_rollover_compresses_previous_file(tmp_path):
    w = RawWriter(tmp_path)
    w.write("BTCUSDT", "depth", T10, RAW1)
    w.write("BTCUSDT", "depth", T11, RAW2)  # 다음 시간 → 10시 파일 닫고 압축
 
    day = tmp_path / "BTCUSDT" / "2026-09-10"
    assert not (day / "10.depth.ndjson").exists()
    assert (day / "11.depth.ndjson").exists()
    assert read_zst(day / "10.depth.ndjson.zst") == b'{"rx":%d,"m":' % T10 + RAW1 + b"}\n"

def test_streams_and_symbols_go_to_separate_files(tmp_path):
    w = RawWriter(tmp_path)
    w.write("BTCUSDT", "depth", T10, RAW1)
    w.write("BTCUSDT", "trade", T10, b'{"e":"trade"}')
    w.write("ETHUSDT", "depth", T10, RAW1)
    w.close()
 
    assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*.zst")) == [
        "BTCUSDT/2026-09-10/10.depth.ndjson.zst",
        "BTCUSDT/2026-09-10/10.trade.ndjson.zst",
        "ETHUSDT/2026-09-10/10.depth.ndjson.zst",
    ]

def test_close_compresses_everything_and_leaves_no_ndjson(tmp_path):
    w = RawWriter(tmp_path)
    w.write("BTCUSDT", "depth", T10, RAW1)
    w.write_snapshot("BTCUSDT", T10, b'{"lastUpdateId":156,"bids":[],"asks":[]}')
    w.close()
    assert list(tmp_path.rglob("*.ndjson")) == []
    assert len(list(tmp_path.rglob("*.zst"))) == 2

def test_past_leftover_is_compressed_on_startup(tmp_path):
    day = tmp_path / "BTCUSDT" / "2020-01-01"
    day.mkdir(parents=True)
    (day / "00.depth.ndjson").write_bytes(b'{"rx":1,"m":{}}\n')
    (day / "snapshots.ndjson").write_bytes(b'{"rx":1,"m":{}}\n')
 
    RawWriter(tmp_path)  # 생성 시점에 정리
 
    assert not (day / "00.depth.ndjson").exists()
    assert (day / "00.depth.ndjson.zst").exists()
    assert (day / "snapshots.ndjson.zst").exists()
 
def test_current_hour_leftover_is_appended_not_compressed(tmp_path, monkeypatch):
    import tickfold.collector.writer as mod
 
    monkeypatch.setattr(mod, "_now_ns", lambda: T10 + 60 * 1_000_000_000)  # 지금은 10시 1분
    day = tmp_path / "BTCUSDT" / "2026-09-10"
    day.mkdir(parents=True)
    (day / "10.depth.ndjson").write_bytes(b"first\n")
 
    w = RawWriter(tmp_path)
    w.write("BTCUSDT", "depth", T10 + 60 * 1_000_000_000, RAW1)
    w.flush()
 
    assert not (day / "10.depth.ndjson.zst").exists()
    assert (day / "10.depth.ndjson").read_bytes().startswith(b"first\n{")

def test_snapshot_file_is_daily(tmp_path):
    w = RawWriter(tmp_path)
    w.write_snapshot("BTCUSDT", T10, b"{}")
    w.write_snapshot("BTCUSDT", T11, b"{}")  # 같은 날 → 같은 파일
    w.write_snapshot("BTCUSDT", ns(2026, 9, 11, 0), b"{}")  # 다음 날 → 이전 파일 압축
 
    assert (tmp_path / "BTCUSDT" / "2026-09-10" / "snapshots.ndjson.zst").exists()
    assert (tmp_path / "BTCUSDT" / "2026-09-11" / "snapshots.ndjson").exists()