"""원본 메시지를 종목/스트림/시간 단위 NDJSON 파일에 append 하고, 시간이 바뀌면 zstd로 압축한다.
 
파일 배치 (root = data/raw):
    {symbol}/{YYYY-MM-DD}/{HH}.{stream}.ndjson        쓰는 중
    {symbol}/{YYYY-MM-DD}/{HH}.{stream}.ndjson.zst    닫힌 뒤
    {symbol}/{YYYY-MM-DD}/snapshots.ndjson(.zst)      REST 스냅샷, 하루 단위
 
줄 형식: {"rx": 수신시각ns, "m": 원본}  — 원본 바이트는 재직렬화 없이 그대로 붙인다.
시간 경계는 로컬 수신 시각(rx)의 UTC 기준이다.
"""

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import IO

import zstandard
 
SNAPSHOT_STREAM = "snapshots"

@dataclass
class _Open:
    period: tuple
    fh: IO[bytes]

class RawWriter:
    def __init__(self, root: Path, zstd_level: int = 3):
        self.root = Path(root)
        self.level = zstd_level
        self._files: dict[tuple[str, str], _Open] = {}  # (symbol, stream) -> 열린 파일
        self._compress_leftovers()

    # ── 쓰기 ──────────────────────────────────────────────────────────
    
    def write(self, symbol: str, stream: str, rx_ns: int, raw: bytes) -> None:
        date, hour = self._date_hour(rx_ns)
        path = self.root / symbol / date / f"{hour}.{stream}.ndjson"
        fh = self._ensure((symbol, stream), (date, hour), path)
        fh.write(b'{"rx":%d,"m":' % rx_ns + raw + b"}\n")

    def write_snapshot(self, symbol: str, rx_ns: int, raw: bytes) -> None:
        date, _ = self._date_hour(rx_ns)
        path = self.root / symbol / date / f"{SNAPSHOT_STREAM}.ndjson"
        fh = self._ensure((symbol, SNAPSHOT_STREAM), (date,), path)
        fh.write(b'{"rx":%d,"m":' % rx_ns + raw + b"}\n")

    def flush(self) -> None:
        """열린 파일을 전부 OS에 내리고 디스크에 동기화한다. 주기 호출 + SIGTERM 시."""
        for o in self._files.values():
            o.fh.flush()
            os.fsync(o.fh.fileno())

    def close(self) -> None:
        for o in self._files.values():
            self._close_and_compress(o.fh)
        self._files.clear()

    # ── 내부 ──────────────────────────────────────────────────────────

    @staticmethod
    def _date_hour(rx_ns: int) -> tuple[str, str]:
        dt = datetime.fromtimestamp(rx_ns / 1e9, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d"), dt.strftime("%H")

    def _ensure(self, key: tuple[str, str], period: tuple, path: Path) -> IO[bytes]:
        cur = self._files.get(key)
        if (cur is not None and cur.period == period):
            return cur.fh
        if (cur is not None):
            self._close_and_compress(cur.fh)
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(path, "ab")
        self._files[key] = _Open(period, fh)
        return fh

    def _close_and_compress(self, fh: IO[bytes]) -> None:
        fh.close()
        self._compress(Path(fh.name))

    def _compress(self, src: Path) -> None:
        dst = src.with_name(src.name + ".zst")
        n = 1
        while (dst.exists()): # 같은 시간 파일이 이미 압축되어 있으면 덮어쓰지 않고 파트 번호를 붙임
            dst = src.with_name(f"{src.name}.{n}.zst")
            n += 1
        cctx = zstandard.ZstdCompressor(level=self.level)
        with open(src, "rb") as i, open(dst, "wb") as o:
            cctx.copy_stream(i, o)
        src.unlink()

    def _compress_leftovers(self) -> None:
        """비정상 종료로 남은 .ndjson 중 이미 지난 시간대의 파일을 압축한다.
        현재 시간대의 파일은 그대로 두고 이어서 append 한다."""

        now_date, now_hour = self._date_hour(_now_ns())
        for src in self.root.glob("*/*/*.ndjson"):
            date = src.parent.name
            first = src.name.split(".")[0]
            if (first == SNAPSHOT_STREAM):
                is_past = date < now_date
            else:
                is_past = (date, first) < (now_date, now_hour)

            if (is_past):
                self._compress(src)

def _now_ns() -> int:
    import time

    return time.time_ns()