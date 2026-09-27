"""
바이낸스 현물 combined stream 수집: 호가 델타, 체결, 검증용 부분 호가

프레임은 {"stream": "btcusdt@depth@100ms", "data": {...}} 로 감싸져 온다.
감싼 전체를 바이트 그대로 저장한다. 파일명이 종목·스트림을 갖고 있어 래퍼가 중복이지만,
잘라내는 순간 "원본 그대로" 가 아니게 되고 중복분은 zstd 가 지운다.
"""

import asyncio
import json
import logging
import time
from pathlib import Path

import aiohttp
import websockets

from tickfold.collector.binance_rest import BinanceBanned, BinanceRestError, fetch_snapshot
from tickfold.collector.orderbook import OrderBook
from tickfold.collector.writer import RawWriter

log = logging.getLogger(__name__)

WS_BASE = "wss://stream.binance.com:9443/stream?streams="
DEPTH = "depth"
STREAMS = (f"{DEPTH}@100ms", "trade", "depth20@100ms")  # 호가창에 적용하는 것은 depth 뿐

# 재동기화용
SNAPSHOT_LIMIT = 5000
MAX_BACKOFF_S = 60.0
RESYNC_COOLDOWN_S = 2.0  # 재동기화 실패 후 다음 시도까지. 가중치를 연속으로 태우지 않으려고
RESYNC_MAX_TRIES = 3

# 연결루프용
FLUSH_S = 5.0
BACKOFF_S = 1.0

def stream_names(symbols) -> str:
    """combined stream URL의 streams 파라미터. 종목 × 스트림 전부를 한 연결로 받는다."""
    return "/".join(f"{s.lower()}@{stream}" for s in symbols for stream in STREAMS)

def split_stream(name: str) -> tuple[str, str]:
    """'btcusdt@depth@100ms' -> ('BTCUSDT', 'depth'). 속도 접미사는 파일명에서 뺀다."""
    parts = name.split("@")
    return parts[0].upper(), parts[1]

class StreamHandler:
    """프레임 하나를 저장하고 호가창에 반영한다.

    handle()은 절대 await 하지 않는다. 여기서 기다리면 그 시간만큼 소켓을 못 읽는다.
    소켓과 분리해 둔 덕에 테스트가 프레임 바이트만으로 돌아간다.
    """

    def __init__(self, session: aiohttp.ClientSession, writer: RawWriter, symbols):
        self._session = session
        self._writer = writer
        self.books = {s.upper(): OrderBook() for s in symbols}
        self.gaps = 0
        self.tasks: set[asyncio.Task] = set()
        self._resyncing: set[str] = set()

    def handle(self, frame: bytes, rx_ns: int) -> None:
        msg = json.loads(frame)
        if ("stream" not in msg):
            # 제어·오류 프레임. 버리지 않고 남긴다. 여기서 예외를 내면 연결이 통째로 끊긴다
            self._writer.write("_control", "raw", rx_ns, frame)
            return
        
        symbol, stream = split_stream(msg["stream"])
        self._writer.write(symbol, stream, rx_ns, frame) # 원본 저장이 항상 먼저

        book = self.books.get(symbol)
        if (stream != DEPTH or book is None):
            return

        last_u = book.last_u # apply_delta가 갭에서 None으로 지우므로 미리 집어둔다
        if (book.apply_delta(msg["data"]) == "gap"):
            self.gaps += 1
            self._write_gap(symbol, rx_ns, last_u, msg["data"])

        if (not book.synced):
            self._start_resync(symbol)


    def _write_gap(self, symbol: str, rx_ns: int, last_u, data: dict) -> None:
        # 갭 기록도 같은 writer로 남긴다. 로테이션·압축을 그대로 쓰고 그 갭이 난 데이터 옆에 남는다
        record = {
            "event": "gap",
            "last_u": last_u,
            "U": data["U"],
            "u": data["u"],
            "missed": None if (last_u is None) else data["U"] - last_u - 1,
        }
        self._writer.write(symbol, "gap", rx_ns, json.dumps(record, separators=(",", ":")).encode())

    def _start_resync(self, symbol: str) -> None:
        if (symbol in self._resyncing):
            return
        self._resyncing.add(symbol)
        task = asyncio.create_task(self._resync(symbol))
        self.tasks.add(task)  # 참조를 놓으면 GC가 돌다 만 태스크를 거둬간다
        task.add_done_callback(self.tasks.discard)

    async def _resync(self, symbol: str) -> None:
        """스냅샷으로 호가창을 다시 세운다.

        받는 동안 들어온 메시지는 OrderBook 버퍼에 쌓였다가 apply_snapshot에서 한 번에 적용된다.
        반환 목록에 'gap'이 있으면 스냅샷이 버퍼보다 오래됐다는 뜻이라 다시 받는다.
        """
        try:
            for _ in range(RESYNC_MAX_TRIES):
                raw = await fetch_snapshot(self._session, symbol, SNAPSHOT_LIMIT)
                rx = time.time_ns()
                self._writer.write_snapshot(symbol, rx, raw)
                if ("gap" not in self.books[symbol].apply_snapshot(json.loads(raw))):
                    log.info("resync %s ok, last_u=%s", symbol, self.books[symbol].last_u)
                    return
                await asyncio.sleep(RESYNC_COOLDOWN_S)
            log.warning("resync %s: %d회 실패", symbol, RESYNC_MAX_TRIES)
        except BinanceBanned as exc:
            # 차단 중에는 REST가 어차피 안 된다. 원본 수집은 그동안에도 계속된다
            log.error("resync %s 차단: %s", symbol, exc)
            await asyncio.sleep(min(exc.retry_after_s or RESYNC_COOLDOWN_S, MAX_BACKOFF_S))
        except (BinanceRestError, OSError) as exc:
            log.warning("resync %s 실패: %s", symbol, exc)
            await asyncio.sleep(RESYNC_COOLDOWN_S)
        finally:
            # 플래그를 풀면 다음 델타가 재시도를 띄운다
            self._resyncing.discard(symbol)

async def _pump(ws, handler: StreamHandler) -> None:
    while True:
        frame = await ws.recv(decode=False)  # 바이트로 받아야 재직렬화가 없다
        handler.handle(frame, time.time_ns())  # 받은 직후에 찍어야 수신 시각이 맞다

async def _pump(ws, handler: StreamHandler) -> None:
    while True:
        frame = await ws.recv(decode=False)  # 바이트로 받아야 재직렬화가 없다
        handler.handle(frame, time.time_ns())  # 받은 직후에 찍어야 수신 시각이 맞다

async def _connect_forever(handler: StreamHandler, url: str) -> None:
    attempt = 0
    while True:
        try:
            async with websockets.connect(url) as ws:
                log.info("연결됨")
                attempt = 0  # 연결이 서면 백오프를 되돌린다
                await _pump(ws, handler)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            wait = min(BACKOFF_S * 2**attempt, MAX_BACKOFF_S)
            attempt += 1
            log.warning("연결 끊김 (%s). %.0f초 후 재연결", exc, wait)
            await asyncio.sleep(wait)

async def _flush_forever(writer: RawWriter) -> None:
    while True:
        await asyncio.sleep(FLUSH_S)
        writer.flush()

async def run(symbols, root, ws_base: str = WS_BASE) -> None:
    """수집을 시작한다. 진입점과 환경변수 설정은 main.py 가 맡는다."""
    writer = RawWriter(Path(root))
    async with aiohttp.ClientSession() as session:
        handler = StreamHandler(session, writer, symbols)
        flusher = asyncio.create_task(_flush_forever(writer))
        try:
            await _connect_forever(handler, ws_base + stream_names(symbols))
        finally:
            flusher.cancel()
            for task in list(handler.tasks):
                task.cancel()
            writer.close()