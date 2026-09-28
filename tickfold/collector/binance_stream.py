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
from tickfold.collector.writer import RawWriter, last_stored_line

log = logging.getLogger(__name__)

WS_BASE = "wss://stream.binance.com:9443/stream?streams="
DEPTH = "depth"
DEPTH20 = "depth20"
STREAMS = (f"{DEPTH}@100ms", "trade", f"{DEPTH20}@100ms")  # 호가창에 적용하는 것은 depth 뿐

# 재동기화용
SNAPSHOT_LIMIT = 5000
MAX_BACKOFF_S = 60.0
RESYNC_COOLDOWN_S = 2.0  # 재동기화 실패 후 다음 시도까지. 가중치를 연속으로 태우지 않으려고
RESYNC_MAX_TRIES = 3
SNAPSHOT_SETTLE_S = 0.5  # 스냅샷은 요청 시점보다 살짝 옛 상태로 온다. 버퍼가 그보다 먼저 시작하도록 기다린다

# 스냅샷 갱신용
REFRESH_BELOW = 0.2  # 범위 끝까지 남은 여유가 이 비율 아래로 줄면 스냅샷을 다시 받는다
REFRESH_MIN_INTERVAL_S = 30.0  # 가격이 출렁일 때 연달아 받아 가중치 한도에 닿지 않게
SNAPSHOT_MAX_AGE_S = 600.0  # 마지막 스냅샷이 이보다 오래되면 범위와 무관하게 다시 받는다
SNAPSHOT_AGE_CHECK_S = 10.0

# 연결루프용
FLUSH_S = 5.0
BACKOFF_S = 1.0
CLOSE_TIMEOUT_S = 1.0  # 닫는 인사를 오래 기다리지 않는다. 기다리는 동안은 아무것도 못 받아 그대로 갭이 된다

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
        self._last_seen_u: dict[str, int] = {}
        self._last_snapshot_at: dict[str, float] = {}

    def resume_sequence(self, root: Path) -> None:
        """직전 실행이 마지막으로 저장한 순번에서 이어 간다. 재시작 사이에 놓친 것이 갭으로 기록되게."""
        for symbol in self.books:
            line = last_stored_line(root, symbol, DEPTH)
            if (line is None):
                continue
            try:
                self._last_seen_u[symbol] = json.loads(line)["m"]["data"]["u"]
            except (ValueError, KeyError, TypeError):
                log.warning("resume %s: 마지막 줄에서 순번을 읽지 못했다", symbol)

    def handle(self, frame: bytes, rx_ns: int) -> None:
        msg = json.loads(frame)
        if ("stream" not in msg):
            # 제어·오류 프레임. 버리지 않고 남긴다. 여기서 예외를 내면 연결이 통째로 끊긴다
            self._writer.write("_control", "raw", rx_ns, frame)
            return
        
        symbol, stream = split_stream(msg["stream"])
        self._writer.write(symbol, stream, rx_ns, frame) # 원본 저장이 항상 먼저

        book = self.books.get(symbol)
        if (stream == DEPTH20 and book is not None):
            self._refresh_if_near_edge(symbol, book, msg["data"])
        
        if (stream != DEPTH or book is None):
            return

        data = msg["data"]
        missed = self._count_missed_updates(symbol, data)
        if (missed):
            self.gaps += 1
            self._write_gap(symbol, rx_ns, data, missed)

        book.apply_delta(data)
        if (not book.synced):
            self._start_resync(symbol)

    def _count_missed_updates(self, symbol: str, data: dict) -> int:
        """스트림에서 빠진 변경 수. 호가창이 동기화됐는지와 무관하게 순번만 본다."""
        last_seen_u = self._last_seen_u.get(symbol)
        if (last_seen_u is None or data["u"] > last_seen_u):
            self._last_seen_u[symbol] = data["u"]
        if (last_seen_u is None or data["U"] <= last_seen_u + 1):
            return 0  # 처음 받은 것, 중복, 겹침은 빠진 게 없다
        return data["U"] - last_seen_u - 1


    def _write_gap(self, symbol: str, rx_ns: int, data: dict, missed: int) -> None:
        # 갭 기록도 같은 writer로 남긴다. 로테이션·압축을 그대로 쓰고 그 갭이 난 데이터 옆에 남는다
        record = {
            "event": "gap",
            "last_u": data["U"] - missed - 1,
            "U": data["U"],
            "u": data["u"],
            "missed": missed,
        }
        self._writer.write(symbol, "gap", rx_ns, json.dumps(record, separators=(",", ":")).encode())

    def _refresh_if_near_edge(self, symbol: str, book: OrderBook, top_levels: dict) -> None:
        # 최우선 호가는 우리 호가창을 훑지 않고 거래소가 준 상위 20레벨의 첫 항목에서 읽는다
        if (not top_levels.get("bids") or not top_levels.get("asks")):
            return  # 비었거나 모양이 다른 프레임. 원본은 이미 저장했으니 여기서는 넘어간다
        best_bid = float(top_levels["bids"][0][0])
        best_ask = float(top_levels["asks"][0][0])
        headroom = book.range_headroom(best_bid, best_ask)
        if (headroom is not None and headroom < REFRESH_BELOW):
            self.refresh(symbol, f"범위 여유 {headroom:.0%}")

    def refresh(self, symbol: str, reason: str) -> bool:
        """스냅샷을 다시 받아 호가창을 새로 세운다. 갭이 아니므로 갭 기록은 남기지 않는다.

        이미 받는 중이거나 최소 간격 안이면 아무것도 하지 않고 False 를 돌려준다.
        """
        last_snapshot_at = self._last_snapshot_at.get(symbol)
        too_soon = last_snapshot_at is not None and time.monotonic() - last_snapshot_at < REFRESH_MIN_INTERVAL_S
        if (symbol in self._resyncing or too_soon):
            return False
        log.info("refresh %s: %s", symbol, reason)
        self.books[symbol].invalidate()
        self._start_resync(symbol)
        return True

    def refresh_stale(self) -> None:
        """마지막 스냅샷이 오래된 종목을 갱신한다. 복원기가 하루 중간부터 재생을 시작할 수 있게."""
        now = time.monotonic()
        for symbol, last_snapshot_at in list(self._last_snapshot_at.items()):
            if (now - last_snapshot_at >= SNAPSHOT_MAX_AGE_S):
                self.refresh(symbol, "주기")

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
                await asyncio.sleep(SNAPSHOT_SETTLE_S)
                raw = await fetch_snapshot(self._session, symbol, SNAPSHOT_LIMIT)
                rx = time.time_ns()
                self._last_snapshot_at[symbol] = time.monotonic()
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

async def _connect_forever(handler: StreamHandler, url: str) -> None:
    attempt = 0
    while True:
        try:
            async with websockets.connect(url, close_timeout=CLOSE_TIMEOUT_S) as ws:
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

async def _refresh_forever(handler: StreamHandler) -> None:
    while True:
        await asyncio.sleep(SNAPSHOT_AGE_CHECK_S)
        handler.refresh_stale()

async def run(symbols, root, ws_base: str = WS_BASE) -> None:
    """수집을 시작한다. 진입점과 환경변수 설정은 main.py 가 맡는다."""
    writer = RawWriter(Path(root))
    async with aiohttp.ClientSession() as session:
        handler = StreamHandler(session, writer, symbols)
        handler.resume_sequence(Path(root))

        flusher = asyncio.create_task(_flush_forever(writer))
        refresher = asyncio.create_task(_refresh_forever(handler))
        try:
            await _connect_forever(handler, ws_base + stream_names(symbols))
        finally:
            flusher.cancel()
            refresher.cancel()
            for task in list(handler.tasks):
                task.cancel()
            writer.close()