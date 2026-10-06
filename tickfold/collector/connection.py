"""
웹소켓 연결을 유지한다. 끊기면 백오프로 다시 붙고, 오래되면 새 연결로 갈아탄다.

갈아탈 때는 새 연결을 먼저 열어 옛 연결과 잠깐 같이 돌린다. 파일에 쓰는 것은 언제나 한 연결이고,
새 연결이 그동안 받은 것은 쌓아 뒀다가 넘겨받는 순간에 handler 로 넘긴다.
받은 프레임을 어떻게 다루는지는 모른다. handler 에 넘길 뿐이다.
"""

import asyncio
import logging
import time

import websockets

log = logging.getLogger(__name__)

BACKOFF_S = 1.0
MAX_BACKOFF_S = 60.0
CLOSE_TIMEOUT_S = 1.0  # 닫는 인사를 오래 기다리지 않는다. 기다리는 동안은 아무것도 못 받아 그대로 갭이 된다

class Feed:
    """연결 하나가 받은 프레임이 가는 길.

    대기 연결의 프레임은 쌓아만 둔다. 두 연결이 같이 넘기면 한쪽이 앞선 만큼
    순번이 건너뛴 것처럼 보여 없는 갭이 기록된다.
    """

    def __init__(self, handler, live: bool):
        self._handler = handler
        self._live = live
        self.pending: list[tuple[bytes, int]] = []

    def receive(self, frame: bytes, rx_ns: int) -> None:
        if (self._live):
            self._handler.handle(frame, rx_ns)
        else:
            self.pending.append((frame, rx_ns))

    def go_live(self) -> None:
        """쌓아 둔 것을 넘기고, 이후로는 받는 대로 넘긴다. 옛 연결이 이미 준 것은 handler 가 버린다."""
        self._handler.begin_handover()
        for frame, rx_ns in self.pending:
            self._handler.handle(frame, rx_ns)
        self.pending = []
        self._live = True

async def _pump(ws, handler) -> None:
    while True:
        frame = await ws.recv(decode=False)  # 바이트로 받아야 재직렬화가 없다
        handler.handle(frame, time.time_ns())  # 받은 직후에 찍어야 수신 시각이 맞다

async def connect_forever(handler, url: str) -> None:
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