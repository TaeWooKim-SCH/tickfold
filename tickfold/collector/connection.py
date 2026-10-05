"""
웹소켓 연결을 유지한다. 끊기면 백오프로 다시 붙는다.

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