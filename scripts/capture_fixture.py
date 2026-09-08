"""
바이낸스 @depth 델타 스트림을 60초간 받아 테스트 픽스쳐로 저장한다.

사용법 (프로젝트 루트에서):
    uv run python scripts/capture_fixture.py

생성 파일:
    tests/fixtures/btcusdt_depth.ndjson 각 줄 = {"rx": 수신시각ns, "m": 원본 메세지}
    tests/fixtures/btcusdt_snapshot.json 연결 직후 받은 REST 스냅샷
"""

import asyncio
import time
import urllib.request
from pathlib import Path

import websockets

URL = "wss://stream.binance.com:9443/ws/btcusdt@depth@100ms"
SNAP = "https://api.binance.com/api/v3/depth?symbol=BTCUSDT&limit=1000"
OUT = Path("tests/fixtures")

async def main(seconds=60):
    OUT.mkdir(parents=True, exist_ok=True)

    with open(OUT / "btcusdt_depth.ndjson", "wb") as f:
        async with websockets.connect(URL) as ws:
            # 웹소켓을 먼저 열고 나서 스냅샷을 받는다 (동기화 절차와 같은 순서)
            snap = urllib.request.urlopen(SNAP).read()
            with open(OUT / "btcusdt_snapshot.json", "wb") as s:
                s.write(snap)
            
            end = time.time() + seconds
            while (time.time() < end):
                raw = await ws.recv()
                raw = raw.encode() if (isinstance(raw, str)) else raw

                # 원본은 파싱하지 않고 바이트 그대로 붙임
                f.write(b'{"rx":%d, "m":' % time.time_ns() + raw + b'}\n')

if (__name__ == "__main__"):
    asyncio.run(main())