"""
바이낸스 combined stream 을 60초간 받아 테스트 픽스쳐로 저장한다.

사용법 (프로젝트 루트에서):
    uv run python scripts/capture_fixture.py

생성 파일:
    tests/fixtures/btcusdt_combined.ndjson  각 줄 = {"rx": 수신시각ns, "m": 감싼 프레임}
    tests/fixtures/btcusdt_snapshot.json    연결 직후 받은 REST 스냅샷

depth, trade, depth20 셋을 다 받아둔다. 상위 20레벨 대조 PR 이 이 픽스처를 그대로 쓴다.
"""

import asyncio
import time
import urllib.request
from pathlib import Path

import websockets

from tickfold.collector.binance_stream import WS_BASE, stream_names

URL = WS_BASE + stream_names(["BTCUSDT"])
SNAP = "https://api.binance.com/api/v3/depth?symbol=BTCUSDT&limit=1000"
OUT = Path("tests/fixtures")

async def main(seconds=60, snap_after=2.0):
    OUT.mkdir(parents=True, exist_ok=True)

    with open(OUT / "btcusdt_combined.ndjson", "wb") as f:
        async with websockets.connect(URL) as ws:
            start = time.time()
            snapped = False
            while (time.time() < start + seconds):
                raw = await ws.recv(decode=False)  # 바이트로 받아야 재직렬화가 없다
                # writer.py 와 같은 줄 형식. 공통 인코더로 빼는 것은 이슈 7
                f.write(b'{"rx":%d,"m":' % time.time_ns() + raw + b"}\n")

                # 스냅샷은 스트림을 몇 초 받은 뒤에 받는다. REST 는 요청 시점보다 살짝 옛 상태를 줘서,
                # 연결 직후에 받으면 첫 델타보다 오래된 스냅샷이 될 수 있다 (공식 절차 4단계)
                if (not snapped and time.time() > start + snap_after):
                    snap = await asyncio.to_thread(lambda: urllib.request.urlopen(SNAP).read())
                    (OUT / "btcusdt_snapshot.json").write_bytes(snap)
                    snapped = True

if (__name__ == "__main__"):
    asyncio.run(main())