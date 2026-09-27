"""
수집기 진입점. 환경변수로 설정을 받고, 종료 신호를 받으면 파일을 닫고 나간다.

환경변수:
    TICKFOLD_SYMBOLS     쉼표로 구분한 종목 목록. 기본 BTCUSDT
    TICKFOLD_DATA_ROOT   원본 저장 경로. 기본 data/raw
    TICKFOLD_LOG_LEVEL   로그 수준. 기본 INFO
"""

import asyncio
import logging
import os
import signal

from tickfold.collector.binance_stream import run

log = logging.getLogger("tickfold.collector")

def settings(env=os.environ) -> dict:
    """환경변수를 읽는다. 테스트가 dict 를 넣을 수 있게 인자로 받는다."""
    symbols = [s.strip().upper() for s in env.get("TICKFOLD_SYMBOLS", "BTCUSDT").split(",") if s.strip()]
    return {
        "symbols": symbols,
        "root": env.get("TICKFOLD_DATA_ROOT", "data/raw"),
        "log_level": env.get("TICKFOLD_LOG_LEVEL", "INFO").upper(),
    }

async def _main(cfg: dict) -> None:
    task = asyncio.create_task(run(cfg["symbols"], cfg["root"]))
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        # 파이썬 기본 SIGTERM 은 finally 를 건너뛰고 죽는다. 취소로 바꿔야 run() 이 파일을 닫는다
        loop.add_signal_handler(sig, task.cancel)
    try:
        await task
    except asyncio.CancelledError:
        log.info("종료 신호를 받아 파일을 닫고 나간다")

def main() -> None:
    cfg = settings()
    logging.basicConfig(level=cfg["log_level"], format="%(asctime)s %(levelname)s %(name)s %(message)s")
    log.info("수집 시작: %s -> %s", cfg["symbols"], cfg["root"])
    asyncio.run(_main(cfg))

if (__name__ == "__main__"):
    main()