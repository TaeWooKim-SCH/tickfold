import asyncio
import os
import signal

from tickfold.collector import main as main_mod

def test_settings_reads_symbols_and_defaults():
    cfg = main_mod.settings({"TICKFOLD_SYMBOLS": "btcusdt, ethusdt ,"})
    assert cfg["symbols"] == ["BTCUSDT", "ETHUSDT"]
    assert main_mod.settings({}) == {"symbols": ["BTCUSDT"], "root": "data/raw", "log_level": "INFO"}

async def test_sigterm_cancels_run_so_cleanup_happens(monkeypatch):
    closed = []

    async def fake_run(_symbols, _root):
        try:
            await asyncio.sleep(3600)
        finally:
            closed.append(True)  # 진짜 run() 은 여기서 writer.close() 를 부른다

    monkeypatch.setattr(main_mod, "run", fake_run)
    asyncio.get_running_loop().call_later(0.05, os.kill, os.getpid(), signal.SIGTERM)
    await main_mod._main({"symbols": ["BTCUSDT"], "root": "x", "log_level": "INFO"})

    assert closed == [True]