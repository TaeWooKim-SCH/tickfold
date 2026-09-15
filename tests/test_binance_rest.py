import json

import pytest

from tickfold.collector import binance_rest
from tickfold.collector.binance_rest import (
    BinanceBanned,
    BinanceRestError,
    fetch_exchange_info,
    fetch_snapshot
)

SNAP = {
    "lastUpdateId": 100,
    "bids": [["50000.00", "2.00000000"]],
    "asks": [["50001.00", "1.50000000"]],
}

EXCHANGE_INFO = {
    "symbols": [
        {
            "symbol": "BTCUSDT",
            "filters": [
                {"filterType": "PRICE_FILTER", "tickSize": "0.01000000", "minPrice": "0.01"},
                {"filterType": "LOT_SIZE", "stepSize": "0.00001000", "minQty": "0.00001"},
                {"filterType": "MIN_NOTIONAL"},
            ],
        }
    ]
}

class FakeResponse:
    """aiohttp 응답 흉내: status, headers, read(), text(), async with 지원."""
 
    def __init__(self, status, payload=None, raw=None, body="", headers=None):
        self.status = status
        self.headers = headers or {}
        self._body = body
        if (raw is None and payload is not None):
            raw = json.dumps(payload, separators=(",", ":")).encode()
        self._raw = raw or b""
 
    async def read(self):
        return self._raw
 
    async def text(self):
        return self._body
 
    async def __aenter__(self):
        return self
 
    async def __aexit__(self, *exc):
        return False

class _Raiser:
    """__aenter__에서 예외를 던진다. 진짜 aiohttp도 요청을 __aenter__에서 보내므로
    타임아웃이 get() 이 아니라 거기서 뜬다."""

    def __init__(self, exc):
        self._exc = exc

    async def __aenter__(self):
        raise self._exc

    async def __aexit__(self, *exc):
        return False

class FakeSession:
    """session.get()이 등록된 응답을 순서대로 돌려주고 호출 내역을 기록한다.

    응답 자리에 예외를 넣으면 그 요청에서 던진다.
    """
 
    def __init__(self, *responses):
        self._responses = list(responses)
        self.calls = []
 
    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        nxt = self._responses.pop(0)
        return _Raiser(nxt) if (isinstance(nxt, BaseException)) else nxt

async def test_fetch_snapshot_returns_raw_bytes_and_uppercases_symbol():
    s = FakeSession(FakeResponse(200, payload=SNAP))
    raw = await fetch_snapshot(s, "btcusdt", limit=1000)
    assert json.loads(raw) == SNAP
    url, params = s.calls[0]
    assert url.endswith("/api/v3/depth")
    assert params == {"symbol": "BTCUSDT", "limit": "1000"}

async def test_retries_after_429_then_succeeds():
    s = FakeSession(
        FakeResponse(429, headers={"Retry-After": "0"}),
        FakeResponse(200, payload=SNAP),
    )
    assert json.loads(await fetch_snapshot(s, "BTCUSDT")) == SNAP
    assert len(s.calls) == 2


async def test_gives_up_after_retries_exhausted():
    s = FakeSession(*[FakeResponse(500, headers={"Retry-After": "0"}) for _ in range(3)])
    with pytest.raises(BinanceRestError):
        await fetch_snapshot(s, "BTCUSDT")
    assert len(s.calls) == 3

async def test_client_error_is_not_retried():
    s = FakeSession(FakeResponse(400, body='{"code":-1121,"msg":"Invalid symbol."}'))
    with pytest.raises(BinanceRestError):
        await fetch_snapshot(s, "NOPE")
    assert len(s.calls) == 1

async def test_fetch_exchange_info_extracts_tick_and_step():
    s = FakeSession(FakeResponse(200, payload=EXCHANGE_INFO))
    got = await fetch_exchange_info(s, ["btcusdt"])
    assert got == {"BTCUSDT": {"tickSize": "0.01000000", "stepSize": "0.00001000"}}
    _, params = s.calls[0]
    assert params == {"symbols": '["BTCUSDT"]'}  # 공백 없는 JSON 배열

async def test_snapshot_bytes_are_exchange_bytes_verbatim():
    raw = b'{"lastUpdateId":100,"bids":[["50000.00","2.00000000"]],"asks":[["50001.00","1.50000000"]]}'
    s = FakeSession(FakeResponse(200, raw=raw))
    assert await fetch_snapshot(s, "BTCUSDT") == raw

async def test_timeout_is_retried_then_succeeds(monkeypatch):
    monkeypatch.setattr(binance_rest, "BACKOFF_S", 0.0)
    s = FakeSession(TimeoutError(), FakeResponse(200, payload=SNAP))
    assert json.loads(await fetch_snapshot(s, "BTCUSDT")) == SNAP
    assert len(s.calls) == 2

async def test_ban_418_raises_immediately_with_wait_seconds():
    s = FakeSession(FakeResponse(418, headers={"Retry-After": "120"}))
    with pytest.raises(BinanceBanned) as exc:
        await fetch_snapshot(s, "BTCUSDT")
    assert exc.value.retry_after_s == 120
    assert len(s.calls) == 1