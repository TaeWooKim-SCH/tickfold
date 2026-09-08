import pytest

from tickfold.collector.binance_rest import (
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
    """aiohttp 응답 흉내: status, headers, json(), text(), async with 지원."""
 
    def __init__(self, status, payload=None, body="", headers=None):
        self.status = status
        self.headers = headers or {}
        self._payload = payload
        self._body = body
 
    async def json(self):
        return self._payload
 
    async def text(self):
        return self._body
 
    async def __aenter__(self):
        return self
 
    async def __aexit__(self, *exc):
        return False

class FakeSession:
    """session.get()이 등록된 응답을 순서대로 돌려주고 호출 내역을 기록한다."""
 
    def __init__(self, *responses):
        self._responses = list(responses)
        self.calls = []
 
    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        return self._responses.pop(0)

async def test_fetch_snapshot_returns_parsed_json_and_uppercases_symbol():
    s = FakeSession(FakeResponse(200, payload=SNAP))
    got = await fetch_snapshot(s, "btcusdt", limit=1000)
    assert got == SNAP
    url, params = s.calls[0]
    assert url.endswith("/api/v3/depth")
    assert params == {"symbol": "BTCUSDT", "limit": "1000"}

async def test_retries_after_429_then_succeeds():
    s = FakeSession(
        FakeResponse(429, headers={"Retry-After": "0"}),
        FakeResponse(200, payload=SNAP),
    )
    assert await fetch_snapshot(s, "BTCUSDT") == SNAP
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