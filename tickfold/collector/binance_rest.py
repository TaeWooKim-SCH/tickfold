"""
바이낸스 현물 REST 호출: 호가창 스냅샷과 exchangeInfo

세션(aiohttp.ClientSession)은 호출자가 만들어 넘긴다.
429/418(가중치 초과, 일시 차단)과 5xx 응답은 Retry-After 또는 지수 백오프로 재시도하고,
그 외 4xx는 즉시 예외를 올린다.
"""

import asyncio
import json

import aiohttp

BASE = "https://api.binance.com"
TIMEOUT = aiohttp.ClientTimeout(total=10)

class BinanceRestError(Exception):
    pass

async def _get_json(session: aiohttp.ClientSession, path: str, params: dict, retries: int = 3):
    last_status = None
    for attempt in range(retries):
        async with session.get(BASE + path, params=params, timeout=TIMEOUT) as res:
            if (res.status == 200):
                return await res.json()
            if (res.status in (429, 418) or res.status >= 500):
                last_status = res.status
                wait = float(res.headers.get("Retry-After", 2**attempt))
                await asyncio.sleep(wait)
                continue
            body = await res.text()
            raise BinanceRestError(f"{path} -> {res.status}: {body[:200]}")
    raise BinanceRestError(f"{path}: {retries}회 재시도 실패 (마지막 상태 {last_status})")

async def fetch_snapshot(session: aiohttp.ClientSession, symbol: str, limit: int = 5000) -> dict:
    """
    GET /api/v3/depth
    반환: {"lastUpdateId": int, "bids": [[p, q], ...], "asks": [...]}
    """
    params = {"symbol": symbol.upper(), "limit": str(limit)}
    return await _get_json(session, "/api/v3/depth", params)

async def fetch_exchange_info(session: aiohttp.ClientSession, symbols: list[str]) -> dict[str, dict]:
    """
    GET /api/v3/exchangeInfo
    반환: {"BTCUSDT": {"tickSize": "...", "stepSize": "..."}, ...}

    값은 거래소가 준 문자열 그대로 둔다 (예: "0.01000000"). 정수화 쪽에서 Decimal로 다룬다.
    """

    # 바이낸스는 symbols 파라미터의 JSON 배열에 공백이 있으면 거부한다 - 여러 종목 받을 땐 symbols param 사용
    params = {"symbols": json.dumps([symbol.upper() for symbol in symbols], separators=(",", ":"))}
    data = await _get_json(session, "/api/v3/exchangeInfo", params)

    out = {}
    for symbol_info in data["symbols"]:
        filters = {filter["filterType"]: filter for filter in symbol_info["filters"]}
        out[symbol_info["symbol"]] = {
            "tickSize": filters["PRICE_FILTER"]["tickSize"],
            "stepSize": filters["LOT_SIZE"]["stepSize"]
        }
    
    return out
