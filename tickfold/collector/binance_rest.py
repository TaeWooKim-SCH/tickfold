"""
바이낸스 현물 REST 호출: 반환은 원본 바이트다. 원본을 재직렬화 없이 저장해야 하므로 파싱은 호출자가 한다.

세션(aiohttp.ClientSession)은 호출자가 만들어 넘긴다.
429/418(가중치 초과, 일시 차단)과 5xx 응답은 Retry-After 또는 지수 백오프로 재시도하고,
그 외 4xx는 즉시 예외를 올린다.
"""

import asyncio
import json

import aiohttp

BASE = "https://api.binance.com"
TIMEOUT = aiohttp.ClientTimeout(total=10)
BACKOFF_S = 1.0
MAX_WAIT_S = 60.0

class BinanceRestError(Exception):
    pass

class BinanceBanned(BinanceRestError):
    """418. IP가 차단됐다. 호출자는 원본 수집을 계속하고 스냅샷만 나중에 받는다."""
    def __init__(self, path: str, retry_after_s: float | None):
        self.retry_after_s = retry_after_s
        super().__init__(f"{path} -> 418 IP 차단, 해제까지 {retry_after_s}초")

async def _get(session: aiohttp.ClientSession, path: str, params: dict, retries: int = 3) -> bytes:
    """응답 본문을 바이트 그대로 돌려줌. json 파싱은 호출자에서 진행"""
    last_status = None
    for attempt in range(retries):
        backoff = BACKOFF_S * 2**attempt
        try:
            async with session.get(BASE + path, params=params, timeout=TIMEOUT) as res:
                if (res.status == 200):
                    return await res.read()
                if (res.status == 418):
                    raise BinanceBanned(path, _retry_after(res.headers, None))
                if (res.status == 429 or res.status >= 500):
                    last_status = f"마지막 상태 {res.status}"
                    wait = min(_retry_after(res.headers, backoff), MAX_WAIT_S)
                else:
                    body = await res.text()
                    raise BinanceRestError(f"{path} -> {res.status}: {body[:200]}")
        except (aiohttp.ClientError, TimeoutError) as exc:
            last_status = f"{type(exc).__name__}: {exc}"
            wait = backoff

        if (attempt + 1 < retries):
            await asyncio.sleep(wait) # 응답 컨텍스트 밖에서 대기

    raise BinanceRestError(f"{path}: {retries}회 재시도 실패 ({last_status})")

def _retry_after(headers, default):
    # 바이낸스는 초 단위 정수를 보낸다. HTTP 날짜 형식 등 다른 값이 오면 백오프로 떨어진다
    raw = headers.get("Retry-After")
    if (raw is None):
        return default
    try:
        return float(raw)
    except ValueError:
        return default

async def fetch_snapshot(session: aiohttp.ClientSession, symbol: str, limit: int = 5000) -> bytes:
    """
    GET /api/v3/depth

    반환은 거래소가 보낸 바이트 그대로다. RawWriter.write_snapshot에 그대로 넘기고,
    호가창에 적용할 때만 json.loads로 푼다. 저장본과 해석이 같은 바이트에서 나온다.
    본문: {"lastUpdateId": int, "bids": [[p, q], ...], "asks": [...]}

    가중치는 limit 100까지 5, 1000까지 50, 5000까지 250이고 IP당 분당 6000이다.
    """
    params = {"symbol": symbol.upper(), "limit": str(limit)}
    return await _get(session, "/api/v3/depth", params)

async def fetch_exchange_info(session: aiohttp.ClientSession, symbols: list[str]) -> dict[str, dict]:
    """
    GET /api/v3/exchangeInfo
    반환: {"BTCUSDT": {"tickSize": "...", "stepSize": "..."}, ...}

    값은 거래소가 준 문자열 그대로 둔다 (예: "0.01000000"). 정수화 쪽에서 Decimal로 다룬다.
    """

    # 바이낸스는 symbols 파라미터의 JSON 배열에 공백이 있으면 거부한다 - 여러 종목 받을 땐 symbols param 사용
    params = {"symbols": json.dumps([symbol.upper() for symbol in symbols], separators=(",", ":"))}
    data = json.loads(await _get(session, "/api/v3/exchangeInfo", params))

    out = {}
    for symbol_info in data["symbols"]:
        filters = {filter["filterType"]: filter for filter in symbol_info["filters"]}
        out[symbol_info["symbol"]] = {
            "tickSize": filters["PRICE_FILTER"]["tickSize"],
            "stepSize": filters["LOT_SIZE"]["stepSize"]
        }
    
    return out
