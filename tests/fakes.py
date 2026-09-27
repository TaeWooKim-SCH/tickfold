"""테스트용 aiohttp 흉내. 호출자가 둘이라 파일로 뺐다."""

import json

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