"""테스트용 aiohttp·websockets 흉내. 호출자가 둘이라 파일로 뺐다."""

import asyncio
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

class FakeSocket:
    """websockets 연결 흉내. push() 로 넣은 프레임을 recv() 가 순서대로 돌려주고, 없으면 기다린다.

    프레임 자리에 예외를 넣으면 그 차례에 던진다. 연결이 끊기는 것을 흉내 낸다.
    """

    def __init__(self):
        self._incoming = asyncio.Queue()
        self.closed = False

    def push(self, *frames):
        for item in frames:
            self._incoming.put_nowait(item)

    async def recv(self, decode=None):
        item = await self._incoming.get()
        if (isinstance(item, BaseException)):
            raise item
        return item if (decode is False) else item.decode()  # 진짜도 decode=False 일 때만 바이트를 준다

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.closed = True
        return False

class FakeConnect:
    """websockets.connect 흉내. 등록한 소켓을 순서대로 내준다. 예외를 등록하면 그 연결은 열리지 않는다."""

    def __init__(self, *sockets):
        self._sockets = list(sockets)
        self.calls = 0

    def __call__(self, url, close_timeout=None):
        self.calls += 1
        nxt = self._sockets.pop(0)
        return _Raiser(nxt) if (isinstance(nxt, BaseException)) else nxt