"""
웹소켓 연결을 유지한다. 끊기면 백오프로 다시 붙고, 오래되면 새 연결로 갈아탄다.

갈아탈 때는 새 연결을 먼저 열어 옛 연결과 잠깐 같이 돌린다. 파일에 쓰는 것은 언제나 한 연결이고,
새 연결이 그동안 받은 것은 쌓아 뒀다가 넘겨받는 순간에 handler 로 넘긴다.
받은 프레임을 어떻게 다루는지는 모른다. handler 에 넘길 뿐이다.
"""

import asyncio
import logging
import time

import websockets

log = logging.getLogger(__name__)

BACKOFF_S = 1.0
MAX_BACKOFF_S = 60.0
# 닫는 인사를 오래 기다리지 않는다. 갈아탈 때는 옛 연결이 뒤에서 닫히고, 종료할 때는 기다리지 않고 나간다
CLOSE_TIMEOUT_S = 1.0
CONNECTION_CHECK_S = 1.0

# 교대용
ROTATE_AFTER_S = 12 * 3600.0  # 거래소가 끊기 전에 먼저 갈아탄다. 끊길 때는 7~8초 전부터 프레임이 안 온다
OVERLAP_S = 3.0  # 두 연결을 같이 돌리는 최소 시간. 옛 연결이 밀려 있으면 따라올 때까지 더 겹친다
OVERLAP_MAX_S = 30.0  # 이만큼 겹쳐도 넘겨받을 수 없으면 기다리기를 그만둔다. 후임이 받고 있으면 넘기고, 아니면 후임을 버린다
STANDBY_QUIET_S = 1.0  # 후임에 이만큼 프레임이 안 오면 지금은 받고 있지 않은 것으로 본다. depth 는 100ms 마다 온다

def _backoff(attempt: int) -> float:
    """attempt 번 실패한 뒤 기다릴 시간."""
    return min(BACKOFF_S * 2 ** min(attempt, 16), MAX_BACKOFF_S)  # 지수를 안 묶으면 1024번째 실패에서 터진다

class Feed:
    """연결 하나가 받은 프레임이 가는 길.

    대기 연결의 프레임은 쌓아만 둔다. 두 연결이 같이 넘기면 한쪽이 앞선 만큼
    순번이 건너뛴 것처럼 보여 없는 갭이 기록된다.
    """

    def __init__(self, handler, live: bool):
        self._handler = handler
        self._live = live
        self.pending: list[tuple[bytes, int]] = []
        self._last_received_at: float | None = None


    def receive(self, frame: bytes, rx_ns: int) -> None:
        self._last_received_at = time.monotonic()
        if (self._live):
            self._handler.handle(frame, rx_ns)
        else:
            self.pending.append((frame, rx_ns))

    def go_live(self) -> None:
        """쌓아 둔 것을 넘기고, 이후로는 받는 대로 넘긴다. 옛 연결이 이미 준 것은 handler 가 버린다."""
        self._handler.begin_handover()
        for frame, rx_ns in self.pending:
            self._handler.handle(frame, rx_ns)
        self.pending = []
        self._live = True

    def is_receiving(self) -> bool:
        """방금까지 프레임이 왔는가. 열리기만 했거나 받다가 멈춘 연결을 가린다."""
        return self._last_received_at is not None and time.monotonic() - self._last_received_at < STANDBY_QUIET_S

class Connection:
    """웹소켓 연결 하나와 그것을 읽는 태스크. 태스크는 스스로 끝나지 않고, 끊기면 예외로 끝난다."""

    def __init__(self, url: str, feed: Feed, receiving: set):
        self.feed = feed
        self.opened_at: float | None = None
        self.task = asyncio.create_task(self._receive(url))
        receiving.add(self.task)  # 참조를 놓으면 GC가 닫는 중인 태스크를 거둬간다
        self.task.add_done_callback(receiving.discard)

    async def _receive(self, url: str) -> None:
        async with websockets.connect(url, close_timeout=CLOSE_TIMEOUT_S) as ws:
            self.opened_at = time.monotonic()
            log.info("연결됨")
            while True:
                frame = await ws.recv(decode=False)  # 바이트로 받아야 재직렬화가 없다
                self.feed.receive(frame, time.time_ns())  # 받은 직후에 찍어야 수신 시각이 맞다

    def age(self) -> float:
        return 0.0 if (self.opened_at is None) else time.monotonic() - self.opened_at

    def close(self) -> None:
        self.task.cancel()

    def lost_because(self) -> BaseException:
        """끝난 수신 태스크가 왜 끝났는지. 취소로 끝났어도 끊김으로 돌려준다."""
        # 취소를 그대로 올리면 감독이 자기가 취소된 줄 알고 수집기가 조용히 끝난다
        if (self.task.cancelled()):
            return ConnectionError("수신 태스크가 취소됐다")
        return self.task.exception() or ConnectionError("수신 태스크가 예외 없이 끝났다")

async def _sleep_unless_lost(connection: Connection, seconds: float) -> None:
    """seconds 만큼 기다린다. 그 사이 연결이 끊기면 끊긴 이유를 그대로 올린다."""
    done, _ = await asyncio.wait({connection.task}, timeout=seconds)
    if (done):
        raise connection.lost_because()

async def _overlap(handler, primary: Connection, standby: Connection) -> str | None:
    """후임이 넘겨받을 수 있을 때까지 두 연결을 같이 돌린다.

    넘겨받아도 되면 None 을, 후임을 쓸 수 없게 되면 그 이유를 돌려준다.
    """
    while True:
        done, _ = await asyncio.wait(
            {primary.task, standby.task}, timeout=CONNECTION_CHECK_S, return_when=asyncio.FIRST_COMPLETED
        )
        if (standby.task in done):
            return str(standby.lost_because())
        if (primary.task in done):
            log.warning("교대 중에 옛 연결이 끊겼다 (%s). 새 연결로 잇는다", primary.lost_because())
            return None
        receiving = standby.feed.is_receiving()
        if (standby.age() >= OVERLAP_S and receiving and handler.can_hand_over(standby.feed.pending)):
            return None
        if (standby.age() >= OVERLAP_MAX_S):
            if (not receiving):
                standby.close()  # 열리기만 했거나 받다가 멈춘 연결로는 넘어가지 않는다
                return f"{OVERLAP_MAX_S:.0f}초가 지났는데 프레임이 오지 않고 있다"
            log.warning("옛 연결이 %.0f초 동안 따라오지 못했다. 그대로 넘긴다", OVERLAP_MAX_S)
            return None

async def _rotate_when_due(handler, url: str, primary: Connection, receiving: set) -> Connection:
    """갈아탈 때까지 기다렸다가 후임을 세워 넘기고, 후임을 돌려준다.

    주 연결이 끊기면 그 예외가 올라간다. 후임과 겹치는 중에 끊기면 후임이 바로 넘겨받는다.
    후임이 실패하면 주 연결로 계속 받으면서 다시 시도한다.
    """
    attempt = 0
    while True:
        while (primary.age() < ROTATE_AFTER_S and not handler.shutdown_noticed):
            await _sleep_unless_lost(primary, CONNECTION_CHECK_S)
        standby = Connection(url, Feed(handler, live=False), receiving)
        failure = await _overlap(handler, primary, standby)
        if (failure is None):
            break
        wait = _backoff(attempt)
        attempt += 1
        log.warning("후임 연결 실패 (%s). 옛 연결로 계속 받고 %.0f초 후 다시 시도", failure, wait)
        await _sleep_unless_lost(primary, wait)

    # 여기부터 go_live 까지는 await 가 없어야 한다. 사이에 프레임이 끼어들면 순서가 엉킨다
    primary.close()
    handler.shutdown_noticed = False
    stacked = len(standby.feed.pending)
    dropped_before = handler.duplicates_dropped
    standby.feed.go_live()
    dropped = handler.duplicates_dropped - dropped_before
    log.info("교대: 겹친 %.1f초 동안 쌓인 %d개 중 %d개는 옛 연결이 이미 줬다", standby.age(), stacked, dropped)
    return standby

async def connect_forever(handler, url: str) -> None:
    """연결을 유지한다. 끊기면 백오프로 다시 붙고, 오래되면 새 연결로 갈아탄다."""
    receiving: set[asyncio.Task] = set()  # 평소 하나, 교대 중에는 둘, 옛 연결이 닫히는 동안은 그것까지
    primary = None
    attempt = 0
    try:
        while True:
            if (primary is None):
                handler.shutdown_noticed = False  # 알림은 그것을 받은 연결의 것이다. 그 연결은 이제 없다
                primary = Connection(url, Feed(handler, live=True), receiving)
            try:
                primary = await _rotate_when_due(handler, url, primary, receiving)
                attempt = 0  # 넘겨받았으면 연결이 섰던 것이다
            except Exception as exc:
                for task in list(receiving):
                    if (not task.cancelling()):  # 닫는 중인 연결을 또 취소하면 닫기 대기가 끊겨 소켓이 남는다
                        task.cancel()
                if (primary.opened_at is not None):
                    attempt = 0  # 연결이 섰다가 끊긴 것이면 백오프를 처음부터
                primary = None
                wait = _backoff(attempt)
                attempt += 1
                log.warning("연결 끊김 (%s). %.0f초 후 재연결", exc, wait)
                await asyncio.sleep(wait)
    finally:
        # 닫기 응답을 기다리지 않는다. 기다리는 1초가 그대로 재시작 갭이 된다
        for task in list(receiving):
            task.cancel()