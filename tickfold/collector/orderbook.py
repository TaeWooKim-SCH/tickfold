from decimal import Decimal

class OrderBook:
    """
    바이낸스 현물 @depth 델타 스트림을 검증하며 로컬 호가창을 유지한다

    apply_delta 반환값:
        "pending" - 아직 스냅샷이 없어 버퍼에 보관
        "dup" - 이미 처리한 순번 (u <= last_u), 버림
        "ok" - 정상 적용
        "gap" - 순번이 끊김. 호가창 무효화, 새 스냅샷 필요
    """

    def __init__(self):
        self.bids: dict[str, str] = {} # 가격 문자열 -> 수량 문자열
        self.asks: dict[str, str] = {}
        self.last_u: int | None = None # None이면 동기화 안됨
        self.awaiting_first = False # 스냅샷 직후 첫 이벤트 대기중
        self.buffer: list[dict] = [] # 동기화 전에 받은 메세지

    @property
    def synced(self) -> bool:
        return self.last_u is not None

    def apply_snapshot(self, snap: dict) -> list[str]:
        """
        REST 스냅샷으로 호가창을 채우고 버퍼에 있던 메세지를 순서대로 적용한다.
        반환값은 버퍼 메세지 각각의 apply_delta 결과 목록.
        bids/asks 채우고 last_u = lastUpdateId
        """
        self.bids = {p: q for p, q in snap["bids"]}
        self.asks = {p: q for p, q in snap["asks"]}
        self.last_u = snap["lastUpdateId"]
        self.awaiting_first = True

        pending, self.buffer = self.buffer, []
        return [self.apply_delta(m) for m in pending]

    def apply_delta(self, msg: dict) -> str: # "pending" | "dup" | "ok" | "gap"
        U, u = msg["U"], msg["u"]

        if (self.last_u is None):
            self.buffer.append(msg)
            return "pending"

        if (u <= self.last_u):
            return "dup"

        if (self.awaiting_first):
            # 스냅샷 직후 첫 이벤트: U <= lastUpdateId + 1 <= u 이어야 한다
            if (not (U <= self.last_u + 1 <= u)):
                return self._gap(msg)
            self.awaiting_first = False
        elif (U != self.last_u + 1):
            return self._gap(msg)

        self._apply_levels(self.bids, msg.get("b", [])) # b는 매수 쪽에서 바뀐 레벨
        self._apply_levels(self.asks, msg.get("a", [])) # a는 매도 쪽에서 바뀐 레벨
        self.last_u = u
        return "ok"

    def _gap(self, msg: dict) -> str:
        # 호가창을 무효화하고 이 메세지부터 다시 버퍼링한다
        # 호출자는 새 스냅샷을 받아 apply_snapshot을 호출해야 한다
        self.last_u = None
        self.awaiting_first = False
        self.buffer = [msg]
        return "gap"

    @staticmethod
    def _apply_levels(side: dict[str, str], levels: list) -> None:
        for price, qty in levels:
            if (Decimal(qty) == 0):
                side.pop(price, None)
            else:
                side[price] = qty