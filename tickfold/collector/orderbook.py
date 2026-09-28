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
        self.lowest_known_bid: float | None = None # 스냅샷이 아는 가장 낮은 매수가. 모르면 None
        self.highest_known_ask: float | None = None # 스냅샷이 아는 가장 높은 매도가
        self._bid_span = 0.0 # 스냅샷 시점의 최우선 매수가에서 가장 낮은 매수가까지의 폭
        self._ask_span = 0.0 # 최우선 매도가에서 가장 높은 매도가까지의 폭

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
        self._remember_range(snap)

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

    def range_headroom(self, best_bid: float, best_ask: float) -> float | None:
        """범위 끝까지 남은 여유를 스냅샷 때 폭에 대한 비율로 돌려준다. 양쪽 중 작은 쪽.

        1.0 이면 스냅샷 직후와 같고, 0 이하면 가격이 아는 범위를 벗어났다.
        동기화 전이거나 범위를 모르면 None.
        """
        if (not self.synced or self.lowest_known_bid is None):
            return None
        bid_headroom = (best_bid - self.lowest_known_bid) / self._bid_span
        ask_headroom = (self.highest_known_ask - best_ask) / self._ask_span
        return min(bid_headroom, ask_headroom)

    def _remember_range(self, snap: dict) -> None:
        # 스냅샷은 5000레벨까지만 준다. 그 밖에서 쉬고 있던 주문은 바뀌기 전까지 델타에 안 나오므로
        # 호가창을 믿을 수 있는 것은 이 범위 안뿐이다
        self.lowest_known_bid = self.highest_known_ask = None
        if (not snap["bids"] or not snap["asks"]):
            return
        bid_prices = [float(price) for price, _ in snap["bids"]]
        ask_prices = [float(price) for price, _ in snap["asks"]]
        bid_span = max(bid_prices) - min(bid_prices)
        ask_span = max(ask_prices) - min(ask_prices)
        if (bid_span <= 0 or ask_span <= 0):
            return
        self.lowest_known_bid, self.highest_known_ask = min(bid_prices), max(ask_prices)
        self._bid_span, self._ask_span = bid_span, ask_span

    def invalidate(self) -> None:
        """호가창을 일부러 무효로 만든다. 새 스냅샷으로 다시 세우려는 것이고 갭이 아니다.

        이후 델타는 버퍼에 쌓였다가 apply_snapshot 에서 적용된다.
        """
        self.last_u = None
        self.awaiting_first = False
        self.buffer = []

    def _gap(self, msg: dict) -> str:
        # 순번이 끊긴 이 메세지부터 다시 버퍼링한다
        # 호출자는 새 스냅샷을 받아 apply_snapshot을 호출해야 한다
        self.invalidate()
        self.buffer.append(msg)
        return "gap"

    @staticmethod
    def _apply_levels(side: dict[str, str], levels: list) -> None:
        for price, qty in levels:
            if (Decimal(qty) == 0):
                side.pop(price, None)
            else:
                side[price] = qty