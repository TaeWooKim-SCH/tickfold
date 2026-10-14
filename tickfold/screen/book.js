// 호가창 그림 한 프레임의 막대와 글자를 만든다. 화면(DOM)을 모르는 함수라서 node --test 로 시험한다.
// 글자 형식은 앞서 파이썬이 그리던 그림과 같다. 가격은 소수 2자리와 천 단위 쉼표, 수량은 소수 5자리다

export const LEVELS = 20;
// 매도 파랑, 매수 빨강은 국내 거래소 호가창의 색이다. 초록·빨강 짝은 적록 색각에서 구별이 어려워서 안 쓴다
export const ASK = { name: "매도", color: "#2a78d6" };
export const BID = { name: "매수", color: "#e34948" };

// 한 쪽의 레벨 이름. 매도 1 이 최우선 매도 호가다
export function levelNames(side) {
  return Array.from({ length: LEVELS }, (_, level) => `${side.name} ${level + 1}`);
}

// 세로축의 범주 순서. 범주 축은 아래에서 위로 쌓여서 맨 아래가 매수 20, 맨 위가 매도 20 이다
export function levelOrder() {
  return levelNames(BID).reverse().concat(levelNames(ASK));
}

// 소수 digits 자리로 적고 정수 부분에 천 단위 쉼표를 넣는다
export function groupedText(value, digits) {
  // toLocaleString 은 브라우저 언어에 따라 쉼표와 점이 바뀐다
  const [whole, fraction] = value.toFixed(digits).split(".");
  const grouped = whole.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  return fraction === undefined ? grouped : `${grouped}.${fraction}`;
}

// 막대 하나의 글자. 원본 문자열과 같은 수가 나와야 한다
export function levelText(price, quantity) {
  return `${groupedText(price, 2)}  ${quantity.toFixed(5)}`;
}

// 거래소 시각(밀리초)을 시간대 없는 UTC 문자열로 바꾼다. 그림의 시간 축과 글자가 같은 값을 쓴다
export function utcText(exchangeTime) {
  // 시간대가 붙은 값이나 숫자를 넘기면 plotly 가 브라우저의 시간대로 옮겨 그린다
  return new Date(exchangeTime).toISOString().slice(0, 23);
}

// 그림의 제목. 시각, mid, 스프레드
export function bookTitle(exchangeTime, bestBid, bestAsk) {
  const time = utcText(exchangeTime).replace("T", " ");
  return `${time} UTC · mid ${groupedText((bestBid + bestAsk) / 2, 3)} · 스프레드 ${(bestAsk - bestBid).toFixed(2)}`;
}

// 한 프레임의 막대 둘(매도, 매수). levels 는 쪽마다 { prices, quantities } 이고 프레임마다 20개씩 이어진 배열이다
export function bookTraces(asks, bids, frame, ceiling) {
  return [
    [ASK, asks],
    [BID, bids],
  ].map(([side, levels]) => {
    const prices = levels.prices.subarray(frame * LEVELS, (frame + 1) * LEVELS);
    const quantities = levels.quantities.subarray(frame * LEVELS, (frame + 1) * LEVELS);
    return {
      type: "bar",
      orientation: "h",
      x: Array.from(quantities, (quantity) => Math.min(quantity, ceiling)),
      y: levelNames(side),
      marker: { color: side.color },
      // 잘린 막대도 글자와 마우스를 올렸을 때의 값은 원래 수량이다
      text: Array.from(prices, (price, level) => levelText(price, quantities[level])),
      hoverinfo: "y+text",
      // 축의 절반을 넘는 막대는 오른쪽에 글자 자리가 모자라서 막대 안에 적는다
      textposition: Array.from(quantities, (quantity) => (quantity > ceiling / 2 ? "inside" : "outside")),
      // 그냥 두면 plotly 가 글자를 막대 굵기에 맞춰 줄여 그린다. 40줄이라 막대가 글자보다 가늘다
      constraintext: "none",
    };
  });
}