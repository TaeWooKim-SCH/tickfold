import assert from "node:assert/strict";
import { test } from "node:test";

import { bookTitle, bookTraces, levelOrder, levelText } from "../tickfold/screen/book.js";

test("level text keeps the digits of the raw strings with thousands separators on the price", () => {
  // 원본 문자열에서 읽은 float64 다. 자릿수가 모자라 반올림되거나 쉼표가 빠지면 여기서 걸린다
  const raw = [
    ["82512.10000000", "0.00001000", "82,512.10  0.00001"],
    ["999.99000000", "23.71001000", "999.99  23.71001"],
    ["1000.00000000", "301.90000000", "1,000.00  301.90000"],
    ["1234567.89000000", "0.12345000", "1,234,567.89  0.12345"],
  ];
  for (const [price, quantity, shown] of raw) assert.equal(levelText(Number(price), Number(quantity)), shown);
});

test("title shows the UTC time of the frame, the mid to three decimals and the spread", () => {
  const time = Date.UTC(2026, 9, 7, 7, 32, 0, 14);

  assert.equal(bookTitle(time, 62145.37, 62145.38), "2026-10-07 07:32:00.014 UTC · mid 62,145.375 · 스프레드 0.01");
  assert.equal(bookTitle(time, 82704.9, 82794.01), "2026-10-07 07:32:00.014 UTC · mid 82,749.455 · 스프레드 89.11");
});

test("bars longer than the ceiling are cut while their text keeps the quantity", () => {
  // 두 프레임, 쪽마다 20레벨. 둘째 프레임만 본다
  const levels = (base, step) => Float64Array.from({ length: 40 }, (_, index) => base + step * index);
  const asks = { prices: levels(100, 0.01), quantities: levels(0.5, 0.25) };
  const bids = { prices: levels(99, -0.01), quantities: levels(-1, 0.25) };

  const [askBars, bidBars] = bookTraces(asks, bids, 1, 8);

  assert.deepEqual(askBars.y, Array.from({ length: 20 }, (_, level) => `매도 ${level + 1}`));
  assert.deepEqual(bidBars.y, Array.from({ length: 20 }, (_, level) => `매수 ${level + 1}`));
  assert.deepEqual([askBars.marker.color, bidBars.marker.color], ["#2a78d6", "#e34948"]);
  // 둘째 프레임의 매도 수량은 5.5 부터 0.25 씩 는다. 8 을 넘는 막대는 8 에서 잘리고 글자는 원래 수량이다
  assert.deepEqual(askBars.x.slice(9, 12), [7.75, 8, 8]);
  assert.equal(askBars.text[11], "100.31  8.25000");
  // 축의 절반(4)을 넘는 막대만 글자를 막대 안에 적는다. 둘째 프레임의 매수 1 이 꼭 4 다
  assert.deepEqual(bidBars.textposition.slice(0, 2), ["outside", "inside"]);
  assert.equal(bidBars.text[0], "98.80  4.00000");
});

test("levels stack from bid 20 at the bottom to ask 20 at the top", () => {
  const order = levelOrder();

  assert.deepEqual([order[0], order[19], order[20], order[39]], ["매수 20", "매수 1", "매도 1", "매도 20"]);
  assert.equal(order.length, 40);
});