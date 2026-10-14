import assert from "node:assert/strict";
import { test } from "node:test";

import { Playback, clockAfter, frameAt, gapAt, gapNotice } from "../tickfold/screen/playback.js";

test("frame follows the exchange time at any speed and stays on the last frame past the end", () => {
  // 프레임 간격이 고르지 않다. 프레임 수를 세어 넘기면 0.45초 뒤에 4번이 아니라 다른 프레임이 나온다
  const exchangeTimes = [10_000, 10_100, 10_200, 10_300, 10_450, 10_500];
  const frameAfter = (startFrame, elapsedMilliseconds, speed) =>
    frameAt(exchangeTimes, clockAfter(exchangeTimes[startFrame], elapsedMilliseconds, speed));

  assert.deepEqual([0, 99, 100, 250, 449, 450].map((elapsed) => frameAfter(0, elapsed, 1)), [0, 0, 1, 2, 3, 4]);
  assert.deepEqual([9.9, 10, 25, 45].map((elapsed) => frameAfter(0, elapsed, 10)), [0, 1, 2, 4]);
  assert.deepEqual([19, 20, 50].map((elapsed) => frameAfter(2, elapsed, 5)), [2, 3, 4]);
  assert.deepEqual([500, 600, 3_600_000].map((elapsed) => frameAfter(0, elapsed, 1)), [5, 5, 5]);
  // 첫 프레임보다 앞선 시계에 맞는 프레임은 없다. 끝 프레임이 아니라 첫 프레임을 보인다
  assert.equal(frameAt(exchangeTimes, 9_999), 0);
});

test("clock inside a gap stays on the frame before it and tells how far into the gap it is", () => {
  const exchangeTimes = [10_000, 10_100, 10_200, 15_200, 15_300, 15_400, 16_700, 16_800];
  const gaps = [
    { lastExchangeTime: 10_200, nextExchangeTime: 15_200 },
    { lastExchangeTime: 15_400, nextExchangeTime: 16_700 },
  ];
  const positionAfter = (elapsedMilliseconds, speed) => {
    const clock = clockAfter(10_000, elapsedMilliseconds, speed);
    return [frameAt(exchangeTimes, clock), gapAt(gaps, clock)];
  };

  // 갭 직전 프레임의 시각과 갭 직후 프레임의 시각은 프레임을 받은 때라서 갭 안이 아니다
  assert.deepEqual(positionAfter(200, 1), [2, null]);
  assert.deepEqual(positionAfter(201, 1), [2, { gapSeconds: 5, elapsedSeconds: 0.001 }]);
  assert.deepEqual(positionAfter(2_700, 1), [2, { gapSeconds: 5, elapsedSeconds: 2.5 }]);
  assert.deepEqual(positionAfter(5_200, 1), [3, null]);
  // 배속을 올려도 갭의 몇 초째인지는 거래소 시각으로 센다
  assert.deepEqual(positionAfter(270, 10), [2, { gapSeconds: 5, elapsedSeconds: 2.5 }]);
  assert.deepEqual(positionAfter(600, 10), [5, { gapSeconds: 1.3, elapsedSeconds: 0.6 }]);
  assert.equal(
    gapNotice({ gapSeconds: 24.9, elapsedSeconds: 2.6 }),
    "받은 프레임이 없는 구간이다. 갭 24.9초 가운데 2.6초째. 호가창은 갭 직전 프레임에 멈춰 있다.",
  );
});

test("pausing stops where it was last drawn, playing resumes there and starts over after the end", () => {
  const exchangeTimes = [10_000, 10_100, 10_200, 15_200, 15_300];
  const playback = new Playback(exchangeTimes);
  assert.deepEqual(playback.positionAt(0), { frame: 0, clock: 10_000 });

  playback.toggle(1_000);
  assert.deepEqual(playback.positionAt(1_250), { frame: 2, clock: 10_250 });
  // 누름은 마지막으로 그린 뒤 80ms 에 왔다. 화면에 보인 자리에 서야 멈추는 순간 앞으로 튀지 않는다
  playback.toggle(1_330);
  assert.equal(playback.playing, false);
  assert.deepEqual(playback.positionAt(60_000), { frame: 2, clock: 10_250 });

  // 다시 누르면 선 자리에서 이어 간다. 끝 프레임에 닿으면 끝 프레임의 시각에 멈춘다
  playback.toggle(70_000);
  assert.deepEqual(playback.positionAt(74_000), { frame: 2, clock: 14_250 });
  assert.deepEqual(playback.positionAt(80_000), { frame: 4, clock: 15_300 });
  assert.equal(playback.playing, false);
  assert.deepEqual(playback.positionAt(90_000), { frame: 4, clock: 15_300 });

  // 끝 프레임에서 누르면 첫 프레임부터 재생한다
  playback.toggle(100_000);
  assert.deepEqual(playback.positionAt(100_150), { frame: 1, clock: 10_150 });
});

test("speed and moves while playing carry on from the current place and a frame step pauses first", () => {
  const exchangeTimes = Array.from({ length: 100 }, (_, frame) => 10_000 + frame * 100);
  const playback = new Playback(exchangeTimes);
  playback.toggle(0);

  // 바꾼 순간에는 자리가 그대로다. 그 뒤부터 새 배속으로 간다
  playback.setSpeed(10, 3_000);
  assert.deepEqual(playback.positionAt(3_000), { frame: 30, clock: 13_000 });
  assert.deepEqual(playback.positionAt(3_500), { frame: 80, clock: 18_000 });

  // 재생 중에 옮기면 옮긴 자리에서 이어 간다
  playback.moveTo(11_000, 3_600);
  assert.deepEqual(playback.positionAt(3_700), { frame: 20, clock: 12_000 });

  // 한 프레임 넘기기는 그린 자리에 멈춘 뒤 옮긴다. 양 끝 밖으로는 안 간다
  playback.step(1, 3_750);
  assert.equal(playback.playing, false);
  assert.deepEqual(playback.positionAt(9_000), { frame: 21, clock: 12_100 });
  playback.moveTo(10_050, 9_000);
  playback.step(-1, 9_000);
  assert.deepEqual(playback.positionAt(9_000), { frame: 0, clock: 10_000 });
  playback.moveTo(19_900, 9_000);
  playback.step(1, 9_000);
  assert.deepEqual(playback.positionAt(9_000), { frame: 99, clock: 19_900 });
});