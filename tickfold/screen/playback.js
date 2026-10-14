// 재생 시계를 계산하고 재생 상태를 바꾼다. 재생은 프레임 번호가 아니라 거래소 시각을 따라간다.
// 화면(DOM)도 실제 시계도 모르는 코드라서 node --test 로 시간을 손으로 돌려 시험한다. now 는 performance.now() 밀리초다

export const SPEEDS = [1, 2, 5, 10, 30];
const DAY_MILLISECONDS = 86_400_000;

// 기준점에서 실제 시간이 elapsedMilliseconds 만큼 흐른 뒤의 재생 시계(거래소 시각, 밀리초)
export function clockAfter(anchorClock, elapsedMilliseconds, speed) {
  // 프레임 수를 세어 넘기면 프레임 간격이 고르지 않아 시간이 틀어지고, 갭이 한 걸음에 지나가 버린다
  return anchorClock + Math.round(elapsedMilliseconds * speed);
}

// 거래소 시각이 재생 시계 이하인 마지막 프레임. 다시 그리는 사이의 프레임은 건너뛴다
export function frameAt(exchangeTimes, clock) {
  // 시각이 clock 보다 큰 첫 프레임을 이분 탐색으로 찾는다
  let searchStart = 0;
  let searchEnd = exchangeTimes.length;
  while (searchStart < searchEnd) {
    const middle = (searchStart + searchEnd) >> 1;
    if (exchangeTimes[middle] <= clock) searchStart = middle + 1;
    else searchEnd = middle;
  }
  // 첫 프레임보다 앞선 시계에 맞는 프레임은 없다. 끝 프레임이 아니라 첫 프레임을 보인다
  return Math.max(searchStart - 1, 0);
}

// 재생 시계가 갭 안이면 갭의 길이와 몇 초째인지를 돌려준다. 몇 초째는 배속과 상관없이 거래소 시각으로 센다
export function gapAt(gaps, clock) {
  // 갭 직전 프레임과 직후 프레임의 시각은 프레임을 받은 때라서 갭 안이 아니다
  const gap = gaps.find((candidate) => candidate.lastExchangeTime < clock && clock < candidate.nextExchangeTime);
  if (gap === undefined) return null;
  return {
    gapSeconds: (gap.nextExchangeTime - gap.lastExchangeTime) / 1000,
    elapsedSeconds: (clock - gap.lastExchangeTime) / 1000,
  };
}

// 재생 시계가 갭 안일 때 띄우는 안내. 없으면 갭을 지나는 동안 화면이 고장 나 멈춘 것처럼 보인다
export function gapNotice(passage) {
  const { gapSeconds, elapsedSeconds } = passage;
  return (
    `받은 프레임이 없는 구간이다. 갭 ${gapSeconds.toFixed(1)}초 가운데 ${elapsedSeconds.toFixed(1)}초째.` +
    " 호가창은 갭 직전 프레임에 멈춰 있다."
  );
}

// 시각 입력칸의 "HH:MM:SS"(UTC, 밀리초는 붙여도 된다)를 재생 시계로 바꾼다. 못 바꾸면 problem 에 까닭("format", "outside")을 담는다
export function clockFromTimeText(text, firstTime, lastTime) {
  // 제목과 마우스 안내에 보이는 밀리초도 받는다. 그 시각 이하의 마지막 프레임에 서므로 프레임 시각과 같을 필요는 없다
  const match = /^(\d{1,2}):(\d{2}):(\d{2})(?:\.(\d{1,3}))?$/.exec(text.trim());
  if (match === null) return { problem: "format" };
  const [hours, minutes, seconds] = match.slice(1, 4).map(Number);
  if (hours > 23 || minutes > 59 || seconds > 59) return { problem: "format" };
  const timeOfDay = ((hours * 60 + minutes) * 60 + seconds) * 1000 + Number((match[4] ?? "").padEnd(3, "0"));
  // 첫 프레임과 같은 초는 구간 안으로 쳐서 07:25:00.014 에서 시작하는 구간에 07:25:00 을 받는다
  const earliest = firstTime - (firstTime % 1000);
  // 구간이 UTC 자정을 넘으면 같은 시각이 두 날짜에 있다. 구간 안에 드는 날짜를 고른다
  for (const day of [firstTime - (firstTime % DAY_MILLISECONDS), lastTime - (lastTime % DAY_MILLISECONDS)]) {
    const clock = day + timeOfDay;
    if (earliest <= clock && clock <= lastTime) return { clock: Math.max(clock, firstTime) };
  }
  return { problem: "outside" };
}


// 재생 상태. 멈춤과 재생 둘이고, 바꾸는 길은 이 클래스의 메서드뿐이다
export class Playback {
  constructor(exchangeTimes) {
    this.exchangeTimes = exchangeTimes;
    this.lastFrame = exchangeTimes.length - 1;
    this.speed = SPEEDS[0];
    this.pausedClock = exchangeTimes[0];
    // 재생 중에만 있다. 기준점을 잡은 순간의 재생 시계와 now. 배속을 바꾸면 그 순간의 시계로 다시 잡아서 자리가 안 튄다
    this.anchor = null;
    this.drawnClock = exchangeTimes[0];
  }

  get playing() {
    return this.anchor !== null;
  }

  // 지금의 재생 시계. 멈춘 동안에는 멈춘 자리다
  clockAt(now) {
    if (this.anchor === null) return this.pausedClock;
    return clockAfter(this.anchor.clock, now - this.anchor.moment, this.speed);
  }

  // 화면이 그릴 자리를 정한다. 재생 중에 끝 프레임에 닿으면 멈춘다
  positionAt(now) {
    let clock = this.clockAt(now);
    const frame = frameAt(this.exchangeTimes, clock);
    if (this.playing && frame === this.lastFrame) {
      // 재생 시계는 끝 프레임을 지나쳐 있다. 멈춘 자리는 끝 프레임의 시각으로 적는다
      this.anchor = null;
      clock = this.pausedClock = this.exchangeTimes[this.lastFrame];
    }
    this.drawnClock = clock;
    return { frame, clock };
  }

  // 재생 단추. 재생 중이면 멈추고, 멈춰 있으면 그 자리에서 재생한다
  toggle(now) {
    if (this.playing) {
      // 누름은 마지막으로 그린 뒤에 온다. 그 사이 흐른 만큼 앞에서 멈추면 멈추는 순간 화면이 앞으로 튄다
      this.anchor = null;
      this.pausedClock = this.drawnClock;
      return;
    }
    // 끝 프레임에서는 더 갈 데가 없어 눌러도 바로 멈춘다. 첫 프레임부터 다시 재생한다
    const atEnd = frameAt(this.exchangeTimes, this.pausedClock) === this.lastFrame;
    this.anchor = { clock: atEnd ? this.exchangeTimes[0] : this.pausedClock, moment: now };
  }

  // 배속을 바꾼다. 지금까지는 앞 배속으로 흐른 것이므로 지금의 재생 시계에서 새 배속으로 이어 간다
  setSpeed(speed, now) {
    if (this.playing) this.anchor = { clock: this.clockAt(now), moment: now };
    this.speed = speed;
  }

  // 슬라이더, 시각 입력, 처음으로가 자리를 옮긴다. 재생 중이면 옮긴 자리에서 이어 간다
  moveTo(clock, now) {
    if (this.playing) this.anchor = { clock, moment: now };
    else this.pausedClock = clock;
  }

  // 갭 바로가기가 자리를 옮긴다. 재생 중이어도 멈춘 채로 간다. 30배로 이어 가면 24.9초 갭을 0.83초에 지나가 거의 안 보인다
  jumpTo(clock) {
    this.anchor = null;
    this.pausedClock = clock;
  }

  // 한 프레임 앞(+1)이나 뒤(-1)로 간다. 재생 중이면 그린 자리에 멈춘 뒤에 옮긴다
  step(direction, now) {
    if (this.playing) this.toggle(now);
    const frame = Math.min(Math.max(frameAt(this.exchangeTimes, this.pausedClock) + direction, 0), this.lastFrame);
    this.pausedClock = this.exchangeTimes[frame];
  }
}