// 호가창 재생 화면. 파이썬이 구간을 고를 때 배열과 고정값을 한 번 넘기고, 재생 시계와 조작은 브라우저 안에서만 돈다.
// book.js 와 playback.js 의 이름을 import 없이 쓴다. 컴포넌트는 js 문자열 하나만 받아서
// player.py 가 plotly.js 뒤에 세 파일을 이어 붙여 모듈 하나로 보낸다

// 재생 중 1초에 다시 그리는 횟수. 화면 주기(초당 60번)마다 그리면 plotly 가 못 따라간다. 브라우저에서 재어 정한다
const REDRAWS_PER_SECOND = 10;
// 갭 바로가기는 갭 직전 프레임보다 이만큼 앞에 선다. 재생하면 호가창이 움직이다 멈추는 것부터 보인다
const GAP_LEAD_MILLISECONDS = 3000;
const MID_COLOR = "#008300";
const PLAYBACK_LINE_COLOR = "#9085e9";
const GAP_EDGE_COLOR = "#898781";
const GAP_FILL_COLOR = "rgba(137, 135, 129, 0.35)";
// 막대 40개에 글자가 한 줄씩 붙는다. 위아래 여백을 빼고 한 줄에 16픽셀이 돌아가는 높이다
const BOOK_HEIGHT = 760;
const MID_HEIGHT = 380;
const MARGIN = { l: 70, r: 20, t: 60, b: 50 };
const PLOT_CONFIG = { responsive: true, displaylogo: false };

// 컴포넌트는 페이지에 그대로 붙는다(isolate_styles=False). 그림자 DOM 안에서는 plotly 가 문서 머리에 넣는 스타일이 안 닿는다
const STYLE = `
.tickfold-player { font-family: var(--st-font); color: var(--st-text-color); }
.tickfold-player .row { display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center; margin-bottom: 0.5rem; }
.tickfold-player button, .tickfold-player input[name=time] {
  font: inherit; color: inherit; background: var(--st-secondary-background-color);
  border: 1px solid var(--st-border-color); border-radius: var(--st-base-radius); padding: 0.25rem 0.75rem;
}
.tickfold-player button[aria-pressed=true] { border-color: var(--st-primary-color); color: var(--st-primary-color); }
.tickfold-player .problem { color: var(--st-red-text-color); }
.tickfold-player .columns { display: flex; flex-wrap: wrap; gap: 1rem; }
.tickfold-player .columns > div { flex: 1 1 28rem; min-width: 0; }
.tickfold-player input[type=range] { width: 100%; }
.tickfold-player .notice { padding: 0.75rem 1rem; border-radius: var(--st-base-radius);
  background: var(--st-yellow-background-color); color: var(--st-yellow-text-color); }
.tickfold-player .notice:empty { display: none; }
`;

const MARKUP = `
<style>${STYLE}</style>
<div class="row">
  <button data-action="toggle"></button>
  <button data-action="first">처음으로</button>
  <button data-action="back">◀ 한 프레임</button>
  <button data-action="forward">한 프레임 ▶</button>
  <span>배속</span><span class="speeds row"></span>
</div>
<form class="row"><label>UTC 시각 <input name="time" placeholder="07:32:10" size="12" autocomplete="off"></label>
<button>이동</button><span class="problem"></span></form>
<div class="row gaps"></div>
<div class="columns">
  <div class="book"></div>
  <div><div class="row"><input type="range" step="1"><span class="frame"></span></div>
  <div class="mid"></div><div class="notice"></div></div>
</div>`;

// player.py 가 float64 바이트를 그대로 base64 로 적어 보낸 배열을 푼다
function decodeFloat64(text) {
  const binary = atob(text);
  // Uint8Array.from(binary, 함수) 는 글자마다 함수를 불러 15분 구간에 0.36초, 한 시간 구간에 1.4초가 든다(node). 루프는 그 20분의 1
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index++) bytes[index] = binary.charCodeAt(index);
  return new Float64Array(bytes.buffer);
}

// 그림의 배경과 글자를 Streamlit 테마에 맞춘다. 컴포넌트에는 st.plotly_chart 가 입히는 테마가 안 입혀진다
function themeLayout(element) {
  const style = getComputedStyle(element);
  const read = (name) => style.getPropertyValue(name).trim() || undefined;
  const axis = { gridcolor: read("--st-border-color-light"), zerolinecolor: read("--st-border-color") };
  return {
    paper_bgcolor: "rgba(0, 0, 0, 0)",
    plot_bgcolor: "rgba(0, 0, 0, 0)",
    font: { family: read("--st-font"), color: read("--st-text-color") },
    xaxis: axis,
    yaxis: { ...axis },
  };
}

// Streamlit 은 data 나 테마가 바뀌면 앞 정리 함수를 안 부르고 이 함수를 같은 parentElement 로 다시 부른다.
// 정리 함수는 화면에서 빠질 때만 부른다. 앞 화면을 여기서 치우지 않으면 화면이 둘이 되고 앞 재생 주기가 계속 돈다
const cleanups = new WeakMap();

// 구간 하나를 재생하는 화면을 parentElement 에 만든다
export default function ({ data, parentElement }) {
  cleanups.get(parentElement)?.();
  const root = document.createElement("div");
  root.className = "tickfold-player";
  root.innerHTML = MARKUP;
  parentElement.appendChild(root);
  const find = (selector) => root.querySelector(selector);
  const exchangeTimes = decodeFloat64(data.exchangeTime);
  const asks = { prices: decodeFloat64(data.askPrice), quantities: decodeFloat64(data.askQuantity) };
  const bids = { prices: decodeFloat64(data.bidPrice), quantities: decodeFloat64(data.bidQuantity) };
  const ceiling = data.quantityCeiling;
  const lastFrame = exchangeTimes.length - 1;
  const playback = new Playback(exchangeTimes);
  const theme = themeLayout(root);

  const bookElement = find(".book");
  const bookLayout = {
    ...theme,
    xaxis: { ...theme.xaxis, title: { text: `수량 (막대는 ${ceiling.toFixed(2)} 에서 자름)` }, range: [0, ceiling] },
    yaxis: {
      ...theme.yaxis,
      categoryarray: levelOrder(),
      categoryorder: "array",
      // 눈금을 plotly 에 맡기면 한 줄이 14픽셀보다 낮아질 때 레벨 이름을 건너뛰며 적는다
      tickmode: "linear",
    },
    showlegend: false,
    bargap: 0.15,
    height: BOOK_HEIGHT,
    margin: MARGIN,
  };
  const drawBook = (frame) => {
    const traces = bookTraces(asks, bids, frame, ceiling);
    const title = bookTitle(exchangeTimes[frame], bids.prices[frame * LEVELS], asks.prices[frame * LEVELS]);
    Plotly.react(bookElement, traces, { ...bookLayout, title: { text: title } }, PLOT_CONFIG);
  };

  const midElement = find(".mid");
  const gapShapes = data.gaps.map((gap) => ({
    type: "rect",
    xref: "x",
    yref: "y domain",
    x0: gap.shadeStart,
    x1: gap.shadeEnd,
    y0: 0,
    y1: 1,
    fillcolor: GAP_FILL_COLOR,
    // 테두리가 없으면 한 시간짜리 구간에서 1초 갭은 한 픽셀이 안 돼 안 보인다
    line: { color: GAP_EDGE_COLOR, width: 1 },
    layer: "below",
  }));
  // 고른 프레임의 시각이 아니라 재생 시계에 긋는다. 갭을 지나는 동안 호가창은 멈춰 있어도 세로선은 띠 안을 지나간다
  const playbackLine = { type: "line", xref: "x", yref: "y domain", y0: 0, y1: 1 };
  playbackLine.line = { color: PLAYBACK_LINE_COLOR, width: 2 };
  const playbackLineKey = `shapes[${gapShapes.length}]`;
  Plotly.newPlot(
    midElement,
    [
      {
        type: "scatter",
        mode: "lines",
        // 갭 자리의 null 에서 선이 끊긴다. 이으면 받지 못한 동안의 가격을 아는 것처럼 보인다
        x: data.midLine.time,
        y: data.midLine.mid,
        line: { color: MID_COLOR, width: 2 },
        hovertemplate: "%{x|%H:%M:%S.%L} UTC<br>mid %{y:,.3f}<extra></extra>",
      },
    ],
    {
      ...theme,
      title: { text: "mid · 회색 띠는 갭, 세로선은 재생 위치" },
      xaxis: { ...theme.xaxis, title: { text: "거래소 시각 (UTC)" } },
      yaxis: { ...theme.yaxis, tickformat: ",.2f" },
      shapes: [...gapShapes, playbackLine],
      showlegend: false,
      height: MID_HEIGHT,
      margin: MARGIN,
    },
    PLOT_CONFIG,
  );

  const toggleButton = find("[data-action=toggle]");
  const slider = find("input[type=range]");
  slider.max = lastFrame;
  const frameLabel = find(".frame");
  const notice = find(".notice");
  const problem = find(".problem");
  let shownFrame = null;
  let shownClock = null;
  let timer = null;
  const draw = () => {
    const { frame, clock } = playback.positionAt(performance.now());
    if (frame !== shownFrame) drawBook(frame);
    if (clock !== shownClock) {
      const time = utcText(clock);
      Plotly.relayout(midElement, { [`${playbackLineKey}.x0`]: time, [`${playbackLineKey}.x1`]: time });
    }
    // 끌고 있는 슬라이더에 값을 적으면 손잡이가 손에서 빠져나간다
    if (!slider.matches(":active")) slider.value = frame;
    frameLabel.textContent = `프레임 ${groupedText(frame, 0)} / ${groupedText(lastFrame, 0)}`;
    const passage = gapAt(data.gaps, clock);
    notice.textContent = passage === null ? "" : gapNotice(passage);
    toggleButton.textContent = playback.playing ? "멈춤" : "재생";
    for (const button of root.querySelectorAll("[data-speed]")) {
      button.setAttribute("aria-pressed", String(Number(button.dataset.speed) === playback.speed));
    }
    [shownFrame, shownClock] = [frame, clock];
    if (playback.playing && timer === null) timer = setInterval(draw, 1000 / REDRAWS_PER_SECOND);
    // 끝 프레임에 닿아 멈추면 다시 그릴 것이 없다
    if (!playback.playing && timer !== null) {
      clearInterval(timer);
      timer = null;
    }
  };
  // 조작마다 재생 상태를 바꾸고 바로 다시 그린다. 다음 주기까지 기다리면 누른 것이 늦게 보인다
  const act = (change) => () => {
    problem.textContent = "";
    change(performance.now());
    draw();
  };

  find(".speeds").append(
    ...SPEEDS.map((speed) => {
      const button = Object.assign(document.createElement("button"), { textContent: `${speed}배` });
      button.dataset.speed = speed;
      button.addEventListener("click", act((now) => playback.setSpeed(speed, now)));
      return button;
    }),
  );
  toggleButton.addEventListener("click", act((now) => playback.toggle(now)));
  find("[data-action=first]").addEventListener("click", act((now) => playback.moveTo(exchangeTimes[0], now)));
  find("[data-action=back]").addEventListener("click", act((now) => playback.step(-1, now)));
  find("[data-action=forward]").addEventListener("click", act((now) => playback.step(1, now)));
  slider.addEventListener("input", act((now) => playback.moveTo(exchangeTimes[Number(slider.value)], now)));
  find(".gaps").append(
    ...data.gaps.map((gap) => {
      const seconds = ((gap.nextExchangeTime - gap.lastExchangeTime) / 1000).toFixed(1);
      const lead = GAP_LEAD_MILLISECONDS / 1000;
      const label = `갭 ${utcText(gap.lastExchangeTime).slice(11, 19)} (${seconds}초) ${lead}초 전으로`;
      const button = Object.assign(document.createElement("button"), { textContent: label });
      const clock = Math.max(gap.lastExchangeTime - GAP_LEAD_MILLISECONDS, exchangeTimes[0]);
      button.addEventListener("click", act(() => playback.jumpTo(clock)));
      return button;
    }),
  );
  const [firstSecond, lastSecond] = [0, lastFrame].map((frame) => utcText(exchangeTimes[frame]).slice(11, 19));
  const problems = {
    format: "HH:MM:SS 로 적는다(밀리초는 붙여도 된다). 예: 07:32:10",
    outside: `구간은 ${firstSecond} ~ ${lastSecond} UTC 다`,
  };
  find("form").addEventListener("submit", (event) => {
    // 그냥 두면 폼이 페이지를 다시 불러온다
    event.preventDefault();
    const answer = clockFromTimeText(event.target.time.value, exchangeTimes[0], exchangeTimes[lastFrame]);
    // 구간 밖이나 틀린 형식이면 자리를 안 바꾸고 고칠 점만 보인다
    if (answer.problem === undefined) act((now) => playback.moveTo(answer.clock, now))();
    else problem.textContent = problems[answer.problem];
  });

  draw();
  const cleanup = () => {
    clearInterval(timer);
    Plotly.purge(bookElement);
    Plotly.purge(midElement);
    root.remove();
  };
  cleanups.set(parentElement, cleanup);
  return cleanup;
}