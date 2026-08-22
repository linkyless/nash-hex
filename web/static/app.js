const SIZE      = 5;
const CELLS     = SIZE * SIZE;
const HEX_W     = 18;
const HEX_H     = 20;
const SLOPE     = 4;
const ROW_STEP  = 16;
const COL_SHIFT = 9;
const ORIGIN_X  = 9;
const ORIGIN_Y  = 8;

const INK       = "#121110";
const BOARD     = "#262219";
const RULE      = "#0e0d0c";
const IVORY     = "#e6dcc4";
const IVORY_LO  = "#a89f8b";
const IVORY_HI  = "#fdf6e4";
const VERD      = "#4f8f80";
const VERD_LO   = "#33604f";
const VERD_HI   = "#7fbfae";
const BRASS     = "#c9a227";
const HEAT_TOP  = [122, 102, 52];
const HEAT_LOW  = [38, 34, 25];

const TONE_YOU  = 392;
const TONE_NASH = 261;
const TONE_END  = 523;

let audio = null;
let muted = false;

function wake() {
  if (muted) return;
  if (!audio) audio = new (window.AudioContext || window.webkitAudioContext)();
  if (audio.state === "suspended") audio.resume();
}

function blip(frequency, seconds, gain) {
  if (muted || !audio) return;

  const now = audio.currentTime;
  const osc = audio.createOscillator();
  const amp = audio.createGain();

  osc.type = "square";
  osc.frequency.setValueAtTime(frequency, now);

  amp.gain.setValueAtTime(0, now);
  amp.gain.linearRampToValueAtTime(gain, now + 0.008);
  amp.gain.exponentialRampToValueAtTime(0.0001, now + seconds);

  osc.connect(amp).connect(audio.destination);
  osc.start(now);
  osc.stop(now + seconds);
}

function chime(won) {
  const steps = won ? [0, 4, 7] : [7, 3, 0];
  steps.forEach((step, i) => {
    setTimeout(() => blip(TONE_END * Math.pow(2, step / 12), 0.16, 0.05), i * 90);
  });
}

const STONE_X = 3;
const STONE_Y = 4;
const STONE = [
  [4, 7], [2, 9], [1, 10], [1, 10], [0, 11], [0, 11],
  [0, 11], [0, 11], [1, 10], [1, 10], [2, 9], [4, 7],
];

const canvas  = document.getElementById("board");
const ctx     = canvas.getContext("2d", { alpha: false });
const gauge   = document.getElementById("gauge");
const verdict = document.getElementById("verdict");
const gloss   = document.getElementById("gloss");
const gaugeYou  = document.getElementById("gaugeYou");
const gaugeNash = document.getElementById("gaugeNash");
const simsInput = document.getElementById("sims");
const simsValue = document.getElementById("simsValue");
const firstInput = document.getElementById("first");
const restart    = document.getElementById("restart");
const again      = document.getElementById("again");

const calm = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const lookup = new Int16Array(canvas.width * canvas.height).fill(-1);

let state    = null;
let overlay  = "off";
let busy     = false;
let lastSims = 300;
let popCell  = -1;

function hexX(row, col) {
  return ORIGIN_X + col * HEX_W + row * COL_SHIFT;
}

function hexY(row) {
  return ORIGIN_Y + row * ROW_STEP;
}

function cutAt(y) {
  if (y < SLOPE) return (SLOPE - y) * 2;
  if (y >= HEX_H - SLOPE) return (y - (HEX_H - SLOPE) + 1) * 2;
  return 0;
}

function fillHex(x, y, color, inset) {
  ctx.fillStyle = color;
  const from = inset ? 1 : 0;
  const to   = inset ? HEX_H - 1 : HEX_H;
  for (let dy = from; dy < to; dy++) {
    const cut = cutAt(dy);
    const x0  = cut + (inset ? 1 : 0);
    const x1  = HEX_W - 1 - cut - (inset ? 1 : 0);
    if (x0 > x1) continue;
    ctx.fillRect(x + x0, y + dy, x1 - x0 + 1, 1);
  }
}

function fillStone(x, y, base, light, dark, small) {
  ctx.fillStyle = base;

  if (small) {
    ctx.fillRect(x + STONE_X + 3, y + STONE_Y + 4, 6, 4);
    ctx.fillRect(x + STONE_X + 4, y + STONE_Y + 3, 4, 6);
    return;
  }

  STONE.forEach(([x0, x1], dy) => {
    ctx.fillRect(x + STONE_X + x0, y + STONE_Y + dy, x1 - x0 + 1, 1);
  });

  ctx.fillStyle = light;
  ctx.fillRect(x + STONE_X + 3, y + STONE_Y + 1, 3, 1);
  ctx.fillRect(x + STONE_X + 2, y + STONE_Y + 2, 2, 1);

  ctx.fillStyle = dark;
  ctx.fillRect(x + STONE_X + 7, y + STONE_Y + 9, 3, 1);
  ctx.fillRect(x + STONE_X + 8, y + STONE_Y + 8, 2, 1);
}

function heatColor(weight) {
  const step = Math.round(weight * 5) / 5;
  const mix  = HEAT_LOW.map((low, i) => Math.round(low + (HEAT_TOP[i] - low) * step));
  return `rgb(${mix[0]}, ${mix[1]}, ${mix[2]})`;
}

function buildLookup() {
  for (let row = 0; row < SIZE; row++) {
    for (let col = 0; col < SIZE; col++) {
      const x = hexX(row, col);
      const y = hexY(row);
      for (let dy = 1; dy < HEX_H - 1; dy++) {
        const cut = cutAt(dy) + 1;
        for (let dx = cut; dx <= HEX_W - 2 - cut; dx++) {
          lookup[(y + dy) * canvas.width + (x + dx)] = row * SIZE + col;
        }
      }
    }
  }
}

function drawEdges() {
  const side = HEX_H - 2 * SLOPE;

  for (let i = 0; i < SIZE; i++) {
    ctx.fillStyle = IVORY;
    ctx.fillRect(hexX(0, i) + 1, ORIGIN_Y - 5, HEX_W - 2, 3);
    ctx.fillRect(hexX(SIZE - 1, i) + 1, hexY(SIZE - 1) + HEX_H + 2, HEX_W - 2, 3);

    ctx.fillStyle = VERD;
    ctx.fillRect(hexX(i, 0) - 5, hexY(i) + SLOPE, 3, side);
    ctx.fillRect(hexX(i, SIZE - 1) + HEX_W + 2, hexY(i) + SLOPE, 3, side);
  }
}

function render() {
  ctx.fillStyle = INK;
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  drawEdges();

  const weights = state ? (overlay === "search" ? state.search : overlay === "instinct" ? state.instinct : null) : null;
  const peak    = weights ? Math.max(...weights) : 0;

  for (let index = 0; index < CELLS; index++) {
    const row = Math.floor(index / SIZE);
    const col = index % SIZE;
    const x   = hexX(row, col);
    const y   = hexY(row);
    const own = state ? state.grid[index] : 0;

    const framed = state && state.nash_move === index;
    fillHex(x, y, framed ? BRASS : RULE, false);

    let fill = BOARD;
    if (own === 0 && weights && peak > 0 && weights[index] > 0) {
      fill = heatColor(weights[index] / peak);
    }
    fillHex(x, y, fill, true);

    if (own === 1) {
      fillStone(x, y, IVORY, IVORY_HI, IVORY_LO, popCell === index);
    } else if (own === -1) {
      fillStone(x, y, VERD, VERD_HI, VERD_LO, popCell === index);
    }
  }
}

function paintGauge(evaluation, human) {
  const total = 24;
  const share = Math.round(((evaluation + 1) / 2) * total);
  const mine  = human === 1 ? IVORY : VERD;
  const yours = human === 1 ? VERD : IVORY;

  gauge.textContent = "";
  for (let i = 0; i < total; i++) {
    const tick = document.createElement("span");
    tick.className = "tick";
    tick.style.background = i < share ? mine : yours;
    gauge.appendChild(tick);
  }
}

function paintStatus() {
  verdict.classList.remove("is-waiting", "is-win");

  if (!state) {
    verdict.textContent = "";
    return;
  }

  if (state.winner !== null) {
    verdict.textContent = state.winner === state.human ? "you won" : "nash won";
    verdict.classList.add("is-win");
    canvas.classList.add("is-idle");
    again.classList.remove("is-hidden");
    return;
  }

  canvas.classList.remove("is-idle");
  again.classList.add("is-hidden");

  if (busy) {
    verdict.textContent = "nash is thinking";
    verdict.classList.add("is-waiting");
  } else {
    verdict.textContent = "your move";
  }
}

function paintGloss() {
  if (overlay === "search") {
    gloss.textContent = `how nash spent its ${lastSims} simulations`;
  } else if (overlay === "instinct") {
    gloss.textContent = "where the network would play with no search";
  } else {
    gloss.textContent = "";
  }
}

function paint() {
  render();
  paintStatus();
  paintGloss();
  if (state) {
    paintGauge(state.evaluation, state.human);
    gaugeYou.style.color  = state.human === 1 ? IVORY : VERD;
    gaugeNash.style.color = state.human === 1 ? VERD : IVORY;
  }
}

async function send(move) {
  if (busy) return;
  busy = true;
  lastSims = Number(simsInput.value);
  paintStatus();

  try {
    const response = await fetch("/api/play", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        moves: state ? state.moves : [],
        move: move,
        human: state ? state.human : (firstInput.checked ? 1 : -1),
        simulations: lastSims,
      }),
    });

    if (!response.ok) {
      busy = false;
      paint();
      return;
    }

    const next   = await response.json();
    const landed = next.nash_move;
    state = next;

    if (landed !== null) blip(TONE_NASH, 0.07, 0.06);
    if (next.winner !== null) setTimeout(() => chime(next.winner === next.human), 160);

    if (landed !== null && !calm) {
      popCell = landed;
      busy = false;
      paint();
      setTimeout(() => { popCell = -1; paint(); }, 90);
      return;
    }

    busy = false;
    paint();
  } catch (error) {
    busy = false;
    verdict.textContent = "the engine did not answer. try again";
  }
}

function newGame() {
  state = null;
  popCell = -1;
  const human = firstInput.checked ? 1 : -1;
  state = { moves: [], grid: new Array(CELLS).fill(0), legal: [], current: 1,
            winner: null, human: human, evaluation: 0, instinct: new Array(CELLS).fill(0),
            search: new Array(CELLS).fill(0), nash_move: null, elapsed_ms: 0 };
  paint();
  send(null);
}

canvas.addEventListener("click", (event) => {
  if (!state || busy || state.winner !== null) return;
  if (state.current !== state.human) return;

  const rect = canvas.getBoundingClientRect();
  const x = Math.floor((event.clientX - rect.left) / rect.width * canvas.width);
  const y = Math.floor((event.clientY - rect.top) / rect.height * canvas.height);
  if (x < 0 || y < 0 || x >= canvas.width || y >= canvas.height) return;

  const index = lookup[y * canvas.width + x];
  if (index < 0 || state.grid[index] !== 0) return;

  wake();
  blip(TONE_YOU, 0.06, 0.05);
  send(index);
});

document.querySelectorAll("[data-overlay]").forEach((key) => {
  key.addEventListener("click", () => {
    overlay = key.dataset.overlay;
    document.querySelectorAll("[data-overlay]").forEach((other) => {
      other.classList.toggle("is-live", other === key);
    });
    paint();
  });
});

simsInput.addEventListener("input", () => {
  simsValue.textContent = simsInput.value;
});

document.querySelectorAll("[data-sound]").forEach((key) => {
  key.addEventListener("click", () => {
    muted = key.dataset.sound === "off";
    if (!muted) wake();
    document.querySelectorAll("[data-sound]").forEach((other) => {
      other.classList.toggle("is-live", other === key);
    });
  });
});

restart.addEventListener("click", newGame);
again.addEventListener("click", newGame);
firstInput.addEventListener("change", newGame);

ctx.imageSmoothingEnabled = false;
buildLookup();
newGame();