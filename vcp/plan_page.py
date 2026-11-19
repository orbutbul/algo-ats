"""Sweep-planning page for the VCP detector.

Served at ``/plan`` by the local viewer. The timing model is the 500-ticker,
10-year measurement (seed 17, 8 workers, default parameters). Sliders only
change the plan; they do not run a scan.
"""

from __future__ import annotations

PLAN_PAGE = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>VCP sweep planner</title>
<style>
  :root {
    --ink: #1c2430;
    --muted: #5c6775;
    --line: #d9dee6;
    --paper: #f6f7f9;
    --panel: #ffffff;
    --accent: #0f6a4f;
    --warn: #8a4b08;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    color: var(--ink);
    background: var(--paper);
    font: 15px/1.45 "Segoe UI", Helvetica, sans-serif;
  }
  .wrap { width: min(1180px, calc(100% - 2rem)); margin: 0 auto; padding: 1.4rem 0 3rem; }
  header { margin-bottom: 1rem; }
  h1 { font-size: 1.6rem; font-weight: 650; margin: 0 0 0.3rem; letter-spacing: -0.02em; }
  a { color: var(--accent); }
  .lede { margin: 0; color: var(--muted); max-width: 46rem; }
  .bench {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.6rem;
    margin: 1rem 0;
  }
  .bench div, .card, .result {
    background: var(--panel);
    border: 1px solid var(--line);
  }
  .bench div { padding: 0.7rem 0.8rem; }
  .bench strong { display: block; font-size: 1.15rem; }
  .bench span, .hint { color: var(--muted); font-size: 0.82rem; }
  .layout { display: grid; grid-template-columns: minmax(0, 1.4fr) minmax(280px, 0.8fr); gap: 1rem; align-items: start; }
  .card { padding: 0.85rem 0.95rem 1rem; margin-bottom: 0.7rem; }
  .card h2, .result h2 { font-size: 0.95rem; margin: 0 0 0.45rem; }
  .row { display: flex; justify-content: space-between; gap: 0.8rem; align-items: baseline; }
  .desc { margin: 0.35rem 0 0.6rem; color: var(--muted); font-size: 0.88rem; }
  .controls { display: grid; grid-template-columns: 1fr 1fr 1.4fr; gap: 0.45rem 0.7rem; align-items: end; }
  label { display: grid; gap: 0.15rem; font-size: 0.75rem; color: var(--muted); }
  input[type="number"], input[type="range"] { width: 100%; }
  input[type="number"] { padding: 0.28rem 0.4rem; border: 1px solid var(--line); font: inherit; color: var(--ink); }
  .check { display: flex; align-items: center; gap: 0.4rem; font-size: 0.85rem; color: var(--ink); }
  .presets { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-bottom: 0.7rem; }
  button {
    border: 1px solid var(--line);
    background: var(--panel);
    font: inherit;
    padding: 0.3rem 0.6rem;
    cursor: pointer;
  }
  button.on { border-color: var(--accent); color: var(--accent); }
  .result { position: sticky; top: 0.8rem; padding: 0.95rem 1rem 1.1rem; }
  .big { font-size: 2rem; font-weight: 650; letter-spacing: -0.03em; margin: 0.1rem 0; }
  .stat { display: flex; justify-content: space-between; gap: 1rem; padding: 0.28rem 0; border-top: 1px solid var(--line); }
  .warn { color: var(--warn); margin: 0.6rem 0 0; }
  table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
  th, td { text-align: left; padding: 0.28rem 0.2rem; border-bottom: 1px solid var(--line); vertical-align: top; }
  th { color: var(--muted); font-weight: 600; }
  @media (max-width: 860px) {
    .layout, .bench { grid-template-columns: 1fr; }
    .result { position: static; }
  }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>VCP sweep planner</h1>
    <p class="lede">Set the range and step for each gate. The grid count, the projected run time, and the result size update from the 500-ticker measurement. Nothing here places an order or runs the detector.</p>
    <p class="lede"><a href="/">Back to charts</a></p>
  </header>
  <section class="bench" id="bench"></section>
  <div class="presets">
    <button type="button" id="preset-3" class="on">3 levels each</button>
    <button type="button" id="preset-5">5 levels each</button>
    <button type="button" id="preset-fine">Fine step</button>
    <button type="button" id="mode-grid" class="on">Grid</button>
    <button type="button" id="mode-sobol">Sobol sample</button>
  </div>
  <div class="layout">
    <div>
      <div id="params"></div>
      <section class="card">
        <h2>Held fixed for this plan</h2>
        <p class="desc">These still decide whether a shelf is a VCP. They are not in the product above. Change one only in a later pass.</p>
        <table>
          <thead><tr><th>Gate</th><th>Value</th><th>What it does</th></tr></thead>
          <tbody id="fixed"></tbody>
        </table>
      </section>
    </div>
    <aside class="result">
      <div id="result"></div>
      <label style="margin-top:0.7rem">Names in the run
        <input id="universe" type="number" min="1" max="20000" step="1" value="500"/>
      </label>
      <p class="hint">Presets:
        <button type="button" data-n="500">500 measured</button>
        <button type="button" data-n="4164">4,164 with 10y</button>
        <button type="button" data-n="6671">6,671 all names</button>
      </p>
      <label>Workers <span id="worker-label">8</span>
        <input id="workers" type="range" min="1" max="16" step="1" value="8"/>
      </label>
      <label>Sobol samples, used in that mode
        <input id="sobol" type="number" min="16" max="4096" step="16" value="128"/>
      </label>
      <p class="hint" id="method-note"></p>
    </aside>
  </div>
</div>
<script>
const BENCH = {
  n: 500,
  workers: 8,
  wall_s: 36.409710599982645,
  startup_s: 8.3,
  load_s: 22.649425799958408,
  detect_s: 158.15029900020454,
  hits: 418,
  tickers_with_hits: 220,
  bytes: 407109,
  same_day_extra: 59,
  deep_first_legs: 1,
  conf_min: 0.647,
  window: "2016-11-09 to 2026-11-09",
  eligible: 4164,
  universe: 6671,
  seed: 17
};
BENCH.marginal_s = BENCH.wall_s - BENCH.startup_s;
BENCH.cpu_s = BENCH.load_s + BENCH.detect_s;
BENCH.load_share = BENCH.load_s / BENCH.cpu_s;
BENCH.detect_share = BENCH.detect_s / BENCH.cpu_s;
BENCH.bytes_per_hit = BENCH.bytes / BENCH.hits;
BENCH.hits_per_ticker = BENCH.hits / BENCH.n;

const PARAMS = [
  {id:"min_confidence", def:0.45, lo:0.40, hi:0.80, step:0.20, finest:0.01, on:true,
    text:"Lowest composite score that is allowed to emit. The score mixes how close each return sits to the shelf, whether lows are rising, how fast the legs shrink, how dry the last leg is, and how much volume expands on the breakout. On this 500-name sample every hit that cleared the other gates already scored at least 0.65, so a floor under that did not remove anything."},
  {id:"shelf_tol", def:0.025, lo:0.012, hi:0.040, step:0.014, finest:0.001, on:true,
    text:"How far under the shelf a return high may be and still count as a touch of that same high. Tighter forces a cleaner reclaim. In an earlier 4-draw probe, 0.015 was tight enough that AMD 2019-04-03 stopped qualifying."},
  {id:"noise_depth_frac", def:0.33, lo:0.20, hi:0.50, step:0.15, finest:0.01, on:true,
    text:"A dip shallower than this fraction of the first leg is noise, not its own contraction. It decides whether the last contraction is a tiny micro-coil or a real leg. On SMCI 2024-01-08 the default 0.33 read the last leg as 1.2%; 0.42 read it as 6.2%."},
  {id:"contraction_ratio", def:0.80, lo:0.60, hi:0.95, step:0.175, finest:0.01, on:true,
    text:"Hard gate: each later leg must be no deeper than this fraction of the previous leg. The score still prefers a shrink near one half. The strict corner of an earlier probe (0.61, together with a tight volume gate) cut the hit list sharply and lost AMD."},
  {id:"min_first_depth", def:0.06, lo:0.04, hi:0.12, step:0.04, finest:0.005, on:true,
    text:"Shallowest first leg that still counts, as a fraction of the shelf. 0.06 is a 6% drop. First legs near 10–11%, the CELH shape, die if this moves above about 0.12."},
  {id:"max_final_depth", def:0.10, lo:0.07, hi:0.12, step:0.025, finest:0.005, on:true,
    text:"Deepest last leg that still counts. 0.10 is 10% under the shelf. Clean last legs in the known shelves sit around 2–7%, so a cap under about 0.07 starts cutting those."},
  {id:"vdu_max_ratio", def:0.70, lo:0.60, hi:0.85, step:0.125, finest:0.01, on:true,
    text:"The last leg's average volume must be at or under this fraction of the 50-day volume average. AMD 2019-04-03 printed 63%. A cap of 0.63 removed that date."},
  {id:"breakout_vs_dryup_mult", def:1.6, lo:1.2, hi:2.2, step:0.50, finest:0.05, on:true,
    text:"Breakout-bar volume must be at least this many times the last leg's average volume. It also has to clear 1.3 times the 50-day average; that 1.3 stays fixed."},
  {id:"max_first_depth", def:0.50, lo:0.35, hi:0.60, step:0.125, finest:0.01, on:false,
    text:"Cap on the first leg. Not applied by the detector yet — including it multiplies the plan, and the scan time per combo stays about the same once the comparison exists. In the 500-name run, 1 of 418 hits had a first leg of 50% or more (the deepest was 64%). AMD's known keeper is 33%, so a cap near 0.45 is the split worth seeing."}
];

const FIXED = [
  ["price_space", "log", "Leg depths are compared as log drops. Footprints still print ordinary percents."],
  ["pivot_up_th / pivot_down_th", "0.035", "Zigzag only seeds the left-side shelf. It does not define the contractions."],
  ["min / max contractions", "3 / 6", "A base needs at least three legs and is cut off at six."],
  ["min_base_bars / lookback_bars", "30 / 150", "The base must span 30 to 150 trading days from the first peak to the last trough."],
  ["higher_low_tol", "0.01", "Each trough must be at least the prior trough, within 1%."],
  ["shelf_overshoot", "0.015", "A high may poke 1.5% through the shelf before that touch is treated as a break."],
  ["volume_ma_window", "50", "Dry-up and breakout volume are measured against the 50-day average."],
  ["vdu_quiet_bar_ratio", "0.55", "The quietest bar of the last leg must be at or under 55% of that average."],
  ["volume_rise_tol", "2.5", "A later leg whose average volume is more than 2.5× the previous leg fails."],
  ["breakout_rel_vol", "1.3", "Breakout volume must also be at least 1.3× the 50-day average."],
  ["max_bars_to_breakout", "20", "Close and high must clear the shelf within 20 bars of the last contraction."],
  ["trend template", "on", "Close above the 50, 150, and 200 day averages, those averages stacked, the 200 rising over 22 sessions, close at least 30% above the 52-week low and within 25% of the 52-week high."],
  ["dedupe_pivot_tol", "0.005", "A second hit is dropped when the shelf is within 0.5% and the last peak is within 3 bars. Same-day duplicates still get through: 59 extra hits in this run."]
];

const state = {mode: "grid", universe: 500, workers: 8, sobol: 128};

function levels(lo, hi, step) {
  if (!(step > 0) || !(hi >= lo)) return 1;
  return Math.max(1, Math.floor((hi - lo) / step + 1e-9) + 1);
}
function valuesOf(p) {
  if (!p.on) return [p.def];
  const n = levels(p.lo, p.hi, p.step);
  const out = [];
  for (let i = 0; i < n; i++) {
    const v = p.lo + i * p.step;
    if (v <= p.hi + 1e-9) out.push(Math.round(v * 1e6) / 1e6);
  }
  return out.length ? out : [p.lo];
}
function fmt(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return "—";
  const abs = Math.abs(n);
  if (abs >= 100) return n.toFixed(0);
  if (abs >= 10) return n.toFixed(2);
  return String(Math.round(n * 10000) / 10000);
}
function fmtTime(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return "—";
  if (seconds < 90) return seconds.toFixed(0) + " sec";
  const minutes = seconds / 60;
  if (minutes < 90) return minutes.toFixed(1) + " min";
  const hours = seconds / 3600;
  if (hours < 48) return hours.toFixed(1) + " hr";
  return (hours / 24).toFixed(1) + " days";
}
function fmtBytes(n) {
  if (n < 1e6) return (n / 1e3).toFixed(0) + " KB";
  if (n < 1e9) return (n / 1e6).toFixed(1) + " MB";
  return (n / 1e9).toFixed(2) + " GB";
}

function project(combos) {
  const scaleN = state.universe / BENCH.n;
  const scaleW = BENCH.workers / Math.max(1, state.workers);
  const startup = BENCH.startup_s * (state.workers / BENCH.workers);
  const load = BENCH.marginal_s * BENCH.load_share * scaleN * scaleW;
  const detect = BENCH.marginal_s * BENCH.detect_share * scaleN * scaleW * combos;
  return startup + load + detect;
}

function renderParams() {
  const root = document.getElementById("params");
  root.innerHTML = "";
  PARAMS.forEach((p, i) => {
    const card = document.createElement("section");
    card.className = "card";
    const n = p.on ? levels(p.lo, p.hi, p.step) : 1;
    const shown = valuesOf(p).slice(0, 8).map(fmt).join(", ");
    const more = p.on && valuesOf(p).length > 8 ? " …" : "";
    card.innerHTML = `
      <div class="row">
        <label class="check"><input type="checkbox" data-i="${i}" data-k="on" ${p.on ? "checked" : ""}/> <strong>${p.id}</strong></label>
        <span class="hint">${p.on ? n + " values" : "fixed at " + fmt(p.def)} · default ${fmt(p.def)}</span>
      </div>
      <p class="desc">${p.text}</p>
      <div class="controls">
        <label>Range low<input type="number" step="${p.finest}" data-i="${i}" data-k="lo" value="${p.lo}" ${p.on ? "" : "disabled"}/></label>
        <label>Range high<input type="number" step="${p.finest}" data-i="${i}" data-k="hi" value="${p.hi}" ${p.on ? "" : "disabled"}/></label>
        <label>Step ${fmt(p.step)}<input type="range" min="${p.finest}" max="${Math.max(p.finest, p.hi - p.lo)}" step="any" data-i="${i}" data-k="step" value="${p.step}" ${p.on ? "" : "disabled"}/></label>
      </div>
      <p class="hint points" style="margin:0.45rem 0 0">Grid points: ${shown}${more}</p>`;
    root.appendChild(card);
  });
  root.querySelectorAll("input").forEach((el) => {
    el.addEventListener("input", onEdit);
  });
}

function refreshCard(card, p) {
  const n = p.on ? levels(p.lo, p.hi, p.step) : 1;
  card.querySelector(".row .hint").textContent =
    (p.on ? n + " values" : "fixed at " + fmt(p.def)) + " · default " + fmt(p.def);
  const pts = valuesOf(p);
  const shown = pts.slice(0, 8).map(fmt).join(", ");
  card.querySelector(".points").textContent = "Grid points: " + shown + (pts.length > 8 ? " …" : "");
  card.querySelectorAll("input:not([type=checkbox])").forEach((inp) => { inp.disabled = !p.on; });
}

function onEdit(ev) {
  const el = ev.target;
  const p = PARAMS[Number(el.dataset.i)];
  const key = el.dataset.k;
  if (key === "on") p.on = el.checked;
  else p[key] = Number(el.value);
  if (!(p.step > 0)) p.step = p.finest;
  if (p.hi < p.lo) p.hi = p.lo;
  const stepInput = el.closest(".card").querySelector('input[data-k="step"]');
  if (key === "lo" || key === "hi") stepInput.max = String(Math.max(p.finest, p.hi - p.lo));
  if (key === "step") el.parentElement.childNodes[0].textContent = "Step " + fmt(p.step);
  refreshCard(el.closest(".card"), p);
  renderResult();
}

function setSteps(gaps) {
  PARAMS.forEach((p) => {
    const raw = gaps <= 0 ? p.finest : (p.hi - p.lo) / gaps;
    p.step = Math.max(p.finest, Math.round(raw * 1e6) / 1e6);
  });
  renderParams();
  renderResult();
}

function renderResult() {
  const included = PARAMS.filter((p) => p.on);
  const grid = included.reduce((acc, p) => acc * levels(p.lo, p.hi, p.step), 1);
  const combos = state.mode === "sobol" ? Math.max(1, Math.round(state.sobol)) : grid;
  const seconds = project(combos);
  const hits = state.universe * BENCH.hits_per_ticker * combos;
  const bytes = hits * BENCH.bytes_per_hit;
  const missingGate = included.some((p) => p.id === "max_first_depth");
  const lines = included.map((p) => `<div class="stat"><span>${p.id}</span><span>${levels(p.lo, p.hi, p.step)}</span></div>`).join("");
  const warn = [];
  if (grid > 20000) warn.push("This grid is large enough that a full Cartesian run is the wrong tool. A Sobol sample of 128 or 256 covers the same box.");
  if (seconds > 6 * 3600 && state.mode === "grid") warn.push("Projected time is longer than a working day on this machine at the chosen worker count.");
  if (missingGate) warn.push("max_first_depth is in the count, and the current detector does not apply that gate yet.");
  document.getElementById("result").innerHTML = `
    <h2>${state.mode === "sobol" ? "Sobol run" : "Grid run"}</h2>
    <div class="big" id="test-count">${combos.toLocaleString()}</div>
    <div class="hint" id="combo-note">${state.mode === "sobol" ? "samples, not a grid" : "parameter combinations"} · full grid would be ${grid.toLocaleString()}</div>
    <div class="stat"><span>Projected time</span><span id="wall-time">${fmtTime(seconds)}</span></div>
    <div class="stat"><span>Projected result</span><span id="data-size">${fmtBytes(bytes)}</span></div>
    <div class="stat"><span>Expected hits</span><span id="hit-count">${Math.round(hits).toLocaleString()}</span></div>
    <div class="stat"><span>Names × workers</span><span id="shape">${state.universe.toLocaleString()} × ${state.workers}</span></div>
    ${lines}
    ${warn.map((w) => `<p class="warn">${w}</p>`).join("")}`;
  document.getElementById("worker-label").textContent = String(state.workers);
  document.getElementById("method-note").textContent =
    "Time scales from the 8-worker run: about " + BENCH.startup_s.toFixed(1) +
    "s to start the processes, history loaded once, then each extra combination pays the detect share. " +
    "Hit count and file size use this run's default-parameter rate (" + BENCH.hits_per_ticker.toFixed(2) +
    " hits per name, " + Math.round(BENCH.bytes_per_hit) +
    " bytes per hit) and will move when the gates change. Other worker counts are a linear scale of that measurement, not a separate timing.";
}

function renderFixed() {
  document.getElementById("fixed").innerHTML = FIXED.map(([name, value, text]) =>
    `<tr><td>${name}</td><td>${value}</td><td>${text}</td></tr>`).join("");
}

function renderBench() {
  document.getElementById("bench").innerHTML = `
    <div><strong>36.4 sec</strong><span>500 names, 1 parameter set, 8 workers</span></div>
    <div><strong>418 hits</strong><span>220 of 500 names · 59 extra same-day copies</span></div>
    <div><strong>407 KB</strong><span>dates, footprints, depths, dry-up, next 20 sessions</span></div>
    <div><strong>2016-11-09</strong><span>through 2026-11-09 · seed 17 · 4,164 names had the window</span></div>`;
}

document.getElementById("preset-3").onclick = () => { setSteps(2); markPreset("preset-3"); };
document.getElementById("preset-5").onclick = () => { setSteps(4); markPreset("preset-5"); };
document.getElementById("preset-fine").onclick = () => {
  PARAMS.forEach((p) => { p.step = p.finest; });
  renderParams(); renderResult(); markPreset("preset-fine");
};
function markPreset(id) {
  ["preset-3","preset-5","preset-fine"].forEach((k) => document.getElementById(k).classList.toggle("on", k === id));
}
document.getElementById("mode-grid").onclick = () => { state.mode = "grid"; markMode(); renderResult(); };
document.getElementById("mode-sobol").onclick = () => { state.mode = "sobol"; markMode(); renderResult(); };
function markMode() {
  document.getElementById("mode-grid").classList.toggle("on", state.mode === "grid");
  document.getElementById("mode-sobol").classList.toggle("on", state.mode === "sobol");
}

document.getElementById("universe").addEventListener("input", (ev) => {
  state.universe = Math.max(1, Number(ev.target.value) || 1);
  renderResult();
});
document.getElementById("workers").addEventListener("input", (ev) => {
  state.workers = Math.max(1, Number(ev.target.value) || 1);
  renderResult();
});
document.getElementById("sobol").addEventListener("input", (ev) => {
  state.sobol = Math.max(1, Number(ev.target.value) || 1);
  renderResult();
});
document.querySelectorAll("[data-n]").forEach((btn) => {
  btn.addEventListener("click", () => {
    state.universe = Number(btn.dataset.n);
    document.getElementById("universe").value = String(state.universe);
    renderResult();
  });
});

renderBench();
renderFixed();
renderParams();
renderResult();
</script>
</body>
</html>
'''
