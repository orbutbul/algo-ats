"""Local web viewer: type a ticker, see log history with VCP breakouts marked."""

from __future__ import annotations

import json
import math
from pathlib import Path

from flask import Flask, Response, jsonify, render_template_string, request
from plotly.utils import PlotlyJSONEncoder

from vcp.detector import detect_vcp_breakouts
from vcp.io import OHLCV_DUCKDB_PATH, list_tickers, load_daily_ohlcv, seed_daily_ohlcv
from vcp.params import VCPParams
from vcp.plotly_chart import build_vcp_figure

PAGE = r'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>VCP Viewer</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,560;9..144,700&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet" />
  <style>
    :root {
      --ink: #1b2a4a;
      --muted: #5b667a;
      --paper: #f3efe6;
      --panel: rgba(255, 252, 245, 0.82);
      --line: rgba(27, 42, 74, 0.12);
      --accent: #0f6a4f;
      --accent-2: #b45309;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      font-family: "IBM Plex Sans", Helvetica, sans-serif;
      background:
        radial-gradient(1200px 600px at 10% -10%, rgba(15, 106, 79, 0.16), transparent 55%),
        radial-gradient(900px 500px at 100% 0%, rgba(180, 83, 9, 0.14), transparent 50%),
        linear-gradient(165deg, #ebe4d6 0%, var(--paper) 45%, #e7eef0 100%);
    }
    .wrap {
      width: min(1180px, calc(100% - 2rem));
      margin: 0 auto;
      padding: 1.6rem 0 2.5rem;
    }
    header {
      display: grid;
      gap: 0.55rem;
      margin-bottom: 1.25rem;
    }
    .brand {
      font-family: Fraunces, Georgia, serif;
      font-size: clamp(2.4rem, 5vw, 3.6rem);
      font-weight: 700;
      letter-spacing: -0.03em;
      line-height: 0.95;
      margin: 0;
    }
    .lede {
      margin: 0;
      max-width: 36rem;
      color: var(--muted);
      font-size: 1.02rem;
      line-height: 1.45;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      backdrop-filter: blur(8px);
      padding: 0.9rem;
    }
    form.controls {
      display: grid;
      grid-template-columns: 1.1fr 1fr 1fr auto;
      gap: 0.7rem;
      align-items: end;
    }
    label {
      display: grid;
      gap: 0.28rem;
      font-size: 0.78rem;
      font-weight: 600;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      color: var(--muted);
    }
    input {
      width: 100%;
      border: 1px solid var(--line);
      background: #fffdf8;
      color: var(--ink);
      padding: 0.7rem 0.75rem;
      font: 500 1rem "IBM Plex Sans", Helvetica, sans-serif;
      outline: none;
    }
    input:focus { border-color: rgba(15, 106, 79, 0.55); }
    button {
      border: 0;
      background: var(--ink);
      color: #f7f2e8;
      padding: 0.78rem 1.15rem;
      font: 600 0.95rem "IBM Plex Sans", Helvetica, sans-serif;
      cursor: pointer;
      transition: transform 120ms ease, background 120ms ease;
    }
    button:hover { background: var(--accent); transform: translateY(-1px); }
    button:disabled { opacity: 0.55; cursor: wait; transform: none; }
    .meta {
      display: flex;
      flex-wrap: wrap;
      gap: 0.55rem 1rem;
      margin: 0.85rem 0 0;
      color: var(--muted);
      font-size: 0.92rem;
    }
    .meta strong { color: var(--ink); font-weight: 600; }
    .chip {
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      padding: 0.2rem 0.5rem;
      background: rgba(15, 106, 79, 0.1);
      color: var(--accent);
      font-size: 0.82rem;
      font-weight: 600;
    }
    #chart {
      margin-top: 0.9rem;
      min-height: 720px;
      background: rgba(255,252,245,0.55);
      border: 1px solid var(--line);
    }
    .table-wrap {
      margin-top: 0.9rem;
      overflow: auto;
      border: 1px solid var(--line);
      background: rgba(255,252,245,0.7);
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.92rem;
    }
    th, td {
      padding: 0.65rem 0.75rem;
      border-bottom: 1px solid var(--line);
      text-align: left;
      white-space: nowrap;
    }
    th {
      font-size: 0.72rem;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      color: var(--muted);
      background: rgba(27, 42, 74, 0.04);
    }
    tr:last-child td { border-bottom: 0; }
    .error {
      margin-top: 0.8rem;
      padding: 0.75rem 0.85rem;
      background: rgba(190, 18, 60, 0.08);
      color: #9f1239;
      border: 1px solid rgba(190, 18, 60, 0.18);
      display: none;
    }
    .hint { color: var(--muted); font-size: 0.86rem; }
    @media (max-width: 820px) {
      form.controls { grid-template-columns: 1fr 1fr; }
      form.controls button { grid-column: 1 / -1; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <h1 class="brand">VCP</h1>
      <p class="lede">Type a ticker. See the log-scale history and every volatility contraction breakout the detector marks.</p>
    </header>

    <section class="panel">
      <form class="controls" id="form">
        <label>Ticker
          <input id="ticker" name="ticker" value="NVDA" autocomplete="off" spellcheck="false" />
        </label>
        <label>Start
          <input id="start" name="start" type="date" value="2018-01-01" />
        </label>
        <label>End
          <input id="end" name="end" type="date" />
        </label>
        <button id="go" type="submit">Show VCP</button>
      </form>
      <div class="meta" id="meta">
        <span class="hint">Loaded tickers: {{ tickers }}</span>
      </div>
      <div class="error" id="error"></div>
    </section>

    <div id="chart"></div>
    <div class="table-wrap" id="table-wrap" style="display:none">
      <table>
        <thead>
          <tr>
            <th>Breakout</th>
            <th>Footprint</th>
            <th>Pivot</th>
            <th>Stop</th>
            <th>Depths %</th>
            <th>Coil vol / MA</th>
            <th>Vs dry-up</th>
            <th>Rel vol</th>
          </tr>
        </thead>
        <tbody id="tbody"></tbody>
      </table>
    </div>
  </div>

  <script>
    const form = document.getElementById('form');
    const errorEl = document.getElementById('error');
    const metaEl = document.getElementById('meta');
    const goBtn = document.getElementById('go');
    const tableWrap = document.getElementById('table-wrap');
    const tbody = document.getElementById('tbody');

    async function loadChart() {
      errorEl.style.display = 'none';
      errorEl.textContent = '';
      goBtn.disabled = true;
      goBtn.textContent = 'Loading…';
      const ticker = document.getElementById('ticker').value.trim().toUpperCase();
      const start = document.getElementById('start').value;
      const end = document.getElementById('end').value;
      const qs = new URLSearchParams({ ticker });
      if (start) qs.set('start', start);
      if (end) qs.set('end', end);
      qs.set('autoseed', '1');
      try {
        const res = await fetch('/api/chart?' + qs.toString());
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || res.statusText);
        Plotly.newPlot('chart', data.figure.data, data.figure.layout, {
          responsive: true,
          displaylogo: false,
        });
        metaEl.innerHTML = `
          <span class="chip">${data.ticker}</span>
          <span><strong>${data.breakouts}</strong> VCP breakout(s)</span>
          <span>${data.bars} daily bars</span>
          <span>${data.seeded ? 'seeded on demand' : 'from local duckdb'}</span>
        `;
        tbody.innerHTML = '';
        if (data.events.length) {
          tableWrap.style.display = 'block';
          for (const e of data.events) {
            const tr = document.createElement('tr');
            const vdu = (e.volume_dryup_ratios || []).map(x => (Number(x) * 100).toFixed(0) + '%').join(' → ');
            tr.innerHTML = `
              <td>${e.breakout_date}</td>
              <td>${e.footprint}</td>
              <td>${e.pivot_price.toFixed(2)}</td>
              <td>${e.stop_price.toFixed(2)}</td>
              <td>${e.depths_pct.map(x => x.toFixed(1)).join(' → ')}</td>
              <td>${vdu}</td>
              <td>${Number(e.breakout_vs_dryup).toFixed(2)}×</td>
              <td>${e.breakout_rel_vol.toFixed(2)}×</td>
            `;
            tbody.appendChild(tr);
          }
        } else {
          tableWrap.style.display = 'none';
        }
      } catch (err) {
        errorEl.style.display = 'block';
        errorEl.textContent = err.message || String(err);
        tableWrap.style.display = 'none';
      } finally {
        goBtn.disabled = false;
        goBtn.textContent = 'Show VCP';
      }
    }

    form.addEventListener('submit', (ev) => {
      ev.preventDefault();
      loadChart();
    });
    loadChart();
  </script>
</body>
</html>
'''


def create_app(duckdb_path: Path | None = None) -> Flask:
    duckdb_path = Path(duckdb_path) if duckdb_path is not None else OHLCV_DUCKDB_PATH
    app = Flask(__name__)
    app.config['VCP_DUCKDB'] = duckdb_path

    @app.get('/')
    def index():
        try:
            tickers = ', '.join(list_tickers(duckdb_path)) or '(none yet — type a ticker to seed)'
        except Exception:
            tickers = '(no duckdb yet — type a ticker to seed)'
        return render_template_string(PAGE, tickers=tickers)

    @app.get('/api/tickers')
    def api_tickers():
        try:
            return jsonify({'tickers': list_tickers(duckdb_path)})
        except Exception as exc:
            return jsonify({'tickers': [], 'error': str(exc)}), 200

    @app.get('/api/chart')
    def api_chart():
        ticker = (request.args.get('ticker') or '').strip().upper()
        if not ticker:
            return jsonify({'error': 'ticker is required'}), 400
        start = request.args.get('start') or None
        end = request.args.get('end') or None
        autoseed = request.args.get('autoseed', '1') not in ('0', 'false', 'False')

        seeded = False
        try:
            ohlcv = load_daily_ohlcv(ticker, path=duckdb_path, start=start, end=end)
        except (FileNotFoundError, LookupError):
            if not autoseed:
                return jsonify({'error': f'{ticker} not in {duckdb_path}. Seed it first.'}), 404
            try:
                seed_daily_ohlcv([ticker], path=duckdb_path, period='max')
                seeded = True
                ohlcv = load_daily_ohlcv(ticker, path=duckdb_path, start=start, end=end)
            except Exception as exc:
                return jsonify({'error': f'Could not load/seed {ticker}: {exc}'}), 500
        except Exception as exc:
            return jsonify({'error': str(exc)}), 500

        result = detect_vcp_breakouts(ohlcv, ticker=ticker, params=VCPParams())
        fig = build_vcp_figure(ohlcv, result)
        events = []
        for b in result.breakouts:
            d = b.to_dict()
            d['depths_pct'] = [round((math.exp(x) - 1) * 100.0, 2) for x in b.depths]
            events.append(d)
        payload = {
            'ticker': ticker,
            'bars': len(ohlcv),
            'breakouts': len(result.breakouts),
            'seeded': seeded,
            'events': events,
            'figure': fig.to_plotly_json(),
        }
        return Response(
            json.dumps(payload, cls=PlotlyJSONEncoder),
            mimetype='application/json',
        )

    return app


def run_viewer(host: str = '127.0.0.1', port: int = 8765, duckdb_path: Path | None = None) -> None:
    app = create_app(duckdb_path)
    print(f'VCP viewer → http://{host}:{port}')
    app.run(host=host, port=port, debug=False, use_reloader=False)
