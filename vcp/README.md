# vcp — Volatility Contraction Pattern detector

Research detector for Mark Minervini’s VCP on daily bars from
`data/ohlcv.duckdb::ohlcv_max_daily`. No orders, no broker.

## Quick start

```bash
# Bootstrap local daily history (data/ is gitignored)
python -m vcp seed --tickers NVDA,AMD,META,TSLA,CRWD

# Log-scale history with breakout markers
python -m vcp chart NVDA --out artifacts/nvda_vcp.png --json-out artifacts/nvda_vcp.json

# Recent window
python -m vcp chart NVDA --start 2020-01-01 --out artifacts/nvda_vcp_recent.png

# Interactive web viewer (type a ticker, see history + VCP marks)
python -m vcp view --port 8765
# → http://127.0.0.1:8765
```

Uses `vectorbtpro` when installed; otherwise open-source `vectorbt` for MAs/ATR
(and a directional-change zigzag in place of PRO’s `PIVOTINFO`).

## Coil + ATR model

Contractions are contiguous **peak → trough → recovery → next peak** coils
(shared vertices). ATR drives swing significance and tightness checks, and is
plotted under volume in the viewer.

## Layout

- `params.py` — ideal-path knobs (log depths, ATR, half-rule, VDU, breakout vol)
- `pivots.py` — swing highs/lows (pct or ATR thresholds)
- `detector.py` — coil builder + VCP sequence + breakout scan
- `io.py` — read/seed DuckDB
- `plot.py` — static matplotlib log chart
- `plotly_chart.py` — interactive Plotly chart (price / volume / ATR)
- `viewer.py` — local Flask web UI
- `cli.py` — `seed` / `chart` / `scan` / `view`
