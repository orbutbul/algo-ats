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
```

Uses `vectorbtpro` when installed; otherwise open-source `vectorbt` for MAs
(and a directional-change zigzag in place of PRO’s `PIVOTINFO`).

## Layout

- `params.py` — ideal-path knobs (log depths, half-rule, VDU, breakout vol)
- `pivots.py` — swing highs/lows
- `detector.py` — contraction sequence + breakout scan
- `io.py` — read/seed DuckDB
- `plot.py` — log price + volume chart
- `cli.py` — `seed` / `chart` / `scan`
