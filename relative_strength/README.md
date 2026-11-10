# Relative-strength proxy

Opt-in calculator for an IBD/Minervini-inspired relative-strength percentile and a separate price-ratio line. It is a research tool. It is not imported by `daily_run.py`, `hourly_run.py`, `strategy_runner.py`, Airflow, or the order path. It does not download prices, open a broker, or write `data/ohlcv.duckdb`.

This is **not official IBD RS**. IBD's rating and its universe are proprietary and were updated in 2026. The number below is a transparent proxy on whatever names you pass in.

P/E is not an input. `trailingPE` and `forwardPE` stay in `extraction/fundamentals.py` (`EQUITY_MONTHLY_FIELDS`) and are not read here.

## Score

For split-aware daily closes `P` on one NYSE session `t`:

```text
R_n = P_t / P_(t-n) - 1
S   = 0.40*R_63 + 0.20*R_126 + 0.20*R_189 + 0.20*R_252
```

`n` is 63, 126, 189, and 252 **NYSE sessions**, not rows of a gappy series. The four returns are overlapping cumulative returns. They are not four independent quarters and they are not the proprietary IBD formula.

Sessions after `t` are ignored. A missing print is left missing. It is not forward-filled.

### Rank, ties, small universes

Only eligible symbols on that same session are ranked. The highest raw score gets rank 1. Exact score ties share the average of the ranks they span. The ticker is not a tie-break.

```text
rs_proxy = half-up round of 1 + (n_eligible - average_rank) / (n_eligible - 1) * 98
```

Clipped to 1–99. Half-up is used on purpose: Python's `round` uses bankers' rounding (`round(74.5) == 74`).

`rs_proxy` is withheld when `n_eligible < 2` (a single name is not a percentile) or when `n_eligible` is below `--min-universe` (default 30). Returns and `raw_score` are still reported. `broad_market_rating` is true only for a `stocks` or `etfs` universe with at least 1000 eligible names. A watchlist percentile is not a broad-market rating. `unseparated` is never a broad-market rating.

### Eligibility

A symbol is eligible when all of the following hold:

- It belongs to the requested universe.
- One positive finite close exists on the as-of session and on each of the four anchor sessions.
- The 253-session window (the as-of session through `t-252`) is missing at most `--max-missing-sessions` prints (default 5).

| `reason` | Meaning |
| --- | --- |
| `ok` | Score computed. `rs_proxy` may still be empty if the universe is too small. |
| `short_history` | The 252-session anchor is before the symbol's first valid print, or the calendar itself is shorter than 253 sessions. |
| `missing_price` | The calendar has the anchor session, the symbol traded on both sides of it, and that session has no close. |
| `gappy` | Anchor math is still the NYSE date, but too many sessions in the window are missing, so `raw_score` is withheld. |
| `stale` | The last valid print is before the as-of session. The older close is not carried forward. |
| `missing_as_of` | No close on the as-of session and no earlier print either. |
| `invalid_price` | A required close is missing-as-a-number, non-finite, or `<= 0`. |
| `duplicate_session_prints` | Two different closes on the same NYSE session. Neither is chosen. |
| `crypto_excluded` | Crypto suffix (`USDT`/`BUSD`/`BTC`/`ETH`/`BNB`, same rule as `utils._tracked_symbols`) or `asset_class=cryptos`. Left out by default. |
| `outside_universe` | Classified stock in an ETF run, or the reverse. |
| `unclassified` | `--universe stocks` or `etfs` was requested and the symbol has no class. It is not treated as a stock. |
| `conflicting_asset_class` | More than one class was supplied for the ticker. |

If you ask for a date that is not a session (a Saturday, for example), the as-of session is the previous NYSE session when that session is actually in the file. That is not marked stale. If the newest print in the file is older than the NYSE session on or before `--as-of`, the run is `stale_as_of`: the score uses the newest session present, and the summary shows both dates.

## Universes

`--universe stocks` (the default) ranks `equities` / `stock` / `stocks` only. `--universe etfs` ranks `etf` / `etfs`. `--universe unseparated` ranks stocks and ETFs together and always warns that the result is not a separated rating.

Pass classes as an `asset_class` column on the CSV (`equities` or `etfs`, the same labels `extraction/metadata.py` writes) or as `--asset-class-csv` with columns `ticker,asset_class`. `--from-duckdb` also reads `data/metadata_consolidated.parquet` or `data/metadata.json` when those files are already on disk. It does not fetch them. An explicit `--asset-class-csv` overrides repo metadata per ticker. A plain `--input` file does not silently pick up repo metadata.

## Price source and adjustment

The calculator does not decide that a close is adjusted. The label you pass is copied onto every row.

| Input | What the artifact says |
| --- | --- |
| `--input` without `--adjustment` | `unknown`. The note says the file was **not** labeled split-adjusted or dividend-adjusted. |
| `--adjustment split` | Caller label only. Split-adjusted, not dividend-adjusted. Not checked against the vendor. |
| `--adjustment split_and_dividend` | Caller label only. Splits and dividends. Not checked. |
| `--from-duckdb` | Always `split_and_dividend`. Cannot be relabeled. |

`--from-duckdb` reads `data/ohlcv.duckdb` table `ohlcv_max_daily` with a read-only connection. That table is written by `extraction/ohlcv.get_max_data` via `yfinance.download(interval='1d', auto_adjust=True)`, which adjusts OHLC for **splits and dividends**. Those closes are not split-only, not raw, and not IBD's prices.

These tables are **not** read and are not resampled:

- `ohlcv_1min_airflow` — Alpaca `Adjustment.SPLIT`, intraday, incremental window.
- `massive_1min` — 1-minute bars requested with `adjusted=true` (vendor split adjustment).
- `robinhood_1min` — 1-minute `bounds=24_5`. This repo does not record an adjustment policy for them.

Daily bars in `ohlcv_max_daily` are stored at UTC midnight as a session label. The session date is that calendar date. A non-midnight timestamp is rejected so an intraday bar is not guessed into a session.

`data/` is gitignored. A fresh checkout has no `ohlcv.duckdb`. `python -m relative_strength coverage` prints that gap and does not create the file. Until that table is present, use `--input`.

## RS line

The RS line is `close / benchmark_close` on the same NYSE session. The default benchmark symbol is `SPY`. It is not `rs_proxy`. A missing benchmark does not block the percentile. Missing benchmark sessions stay empty; they are not forward-filled. The benchmark ticker itself is not given a line against itself. Pass `--benchmark-csv` (same `date,ticker,close` shape) when SPY is not in the price file. `--no-rs-line` skips the series.

The line uses the same adjustment label as the closes. A ratio is only as comparable as that shared label.

## Commands

From the repo root:

```bash
python -m relative_strength coverage
python -m relative_strength coverage --duckdb-path /path/to/ohlcv.duckdb

python -m relative_strength run \
  --input prices.csv \
  --as-of 2024-06-28 \
  --universe stocks \
  --adjustment unknown \
  --benchmark SPY \
  --out-dir output/relative_strength

python -m relative_strength run \
  --from-duckdb \
  --as-of 2024-06-28 \
  --universe stocks \
  --out-dir output/relative_strength
```

`--from-duckdb` exits with status 2 when the file or `ohlcv_max_daily` is missing. It does not download history. Status 0 means the artifact was written, including when every symbol is ineligible. Warnings go to stderr.

Exit status 2 is also used for a bad price file (intraday timestamps, missing columns) and for trying to relabel duckdb closes as `split`.

## Input schema

Canonical CSV columns:

```text
date,ticker,close,asset_class
```

`date` may be named `datetime` or `session_date`. `ticker` may be named `symbol`. `close` is case-insensitive. `asset_class` is optional. One close per ticker per session. Dates are `YYYY-MM-DD` or a midnight timestamp. Tickers are uppercased.

`asset_class` accepts `equities`, `stocks`, `etfs`, `cryptos` (and the obvious singular forms).

There is no P/E column and none is consulted.

## Output schema

New files, named with the session that was actually used:

- `output/relative_strength/rs_proxy_YYYY-MM-DD.csv`
- `output/relative_strength/rs_proxy_YYYY-MM-DD.json`
- `output/relative_strength/rs_line_YYYY-MM-DD.csv` when a benchmark close exists on the as-of session and `--no-rs-line` was not set

The JSON object is `summary`, `ratings`, and `rs_line`. `summary.method.official_ibd_rs` is false. `summary` also carries `as_of_requested`, `as_of_session`, `stale`, `eligible_count`, `lookback_anchor_sessions`, `adjustment`, `adjustment_note`, `price_source`, `benchmark_symbol`, `benchmark_available`, `broad_market_rating`, and `warnings`.

Rating columns, in order:

```text
as_of_session, as_of_requested, ticker, asset_class, universe,
r_63, r_126, r_189, r_252, raw_score, rs_proxy,
eligible, eligible_count, status, reason, reason_detail,
percentile_withheld_reason, missing_sessions_in_window, first_session, last_session,
price_source, adjustment, adjustment_note, session_calendar,
benchmark_symbol, benchmark_available, benchmark_note,
rating_label, broad_market_rating, run_warnings
```

`rating_label` is `transparent_proxy_not_official_ibd_rs`. Returns are decimals (`0.10` is 10 percent), not the 1–99 proxy.

RS-line columns:

```text
session_date, ticker, close, benchmark_symbol, benchmark_close, rs_line,
price_source, adjustment, note
```

`note` says the value is the close ratio and not the 1–99 percentile.

## Synthetic example

Not a market reading. Two names, one session, so the history is too short to score. The command still writes the schema and the reason:

```bash
mkdir -p /tmp/rs-synthetic
cat > /tmp/rs-synthetic/prices.csv << 'EOF'
date,ticker,close,asset_class
2024-06-28,AAA,10,equities
2024-06-28,SPY,500,etfs
EOF

python -m relative_strength run \
  --input /tmp/rs-synthetic/prices.csv \
  --as-of 2024-06-28 \
  --universe stocks \
  --adjustment unknown \
  --out-dir /tmp/rs-synthetic/out
```

`asset_class=equities` is stored on the rating row as `stocks`. `AAA` is `short_history` because the file has no close on the 252-session anchor (`2023-06-28` for this as-of). `r_63` stays empty; the anchor is not replaced by the only row in the file. `SPY` is `outside_universe` for a stock run and is still the RS-line benchmark on the one session that exists (`rs_line = 10/500 = 0.02`). `eligible_count` is 0. `adjustment` is `unknown`. `broad_market_rating` is false. The only run warning is `no_eligible_symbols`.

A 1–99 proxy needs at least 30 eligible stocks (the default) and 253 NYSE sessions of positive closes. `relative_strength/tests/test_calculator.py` builds that fixture. On it, with anchors at 100 and the as-of close at `100 + i`:

- `T30` and `T29` tie at raw score 0.30 and both receive `rs_proxy` 97 (average rank 1.5 of 30).
- `T28` (score 0.28, rank 3) receives 92.
- `T01` (score 0.01, rank 30) receives 1.
- An ETF with a much larger return, and `BTCUSDT`, do not enter that ranking.

The weighted-score fixture in the same file uses as-of close 140 and anchor closes 100, 100, 80, and 200:

```text
R_63 = 0.40, R_126 = 0.40, R_189 = 0.75, R_252 = -0.30
S = 0.40*0.40 + 0.20*0.40 + 0.20*0.75 + 0.20*(-0.30) = 0.33
```

## Programmatic use

```python
import pandas as pd
from relative_strength import compute_relative_strength

prices = pd.read_csv("prices.csv")  # date, ticker, close, asset_class
result = compute_relative_strength(
    prices,
    as_of="2024-06-28",
    universe="stocks",
    adjustment="unknown",
)
result.ratings.to_csv("rs_proxy.csv", index=False)
```

`compute_relative_strength` does not modify `prices`. Importing `relative_strength` does not open DuckDB or the network.

## Tests

```bash
python -m pytest relative_strength/tests -q
```

The fixtures are synthetic. They do not describe current stocks.
