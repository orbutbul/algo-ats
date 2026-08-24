# pairs_trading/ — reference

Statistical arbitrage / pairs-trading pipeline: buckets the equity+ETF universe
into candidate groups, runs a two-stage filter (rolling-correlation prefilter
→ Engle-Granger cointegration) to find tradeable pairs, and provides both a
static (OLS) and time-varying (Kalman) hedge ratio for the survivors.

All OHLCV comes from `data/ohlcv.duckdb::massive_1min` via the repo-wide
`utils.get_data_duckdb()` — no file here loads price data any other way.
Universe/fundamentals metadata comes from `data/metadata.json` (yfinance dump,
equities + ETFs only; no crypto).

## Pipeline order

```
utils.find_pair_groups()      sector → industry → [tickers], from metadata.json
        ↓
utils.refine_groups()         splits industry buckets >15 tickers via TF-IDF
        ↓                     agglomerative clustering (description similarity)
   refined_groups.json        (274 industry groups, 11 sectors, 1996 tickers)
        ↓
pipeline.find_candidate_pairs()   drives pair_tests Stage A → Stage B per group
        ↓
   candidate_pairs.csv         (4504 pairs, 25 passed both stages — full ranked table)
```

Separately, `kalman.py` and `utils.plot_kalman_dashboard()` give a per-pair
time-varying hedge ratio once you already know which pair you want to trade.
`pairs_filter.py` is an independent, older per-symbol scorer (see below) — not
wired into the pipeline above.

## Files

### `utils.py` — grouping + dashboards
- `find_pair_groups(metadata_path=None) -> dict` — `{sector: {'etfs': [...],
  'industries': {industry: [tickers]}}}`. Equities grouped by (sector,
  industry); ETFs attached at sector level only via `_SECTOR_ALIASES`
  (Morningstar category → sector, only for categories that map cleanly).
  Crypto excluded (not in metadata.json).
- `refine_groups(groups=None, max_group_size=15, metadata=None) -> dict` —
  same shape, but splits any industry bucket over `max_group_size` into
  `'{industry} (1)'`, `'{industry} (2)'`, ... via recursive agglomerative
  clustering (`_cluster_bucket`/`_cluster_once`) on TF-IDF cosine distance
  between `longBusinessSummary` text, 'complete' linkage (more size-balanced
  than 'average', verified empirically). ETFs untouched (no business summary
  to cluster by). This is what produced `refined_groups.json`.
- `plot_pair_dashboard(ticker_a, ticker_b, ...) -> vbt Figure` — cumulative
  log-return chart for one pair with crossing markers, title annotation shows
  Stage A + Stage B stats computed fresh for just this pair (same helpers the
  batch pipeline uses, not re-derived).
- `plot_kalman_dashboard(ticker_a, ticker_b, delta=1e-4, obs_covariance=1e-3,
  lookback_months=None) -> plotly Figure` — 3-panel dashboard (log prices /
  Kalman beta_t / hedge-adjusted log price). Opens in browser via
  `fig.show(renderer='browser')` as a side effect. Panel 3 uses the
  post-observation beta — fine to look at, **not** lookahead-safe for a
  backtest (use `_kalman_filter_hedge_ratio`'s `spread` column for that).

### `pair_tests.py` — Stage A / Stage B core logic
- `compute_correlation_prefilter(group_tickers, corr_window=75,
  recompute_freq='W', stability_lookback=6, corr_threshold=0.65,
  stability_std_cap=0.15, cache_dir=None) -> DataFrame` — **Stage A**. Loads
  daily closes, excludes tickers below `_MIN_COVERAGE=0.9` non-null coverage
  (never forward-fills), computes rolling pairwise correlation matrices
  (recomputed every `recompute_freq`, trailing `corr_window` days), and for
  every pair checks the trailing `stability_lookback` readings against a
  correlation floor + stddev cap. Returns one row per pair (not just passing
  ones): `ticker_a, ticker_b, latest_corr, min_corr, std_corr, passed`.
  <4 tickers (before or after coverage filtering) → empty result, not error.
- `compute_cointegration_test(stage_a_result, coint_pvalue_threshold=0.05,
  min_crossings=6, lookback_months=6) -> DataFrame` — **Stage B**. Takes only
  Stage A's `passed==True` rows, tests both regression directions with
  `statsmodels.tsa.stattools.coint` (a pair is `cointegrated` only if BOTH
  directions clear the p-value threshold), fits OLS for the static
  `hedge_ratio`, counts spread sign-flips (`n_crossings`) over the trailing
  `lookback_months`. `passed = cointegrated and n_crossings > min_crossings`.
  <30 overlapping observations in the lookback window → reported as
  not-cointegrated/not-passed with NaN stats, not dropped.
- Shared internals (imported elsewhere in the package, not private in
  practice): `_load_group_daily_closes`, `_filter_coverage`,
  `_rolling_corr_matrices`, `_pair_correlation_stats` (Stage A logic for one
  pair — also used by `plot_pair_dashboard`), `_pair_cointegration_stats`
  (Stage B logic for one pair — also used by `plot_pair_dashboard`),
  `_count_crossings`/`_crossing_dates`/`_crossing_signs`.

### `pipeline.py` — batch driver across all groups
- `find_candidate_pairs(groups=None, ...corr/coint params..., cache_dir=None,
  output_path=CANDIDATE_PAIRS_PATH, max_groups=None) -> DataFrame` — runs
  Stage A → Stage B over every industry group in `refined_groups.json` (or a
  passed-in groups dict of the same shape). Sector ETFs are **not** included
  — equities-only.
  - **Resumable**: appends each finished group's rows to
    `.candidate_pairs_raw.csv` immediately (not held in memory — 274 groups
    of DB round-trips can exceed an external time limit), and records every
    attempted (sector, group) — even zero-row ones — in
    `.processed_groups.csv` so re-invocation skips already-done groups.
    `max_groups` caps how many *new* groups one call processes, for chunking
    a full run across several invocations.
  - Output `candidate_pairs.csv`: full ranked table, regenerated (not
    appended) each call from the raw store. Columns: `rank, sector, group,
    ticker_a, ticker_b, latest_corr, min_corr, std_corr, passed_stage_a,
    pvalue_ab, pvalue_ba, cointegrated, hedge_ratio, n_crossings,
    passed_stage_b`. `rank`: Stage-B-passed pairs first, then by
    `n_crossings` descending (position, not a score).
  - `.candidate_pairs_raw.csv` / `.processed_groups.csv` are gitignored
    internal bookkeeping (see `.gitignore`) — `candidate_pairs.csv` is the
    real output artifact.
  - `python -m pairs_trading.pipeline` runs the full sweep with defaults.

### `kalman.py` — time-varying hedge ratio
- `_kalman_filter_hedge_ratio(price_a, price_b, delta=1e-4,
  obs_covariance=1e-3, use_log=False) -> DataFrame` — hand-rolled (no
  pykalman) 2-state Kalman filter, state `[alpha_t, beta_t]` random walk
  (`price_a_t = alpha_t + beta_t * price_b_t + eps_t`), `Q =
  delta/(1-delta) * I` so `delta` is a single "how fast can hedge ratio
  drift" knob. Returns `alpha, beta, spread, spread_var` per bar — `spread`
  is the **pre-observation** prediction error, i.e. the lookahead-safe series
  to actually trade/backtest on.
- `compute_kalman_hedge_ratio(ticker_a, ticker_b, delta=1e-4,
  obs_covariance=1e-3, use_log=False, lookback_months=None) -> DataFrame` —
  self-contained entry point, fetches its own daily closes, raises
  `ValueError` if coverage filtering would exclude either ticker (a gap would
  make the filter silently jump across it).

### `pairs_filter.py` — independent per-symbol scorer (not in the pipeline)
Complements `find_pair_groups()`: instead of bucketing the whole universe,
scores **one symbol** against the rest of `metadata.json`.
- `compare_pair(sym1, sym2, metadata=None) -> dict` — `structural_score`
  (industry/sector exact-match for equities, weighted 0.7/0.3; category
  exact-match for ETFs) + `description_score` (TF-IDF cosine similarity of
  `longBusinessSummary`) → `combined_score = 0.6*structural + 0.4*description`
  (or just `description_score` for cross-asset-class pairs, where structural
  fields don't overlap at all).
- `find_pairs(symbol, metadata=None, min_combined=0.3, same_class_only=False)
  -> DataFrame` — ranks every other ticker in metadata.json against `symbol`
  by the same combined score.
- Has its own `__main__` demo (NVDA vs AMD, top pairs for NVDA).

### Data artifacts
- `refined_groups.json` — output of `refine_groups()`: 11 sectors, 274
  industry sub-groups, 1996 tickers total. Input to `pipeline.py`.
- `candidate_pairs.csv` — output of `find_candidate_pairs()`: 4504 candidate
  pairs across all groups, 25 with `passed_stage_b == True` (the actual
  tradeable candidates).
- `.candidate_pairs_raw.csv`, `.processed_groups.csv` — gitignored,
  resumability bookkeeping for `pipeline.py` only. Don't read these for
  results — use `candidate_pairs.csv`.
- `temp_pairs_test.ipynb` — scratch/sanity-check notebook, one section per
  module (`find_pair_groups`, `pairs_filter`, Stage A, Stage B,
  `plot_pair_dashboard`). Exploratory, not a deliverable.

## Conventions worth knowing before editing

- **No forward-filling ever.** Insufficient coverage → exclude the ticker
  (`_filter_coverage`, `_MIN_COVERAGE=0.9`) rather than fabricate zero-return
  days. Same reasoning as `gap_detection.py` elsewhere in the repo.
- **Per-pair "canonical implementation" pattern**: the single-pair stats
  logic (`_pair_correlation_stats`, `_pair_cointegration_stats`) lives once
  in `pair_tests.py` and is called by both the batch loop and the dashboard
  functions in `utils.py` — never re-derived.
- **Windowed cointegration, not full history**: Stage B and the Kalman
  dashboard both restrict to a trailing lookback (default 6 months) anchored
  to the latest date *in the data*, not `today()` — cointegration/hedge
  behavior drifts over multi-year samples.
- **Engle-Granger is direction-asymmetric** — always test both A~B and B~A,
  require both to pass.
- **Return "everything, not just passing rows"** is a hard convention across
  Stage A, Stage B, and the pipeline output — filtering is a column
  (`passed`/`passed_stage_a`/`passed_stage_b`), never a row-drop, so
  near-misses stay inspectable.
