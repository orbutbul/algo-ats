"""
pairs_trading/pair_tests.py — Stage A/B pair-test series for pairs-trading
candidate groups (from pairs_trading/utils.py's find_pair_groups()/
refine_groups()). One function per test, plain-function + orchestrator style
(matching pairs_filter.compare_pair() in this same package) — no class
hierarchy. compute_correlation_prefilter() is Stage A: a stability-aware
rolling-correlation prefilter on daily-resampled returns, meant to narrow a
group down before compute_cointegration_test() (Stage B): an Engle-Granger
cointegration test (statsmodels.tsa.stattools.coint) on Stage A's surviving
pairs, further filtered by how often the resulting spread actually
mean-reverts.

All OHLCV in this package is sourced from data/ohlcv.duckdb::massive_1min via
utils.get_data_duckdb() — the repo-wide established access pattern — never a
standalone file, so every consumer here stays consistent with the rest of
the codebase as that table gets refreshed/backfilled.
"""

import hashlib
import itertools
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from statsmodels.tsa.stattools import coint

from utils import get_data_duckdb

# Minimum overlapping daily observations required to attempt a cointegration
# test at all — below this, Engle-Granger's asymptotics aren't meaningful.
_MIN_COINT_OBS = 30

_RESULT_COLUMNS = ['ticker_a', 'ticker_b', 'latest_corr', 'min_corr', 'std_corr', 'passed']

# Minimum fraction of the group's shared date range a ticker must have
# non-null daily closes for to stay in the pairing pool. Below this, a
# ticker is excluded entirely rather than forward-filled — forward-filling
# would fabricate zero-return days and bias correlation upward (same
# reasoning gap_detection.py already applies to 1-min completeness).
_MIN_COVERAGE = 0.9


def _load_group_daily_closes(group_tickers: list[str]) -> pd.DataFrame:
    """
    Loads daily-resampled close prices for group_tickers from
    data/ohlcv.duckdb::massive_1min via utils.get_data_duckdb(freq='daily'),
    pivoted to a wide (date x ticker) DataFrame. A date/ticker combination
    with no bars that day comes back NaN (get_data_duckdb's per-ticker
    resample already drops empty days rather than fabricating them) — this
    is the "outer join" alignment _filter_coverage() below needs, for free.
    """
    df = get_data_duckdb(tickers=group_tickers, freq='daily', asdf=True)
    if df.empty:
        raise ValueError(f'No data found in massive_1min for any of: {group_tickers}')

    close_wide = df['Close'].unstack('ticker')

    missing = [t for t in group_tickers if t not in close_wide.columns]
    if missing:
        raise ValueError(f'Tickers not found in massive_1min: {missing}')

    return close_wide[group_tickers]


def _filter_coverage(daily_close: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Excludes any ticker whose non-null coverage over the group's shared
    date range falls below _MIN_COVERAGE (new listing, long halt, etc.)
    rather than forward-filling it.

    Returns (daily_close restricted to surviving tickers, dropna'd to only
    the dates where every surviving ticker has data; excluded_tickers).
    """
    coverage = daily_close.notna().mean()
    excluded = coverage[coverage < _MIN_COVERAGE].index.tolist()
    for t in excluded:
        n_present = int(daily_close[t].notna().sum())
        print(f'{t}: only {n_present}/{len(daily_close)} trading days in window — excluded')

    survivors = [t for t in daily_close.columns if t not in excluded]
    filtered = daily_close[survivors].dropna(how='any')
    return filtered, excluded


def _rolling_corr_matrices(
    returns: pd.DataFrame, corr_window: int, recompute_freq: str,
) -> dict[pd.Timestamp, pd.DataFrame]:
    """
    Computes the full pairwise correlation matrix at each recompute_freq
    period end, over the trailing corr_window days ending at that date.
    Periods without a full corr_window of history yet are skipped. Uses
    pandas' own .corr() (vectorized across all column-pairs in one call) —
    the right tool for a periodic full-matrix snapshot, as opposed to
    vbt.rolling_corr's two-array continuous-series primitive.
    """
    index_series = returns.index.to_series()
    period_ends = index_series.resample(recompute_freq).apply(lambda s: s.max() if len(s) else pd.NaT)
    period_ends = period_ends.dropna().tolist()

    matrices = {}
    for date in period_ends:
        window = returns.loc[:date].tail(corr_window)
        if len(window) < corr_window:
            continue
        matrices[date] = window.corr()
    return matrices


def _cache_key(group_tickers: list[str], corr_window: int, recompute_freq: str) -> str:
    raw = f'{sorted(group_tickers)}|{corr_window}|{recompute_freq}'
    return hashlib.sha1(raw.encode()).hexdigest()


def _pair_correlation_stats(
    a: str, b: str, matrices: dict[pd.Timestamp, pd.DataFrame], dates: list[pd.Timestamp],
    stability_lookback: int, corr_threshold: float, stability_std_cap: float,
) -> dict:
    """
    The correlation-stability check for one pair against an already-computed
    set of rolling correlation matrices — the single canonical
    implementation compute_correlation_prefilter()'s loop and
    utils.plot_pair_dashboard() both call, rather than each re-deriving it.
    Returns {'latest_corr', 'min_corr', 'std_corr', 'passed'}.
    """
    readings = [matrices[d].loc[a, b] for d in dates[-stability_lookback:]
                if a in matrices[d].index and b in matrices[d].columns]

    if not readings or any(pd.isna(readings)):
        valid = [r for r in readings if pd.notna(r)]
        latest_corr = valid[-1] if valid else np.nan
        min_corr = min(valid) if valid else np.nan
        std_corr = float(np.std(valid)) if len(valid) > 1 else np.nan
        passed = False  # NaNs/insufficient readings propagate as failed, never silently dropped
    else:
        latest_corr = readings[-1]
        min_corr = min(readings)
        std_corr = float(np.std(readings))
        passed = bool(min_corr >= corr_threshold and std_corr <= stability_std_cap)

    return {'latest_corr': latest_corr, 'min_corr': min_corr, 'std_corr': std_corr, 'passed': passed}


def compute_correlation_prefilter(
    group_tickers: list[str],
    corr_window: int = 75,
    recompute_freq: str = 'W',
    stability_lookback: int = 6,
    corr_threshold: float = 0.65,
    stability_std_cap: float = 0.15,
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """
    Stage A of the pairs-trading pipeline. Loads daily-resampled closes for
    group_tickers from data/ohlcv.duckdb::massive_1min (via
    utils.get_data_duckdb()), computes a rolling pairwise correlation matrix
    on daily log returns (recomputed every `recompute_freq`, trailing
    `corr_window` days each time), and for every unique surviving pair
    checks the trailing `stability_lookback` readings for both a correlation
    floor (`corr_threshold`) and stability (`stability_std_cap` on those
    readings' stddev).

    group_tickers below 4 members returns an empty (correctly-columned)
    DataFrame — too small to pair, not an error. A ticker not found in
    massive_1min raises ValueError naming it. A ticker with insufficient
    daily-close coverage over the group's shared date range (new listing,
    long halt) is excluded from pairing entirely (logged by name/reason),
    never forward-filled.

    Returns a DataFrame — one row per unique pair among surviving tickers
    (not just passing ones) — columns: ticker_a, ticker_b, latest_corr,
    min_corr, std_corr, passed.
    """
    if len(group_tickers) < 4:
        print(f'group_tickers has {len(group_tickers)} members (<4) — skipping, nothing to pair.')
        return pd.DataFrame(columns=_RESULT_COLUMNS)

    cache_path = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f'{_cache_key(group_tickers, corr_window, recompute_freq)}.pkl'
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                matrices = pickle.load(f)
            # Preserve group_tickers' original order (matching the
            # non-cached path below) so ticker_a/ticker_b role assignment
            # and row order are identical whether or not the cache hits.
            present = {t for m in matrices.values() for t in m.columns}
            surviving_tickers = [t for t in group_tickers if t in present]
        else:
            matrices = None

    if cache_path is None or matrices is None:
        daily_close = _load_group_daily_closes(group_tickers)
        daily_close, excluded = _filter_coverage(daily_close)
        surviving_tickers = [t for t in group_tickers if t not in excluded]

        if len(surviving_tickers) < 4:
            print(f'Only {len(surviving_tickers)} tickers survived coverage filtering (<4) — skipping.')
            return pd.DataFrame(columns=_RESULT_COLUMNS)

        returns = np.log(daily_close / daily_close.shift(1)).iloc[1:]
        matrices = _rolling_corr_matrices(returns, corr_window, recompute_freq)

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(matrices, f)

    dates = sorted(matrices.keys())

    rows = []
    for a, b in itertools.combinations(surviving_tickers, 2):
        stats = _pair_correlation_stats(a, b, matrices, dates, stability_lookback, corr_threshold, stability_std_cap)
        rows.append({'ticker_a': a, 'ticker_b': b, **stats})

    return pd.DataFrame(rows, columns=_RESULT_COLUMNS)


_COINT_RESULT_COLUMNS = [
    'ticker_a', 'ticker_b', 'pvalue_ab', 'pvalue_ba', 'cointegrated',
    'hedge_ratio', 'n_crossings', 'passed',
]


def _crossing_signs(spread: pd.Series) -> np.ndarray:
    """Sign of `spread` at each point, with exact zeros treated as a
    continuation of the prior sign (so a spread landing exactly on zero
    isn't its own crossing)."""
    signs = np.sign(spread.to_numpy())
    last = 1
    for i, s in enumerate(signs):
        if s == 0:
            signs[i] = last
        else:
            last = s
    return signs


def _count_crossings(spread: pd.Series) -> int:
    """Counts sign changes in `spread` — see _crossing_signs for the
    zero-handling rule."""
    signs = _crossing_signs(spread)
    return int(np.sum(signs[1:] != signs[:-1]))


def _crossing_dates(spread: pd.Series) -> pd.DatetimeIndex:
    """Dates where `spread`'s sign flips (the later of each flipping pair —
    matches _count_crossings' signs[1:] != signs[:-1] comparison)."""
    signs = _crossing_signs(spread)
    flips = np.flatnonzero(signs[1:] != signs[:-1]) + 1
    return spread.index[flips]


def _threshold_crossings(series: pd.Series, level: float, direction: str) -> pd.Series:
    """
    Boolean mask, indexed like `series`, True at points where `series`
    crosses `level` in `direction` ('down': prior bar >= level, this bar <
    level; 'up': prior bar <= level, this bar > level) — the single
    canonical crossing-direction test for threshold-based entry/exit
    signals (e.g. a z-score crossing +/-z_entry / +/-z_exit), so a caller
    needing the same event on two different series (e.g. plotting both the
    z-score panel and a synthetic spread-price panel's entry/exit markers
    at the same timestamps) computes it once and reuses the mask rather
    than re-deriving it per plot.
    """
    prev = series.shift(1)
    if direction == 'down':
        return (prev >= level) & (series < level)
    if direction == 'up':
        return (prev <= level) & (series > level)
    raise ValueError(f"direction must be 'up' or 'down', got {direction!r}")


def _pair_cointegration_stats(
    price_a: pd.Series, price_b: pd.Series, coint_pvalue_threshold: float, min_crossings: int,
    lookback_months: int = 6,
) -> dict:
    """
    The Engle-Granger cointegration + spread-crossing check for one pair of
    already-aligned price series — the single canonical implementation
    compute_cointegration_test()'s loop and utils.plot_pair_dashboard() both
    call, rather than each re-deriving it.

    Restricted to the trailing `lookback_months` of the given price history
    (anchored to the latest date present, not necessarily today, in case a
    caller passes slightly-stale data) — cointegration/hedge-ratio/crossing
    behavior can drift over a multi-year sample, so this tests recent
    regime behavior rather than the pair's entire history. Both callers get
    this for free since it lives here, not duplicated in each.

    Pairs with fewer than _MIN_COINT_OBS overlapping observations within
    that window aren't tested (Engle-Granger's asymptotics aren't
    meaningful with too little data) — reported with cointegrated=False,
    passed=False, NaN stats, not silently dropped.

    Returns {'pvalue_ab', 'pvalue_ba', 'cointegrated', 'hedge_ratio',
    'n_crossings', 'passed'}.
    """
    def _empty():
        return {
            'pvalue_ab': np.nan, 'pvalue_ba': np.nan, 'cointegrated': False,
            'hedge_ratio': np.nan, 'n_crossings': 0, 'passed': False,
        }

    if len(price_a) == 0:
        return _empty()

    cutoff = price_a.index.max() - pd.DateOffset(months=lookback_months)
    price_a = price_a[price_a.index >= cutoff]
    price_b = price_b[price_b.index >= cutoff]

    if len(price_a) < _MIN_COINT_OBS:
        print(f'{price_a.name}/{price_b.name}: only {len(price_a)} observations in the trailing '
              f'{lookback_months} months (<{_MIN_COINT_OBS}) — skipping cointegration test.')
        return _empty()

    _, pvalue_ab, _ = coint(price_a, price_b)
    _, pvalue_ba, _ = coint(price_b, price_a)
    cointegrated = bool(pvalue_ab < coint_pvalue_threshold and pvalue_ba < coint_pvalue_threshold)

    ols = sm.OLS(price_a, sm.add_constant(price_b)).fit()
    hedge_ratio = float(ols.params.iloc[1])
    n_crossings = _count_crossings(ols.resid)

    passed = bool(cointegrated and n_crossings > min_crossings)

    return {
        'pvalue_ab': pvalue_ab, 'pvalue_ba': pvalue_ba, 'cointegrated': cointegrated,
        'hedge_ratio': hedge_ratio, 'n_crossings': n_crossings, 'passed': passed,
    }


def compute_cointegration_test(
    stage_a_result: pd.DataFrame,
    coint_pvalue_threshold: float = 0.05,
    min_crossings: int = 6,
    lookback_months: int = 6,
) -> pd.DataFrame:
    """
    Stage B of the pairs-trading pipeline. Takes compute_correlation_prefilter()'s
    output directly, tests every pair with passed == True for cointegration
    via statsmodels.tsa.stattools.coint on the trailing `lookback_months` of
    daily close prices (loaded from data/ohlcv.duckdb::massive_1min, same as
    Stage A) — recent regime behavior, not the pair's entire multi-year
    history, which cointegration/hedge-ratio/crossing behavior can drift
    across.

    Engle-Granger's coint() isn't symmetric — regressing A on B vs. B on A
    can give different p-values — so both directions are tested, and a pair
    is only `cointegrated` if BOTH clear coint_pvalue_threshold (stricter
    than either direction alone, which can be a regression-direction
    artifact rather than real cointegration).

    For every pair (cointegrated or not), also builds the spread from the
    a~b OLS regression's residual (mean zero by construction) and counts
    `n_crossings` — how many times that residual's sign flips over the
    windowed price history, a proxy for how often the pair actually
    mean-reverted (a cointegrated-but-flat spread offers little to trade).
    `passed` requires both `cointegrated` and n_crossings > min_crossings.

    Pairs with fewer than 30 observations within the lookback window aren't
    tested (Engle-Granger's asymptotics aren't meaningful with too little
    data) — reported with cointegrated=False, passed=False, and NaN stats
    rather than silently dropped.

    Returns one row per input pair — columns: ticker_a, ticker_b,
    pvalue_ab, pvalue_ba, cointegrated, hedge_ratio, n_crossings, passed.
    """
    candidates = stage_a_result[stage_a_result['passed']]
    if candidates.empty:
        return pd.DataFrame(columns=_COINT_RESULT_COLUMNS)

    pairs = list(candidates[['ticker_a', 'ticker_b']].itertuples(index=False, name=None))
    unique_tickers = sorted({t for pair in pairs for t in pair})
    daily_close = _load_group_daily_closes(unique_tickers)

    rows = []
    for a, b in pairs:
        aligned = daily_close[[a, b]].dropna(how='any')
        stats = _pair_cointegration_stats(aligned[a], aligned[b], coint_pvalue_threshold, min_crossings, lookback_months)
        rows.append({'ticker_a': a, 'ticker_b': b, **stats})

    return pd.DataFrame(rows, columns=_COINT_RESULT_COLUMNS)


def _full_sample_ols_beta(price_a: pd.Series, price_b: pd.Series) -> float:
    """
    Full-sample OLS hedge ratio (price_a ~ const + price_b) — the flat,
    static-beta baseline used when a caller wants "one number for the
    whole window" rather than a rolling refit (e.g.
    utils.plot_pair_diagnostics_dashboard's static_beta_window=None case).
    Same regression _pair_cointegration_stats already runs internally, just
    exposed standalone so callers that only want the beta (no cointegration
    test) don't need to run one.
    """
    ols = sm.OLS(price_a, sm.add_constant(price_b)).fit()
    return float(ols.params.iloc[1])


def _rolling_ols_beta(price_a: pd.Series, price_b: pd.Series, window: int) -> pd.Series:
    """
    Rolling `window`-day OLS hedge ratio (price_a ~ const + price_b), via
    statsmodels' RollingOLS — the periodically-refit static-beta
    counterpart to _kalman_filter_hedge_ratio's continuously-updated
    beta_t, for visualizing how much the time-varying filter actually buys
    over a plain rolling regression. Bars before the first full `window`
    of history come back NaN (RollingOLS's own convention).
    """
    exog = sm.add_constant(price_b)
    fit = RollingOLS(price_a, exog, window=window).fit(params_only=True)
    return fit.params.iloc[:, 1]


def _rolling_pair_correlation(returns_a: pd.Series, returns_b: pd.Series, window: int) -> pd.Series:
    """
    Rolling `window`-day Pearson correlation between two daily log-return
    series — the continuous, single-pair counterpart to
    _rolling_corr_matrices' periodic full-matrix snapshots, for a per-bar
    diagnostic line rather than a periodic Stage-A-style recompute.
    """
    return returns_a.rolling(window).corr(returns_b)


def _rolling_cointegration_stat(
    price_a: pd.Series, price_b: pd.Series, window: int, stride: int = 5,
) -> tuple[pd.Series, float]:
    """
    Rolling Engle-Granger cointegration test statistic
    (statsmodels.tsa.stattools.coint) over a trailing `window`-day window,
    computed only every `stride` days and forward-filled in between.

    Tradeoff, not a silent shortcut: coint() runs an OLS fit plus an ADF
    test per call, expensive enough that recomputing it on every single bar
    of a multi-year daily series is wasteful for what is purely a visual
    diagnostic here — `stride` names that tradeoff explicitly so a caller
    can dial it to 1 for full resolution if they need it.

    Returns (test-statistic series indexed like price_a/price_b, NaN before
    the first full window and forward-filled between stride points; the 5%
    critical value for this window, which — unlike the statistic itself —
    is constant across every stride point since it depends only on `window`
    and coint()'s default trend spec, so it's returned once rather than as
    a series).
    """
    idx = price_a.index
    stat = pd.Series(np.nan, index=idx)
    crit_5pct = np.nan
    for i in range(window - 1, len(idx), stride):
        a_window = price_a.iloc[i - window + 1:i + 1]
        b_window = price_b.iloc[i - window + 1:i + 1]
        test_stat, _, crit_values = coint(a_window, b_window)
        stat.iloc[i] = test_stat
        crit_5pct = crit_values[1]  # coint()'s crit_value order is [1%, 5%, 10%]
    return stat.ffill(), crit_5pct


def _hurst_exponent(series: np.ndarray, max_lag: int = 20) -> float:
    """
    Generalized Hurst exponent via the classic lag-variance method
    (log-log regression of the standard deviation of lagged differences
    against the lag) — the standard pairs-trading convention (Ernie Chan),
    hand-rolled in numpy rather than pulling in the `hurst` package for one
    small computation (consistent with this package's existing preference,
    e.g. kalman._kalman_filter_hedge_ratio, for direct numpy over a library
    dependency for something this size).

    H < 0.5 indicates mean reversion, H = 0.5 a random walk, H > 0.5
    trending. Not meaningful on a window with near-zero variance (a
    constant `series` slice would take log(0)) — an inherent limitation of
    the method, not guarded here since the rolling caller already requires
    a real spread series.
    """
    lags = range(2, max_lag)
    tau = [np.std(series[lag:] - series[:-lag]) for lag in lags]
    slope = np.polyfit(np.log(list(lags)), np.log(tau), 1)[0]
    return slope * 2.0


def _rolling_hurst_exponent(series: pd.Series, window: int, max_lag: int = 20) -> pd.Series:
    """
    Rolling Hurst exponent of `series` over a trailing `window`-bar window,
    via _hurst_exponent. Bars before the first full `window` of history
    come back NaN.
    """
    return series.rolling(window).apply(lambda w: _hurst_exponent(w.to_numpy(), max_lag), raw=False)
