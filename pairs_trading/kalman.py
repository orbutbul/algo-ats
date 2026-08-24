"""
pairs_trading/kalman.py — time-varying (Kalman-filtered) hedge ratio for a
pairs-trading pair, meant to feed a backtest that rebalances its hedge at
each bar rather than trading a single hedge_ratio fixed over the whole
window (pair_tests.py's Stage B hedge_ratio).

State-space model (Ernie Chan's pairs-trading Kalman filter): state
x_t = [alpha_t, beta_t] follows a random walk (F = I), and price_a is
observed as price_a_t = alpha_t + beta_t * price_b_t + eps_t. Process noise
is parameterized as Q = delta/(1-delta) * I_2 so `delta` is a single
interpretable "how fast can the hedge ratio drift" knob (0 = frozen OLS,
close to 1 = state chases every observation) instead of an arbitrary raw
covariance matrix.

No pykalman/filterpy dependency — this is a 2-state, one-observation-per-bar
filter, small enough to hand-roll in numpy (consistent with this package's
existing preference, e.g. pair_tests._rolling_corr_matrices, for direct
numpy/pandas over pulling in a library for something this size).
"""

import numpy as np
import pandas as pd

from pairs_trading.pair_tests import _filter_coverage, _load_group_daily_closes


def _kalman_filter_hedge_ratio(
    price_a: pd.Series,
    price_b: pd.Series,
    delta: float = 1e-4,
    obs_covariance: float = 1e-3,
    use_log: bool = False,
) -> pd.DataFrame:
    """
    Recursive Kalman filter over two already-aligned price Series, tracking
    a time-varying intercept/hedge-ratio state [alpha_t, beta_t] for
    price_a_t = alpha_t + beta_t * price_b_t + eps_t. Pure computation, no
    I/O — the single canonical implementation compute_kalman_hedge_ratio()
    calls, so any future batch/dashboard consumer reuses it rather than
    re-deriving the recursion.

    use_log=True regresses log(price_a) on log(price_b) instead of raw
    prices; default False matches pair_tests._pair_cointegration_stats'
    existing OLS-on-raw-price convention, so hedge ratios stay comparable
    across the static and time-varying computations.

    At each bar, the prediction error e_t = price_a_t - H_t @ x_pred is
    computed from x_pred (the state carried over from the prior bar, before
    today's observation is folded in) — this is the no-lookahead spread a
    backtest can trade on at that bar.

    x0 = [0, 0], P0 = I_2 (uninformative prior, deliberately not seeded from
    an external OLS estimate — consistent with Stage B not doing so either).

    Returns a DataFrame indexed like price_a/price_b, columns
    ['alpha', 'beta', 'spread', 'spread_var'] (spread_var = the innovation
    variance S_t at that bar, for a future z-scored entry/exit rule).
    """
    if use_log:
        price_a = np.log(price_a)
        price_b = np.log(price_b)

    n = len(price_a)
    a_vals = price_a.to_numpy()
    b_vals = price_b.to_numpy()

    Q = (delta / (1 - delta)) * np.eye(2)
    R = obs_covariance

    x = np.zeros(2)
    P = np.eye(2)

    alphas = np.empty(n)
    betas = np.empty(n)
    spreads = np.empty(n)
    spread_vars = np.empty(n)

    for t in range(n):
        # Predict (F = I, random-walk state)
        P_pred = P + Q

        # Update
        H = np.array([1.0, b_vals[t]])
        e = a_vals[t] - H @ x
        S = H @ P_pred @ H + R
        K = (P_pred @ H) / S

        x = x + K * e
        P = P_pred - np.outer(K, H) @ P_pred

        alphas[t] = x[0]
        betas[t] = x[1]
        spreads[t] = e
        spread_vars[t] = S

    return pd.DataFrame(
        {'alpha': alphas, 'beta': betas, 'spread': spreads, 'spread_var': spread_vars},
        index=price_a.index,
    )


def compute_kalman_hedge_ratio(
    ticker_a: str,
    ticker_b: str,
    delta: float = 1e-4,
    obs_covariance: float = 1e-3,
    use_log: bool = False,
    lookback_months: int | None = None,
) -> pd.DataFrame:
    """
    Self-contained entry point for one pair's time-varying hedge ratio —
    fetches its own daily closes from data/ohlcv.duckdb::massive_1min (same
    access pattern as utils.plot_pair_dashboard), then runs
    _kalman_filter_hedge_ratio over them.

    lookback_months restricts to the trailing N months of price history,
    anchored to the latest date present (same convention as
    pair_tests._pair_cointegration_stats); None (default) uses the full
    available history — the filter's own state naturally "forgets" old
    observations via delta, so unlike the static OLS hedge ratio there's no
    strict need to window the input.

    Raises ValueError if coverage filtering excludes either ticker (mirrors
    plot_pair_dashboard's explicit check) — a Kalman filter over a pair
    with a data gap would silently jump across it.

    Returns _kalman_filter_hedge_ratio's output: a DataFrame indexed by
    date, columns ['alpha', 'beta', 'spread', 'spread_var']. To get the
    hedge ratio at a specific time, index into it directly, e.g.
    compute_kalman_hedge_ratio('AAPL', 'MSFT').loc[some_date, 'beta'].
    """
    daily_close = _load_group_daily_closes([ticker_a, ticker_b])
    daily_close, excluded = _filter_coverage(daily_close)
    if excluded:
        raise ValueError(f'Insufficient coverage for: {excluded} — cannot compute a Kalman hedge ratio.')

    if lookback_months is not None:
        cutoff = daily_close.index.max() - pd.DateOffset(months=lookback_months)
        daily_close = daily_close[daily_close.index >= cutoff]

    return _kalman_filter_hedge_ratio(
        daily_close[ticker_a], daily_close[ticker_b], delta, obs_covariance, use_log,
    )
