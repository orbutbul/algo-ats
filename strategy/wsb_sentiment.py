"""
strategy/wsb_sentiment.py — Strategy that allocates to tickers currently
showing positive WallStreetBets sentiment (extraction/wsb.py's hourly
mentions_v2 snapshots in data/wsb.duckdb), sized by how long each ticker has
stayed positive and how strong its current sentiment reading is.

Signal source is the WSB duckdb, not the OHLCV `data` passed into on_data() —
that's only used to restrict the target universe to symbols we can actually
trade (columns present in bars).
"""

from __future__ import annotations

import pandas as pd

from extraction.wsb import load_wsb_data
from strategy.base import Strategy
from strategy.types import Decision, PortfolioState
from strategy.utils import weights_to_decisions
from utils import compute_allocations


def compute_wsb_sentiment_signal(
    mentions: pd.DataFrame,
    bull_threshold: float = 50.0,
    lookback_hours: int = 24 * 14,
) -> pd.Series:
    """
    mentions: extraction.wsb.load_wsb_data('mentions_v2') shape — one row per
    (date, hour, ticker) with a bull_pct column (% of mentions that are
    bullish for that ticker that hour).

    Returns ticker -> signal strength, for tickers whose *latest* snapshot is
    positive (bull_pct > bull_threshold) only. Signal = consecutive positive
    hours running up to the latest snapshot, times how far above the
    threshold the current reading is -- a ticker that has stayed bullish
    longer and more strongly gets a bigger allocation. A ticker missing from
    an hour's ranked list breaks its streak, since we can't tell whether it
    stayed positive that hour.
    """
    if mentions.empty:
        return pd.Series(dtype=float)

    m = mentions.dropna(subset=['bull_pct']).copy()
    m['timestamp'] = pd.to_datetime(m['date']) + pd.to_timedelta(m['hour'], unit='h')

    cutoff = m['timestamp'].max() - pd.Timedelta(hours=lookback_hours)
    m = m[m['timestamp'] >= cutoff]

    all_hours = sorted(m['timestamp'].unique())
    hour_index = {h: i for i, h in enumerate(all_hours)}
    latest_hour = all_hours[-1]

    signal = {}
    for ticker, g in m.groupby('ticker'):
        g = g.sort_values('timestamp')
        latest = g.iloc[-1]
        if latest['timestamp'] != latest_hour or latest['bull_pct'] <= bull_threshold:
            continue  # not currently positive -- no allocation

        positive_hours = {
            hour_index[t] for t, bp in zip(g['timestamp'], g['bull_pct']) if bp > bull_threshold
        }
        streak, cursor = 0, hour_index[latest_hour]
        while cursor in positive_hours:
            streak += 1
            cursor -= 1

        strength = latest['bull_pct'] - bull_threshold
        signal[ticker] = streak * strength

    return pd.Series(signal, dtype=float)


class WsbSentimentStrategy(Strategy):
    """
    Ranks tickers by compute_wsb_sentiment_signal() and rebalances toward the
    resulting target-weight vector on every on_data() call.
    """

    def __init__(
        self,
        top_k: int = 5,
        bull_threshold: float = 50.0,
        lookback_hours: int = 24 * 14,
        allocation_method: str = 'rank',
    ):
        self.top_k = top_k
        self.bull_threshold = bull_threshold
        self.lookback_hours = lookback_hours
        self.allocation_method = allocation_method

    def on_data(self, data: dict[str, pd.DataFrame], portfolio: PortfolioState) -> list[Decision]:
        mentions = load_wsb_data('mentions_v2')
        signal = compute_wsb_sentiment_signal(mentions, self.bull_threshold, self.lookback_hours)
        if signal.empty:
            return []

        signal = signal[signal.index.isin(data.keys())]  # only trade symbols we have bars for
        if signal.empty:
            return []

        weights = compute_allocations(
            pd.DataFrame([signal]), method=self.allocation_method, top_k=self.top_k,
        ).iloc[-1]
        target_weights = weights[weights != 0].to_dict()

        return weights_to_decisions(target_weights, portfolio)
