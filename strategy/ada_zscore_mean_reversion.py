"""
strategy/ada_zscore_mean_reversion.py — long-only Bollinger/z-score mean
reversion, researched against ADAUSDT (see ada_dashboard.html for the
backtest: stationarity diagnostics + Sharpe/drawdown/trade stats across
1H/5min/1min bars).

Rationale: ADA's raw price is non-stationary (it trended hard down over the
backtest windows), but its short-horizon *returns* mean-revert -- variance
ratios collapse well below 1 and return autocorrelation is negative at every
horizon tested (2 bars to 120 bars, across three timeframes). The strategy
buys stretched-below-average dips and exits back at the mean, rather than
predicting direction. Shorting the stretched-above-average side was tested
and consistently underperformed (fees + whipsaw ate the edge faster than the
short-side signal could pay for it), so this is long-only by design, not by
omission.

Entry:  z-score of Close vs its own rolling mean/std drops below -entry_z.
Exit:   z-score reverts back up to exit_z (default 0, i.e. the rolling mean),
        OR price has fallen stop_loss_pct below the recorded entry price
        (tail-risk backstop; rarely binds in the backtest but protects
        against a dip that breaks trend instead of reverting).

Defaults are the 5-minute-bar configuration, the most robust of the three
timeframes backtested (best trade count for the Sharpe achieved -- the
1-minute config scores a higher Sharpe but on only ~40 trades over 6 months,
too thin a sample to lean on). Pass a different window/entry_z/exit_z/
resample_freq to run the 1H or 1-minute variant instead:

    1H bars,   13.5mo backtest: window=20,   entry_z=2.5,  exit_z=0.0, stop_loss_pct=0.05
    5min bars, 6mo backtest:    window=48,   entry_z=3.75, exit_z=0.0, stop_loss_pct=0.05  (default)
    1min bars, 6mo backtest:    window=1440, entry_z=3.5,  exit_z=0.0, stop_loss_pct=0.08

Caveats carried over from the backtest, not resolved by this file:
  - Parameters were grid-searched over the same data the backtest reports,
    i.e. in-sample selection -- treat as a starting point, not a validated
    edge, until it's been walk-forward tested.
  - This has never been run against live fills/spreads, only bar closes with
    flat fee/slippage assumptions (0.075% / 0.05% per side).
  - The bar feed this consumes (via LiveDataClient / strategy_runner.py) is
    presently wired for equities (live_feed.py); routing ADA/crypto bars and
    a crypto-capable executor into that same pipeline is a separate
    integration step, not done here. This class only implements the
    Strategy contract -- symbol lookup, order routing, and account funding
    are the caller's responsibility (see strategy_runner.py).
"""

import pandas as pd

from strategy.base import Strategy
from strategy.types import ClosePosition, Decision, OpenOrder, OrderSide, PortfolioState


class AdaZScoreMeanReversionStrategy(Strategy):
    def __init__(
        self,
        symbol: str = 'ADA/USD',
        window: int = 48,
        entry_z: float = 3.75,
        exit_z: float = 0.0,
        stop_loss_pct: float = 0.05,
        resample_freq: str | None = '5min',
        weight: float = 0.1,
    ):
        """
        symbol: key this strategy looks for in the `data` dict passed to
            on_data() -- must match whatever the caller's LiveDataClient
            publishes this asset as.
        window: rolling lookback (in bars, after resampling) for the mean/std.
        entry_z / exit_z: z-score thresholds -- see module docstring.
        stop_loss_pct: fractional drawdown from entry price that force-closes
            the position regardless of z-score (0.05 = 5%).
        resample_freq: pandas offset alias to resample the incoming bars to
            before computing the signal (e.g. '5min', '1h'). Pass None to
            trade directly on the bars as received (use this if the feed
            already delivers bars at the target timeframe -- e.g. a 1-minute
            feed for the 1-minute config, where resampling would be a no-op).
        weight: fraction of allocatable equity to commit on entry.
        """
        self.symbol = symbol
        self.window = window
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_loss_pct = stop_loss_pct
        self.resample_freq = resample_freq
        self.weight = weight

    def _resampled_close(self, df: pd.DataFrame) -> pd.Series:
        if self.resample_freq is None:
            return df['Close']
        bars = df.resample(self.resample_freq).agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum',
        }).dropna()
        return bars['Close']

    def on_data(self, data: dict[str, pd.DataFrame], portfolio: PortfolioState) -> list[Decision]:
        df = data.get(self.symbol)
        if df is None or df.empty:
            return []

        close = self._resampled_close(df)
        if len(close) < self.window + 1:
            return []

        mean = close.rolling(self.window).mean().iloc[-1]
        std = close.rolling(self.window).std().iloc[-1]
        last_price = close.iloc[-1]
        if pd.isna(mean) or pd.isna(std) or std == 0:
            return []
        z = (last_price - mean) / std

        position = portfolio.positions.get(self.symbol)

        if position is not None:
            stopped_out = last_price <= position.avg_entry_price * (1 - self.stop_loss_pct)
            reverted = z >= self.exit_z
            if stopped_out or reverted:
                return [ClosePosition(symbol=self.symbol)]
            return []

        if z < -self.entry_z:
            return [OpenOrder(symbol=self.symbol, side=OrderSide.BUY, weight=self.weight)]
        return []
