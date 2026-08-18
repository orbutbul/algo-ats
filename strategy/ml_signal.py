"""
strategy/ml_signal.py -- Strategy implementation backed by the LightGBM
model trained in machine_learning/train.py.

Two ways to use the trained model, same as screener.py documents for its
own screens:
  1. `MLSignalStrategy` implements the `Strategy` ABC (strategy/base.py) for
     live/paper trading parity -- on_data() computes features for the
     latest bar only and returns Decisions.
  2. `predict_signal_matrix()` is the fast vectorized path for backtesting:
     runs the model over an entire (time x ticker) feature history at once
     and returns a (time x ticker) predicted-class DataFrame, which plugs
     directly into `vbt.Portfolio.from_signals()` the same way a
     screener.py screen does.

Predictions map to trade direction, not short-selling: "long" opens a
position, "short" closes an existing one (conservative default -- the
triple-barrier "short" class means "downside barrier likely hit first", not
an instruction to short the asset). "timeout" (no clear move expected) does
nothing.
"""

import pandas as pd

from machine_learning.features import build_feature_matrix, load_metadata
from machine_learning.indicators import build_indicator_set
from machine_learning.train import LABEL_MAP, load_model
from strategy.base import Strategy
from strategy.types import ClosePosition, Decision, OpenOrder, OrderSide, PortfolioState

INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}  # model output {0,1,2} -> {-1, 0, 1}


def _data_to_fields(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """{symbol: OHLCV DataFrame (Open/High/Low/Close/Volume)} -> {field: (time x ticker) DataFrame}, lower-cased to match screener.py's convention."""
    fields = {}
    for field in ('open', 'high', 'low', 'close', 'volume'):
        wide = pd.DataFrame({symbol: df[field.capitalize()] for symbol, df in data.items()})
        wide.columns.name = 'ticker'  # matches screener.load_fields()'s column-index name, which indicators.py's _drop_param_levels relies on
        fields[field] = wide
    return fields


def predict_signal_matrix(fields: dict[str, pd.DataFrame], model_bundle: dict, metadata: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    fields: screener.load_fields()-shaped OHLCV dict.
    model_bundle: output of machine_learning.train.load_model().

    Returns a (time x ticker) DataFrame of predicted labels in {-1, 0, 1}
    (short/timeout/long), NaN where features were unavailable (warmup or
    no metadata match). Feed `== 1` as `entries` / `== -1` as `exits` into
    vbt.Portfolio.from_signals().
    """
    X = build_feature_matrix(fields, metadata=metadata)
    X = X[model_bundle['feature_columns']]

    preds = model_bundle['model'].predict(X)
    preds = pd.Series(preds, index=X.index).map(INV_LABEL_MAP)

    return preds.unstack('ticker')


class MLSignalStrategy(Strategy):
    def __init__(self, model_path=None, position_weight: float = 0.1):
        self.model_bundle = load_model(model_path) if model_path else load_model()
        self.metadata = load_metadata()
        self.position_weight = position_weight

    def on_data(self, data: dict[str, pd.DataFrame], portfolio: PortfolioState) -> list[Decision]:
        fields = _data_to_fields(data)
        indicators = build_indicator_set(fields)

        latest = {}
        for name, df in indicators.items():
            row = df.iloc[-1]
            if row.isna().all():
                return []  # not enough history yet for any symbol
            latest[name] = row
        # Drop rows with missing indicators (rolling-window warmup) before
        # the metadata join -- metadata columns like market_cap_rank are
        # legitimately NaN for many tickers (see features.py), and LightGBM
        # handles NaN features natively, so they shouldn't cause a row drop.
        X_latest = pd.DataFrame(latest).dropna(how='any')
        if X_latest.empty:
            return []
        X_latest = X_latest.join(self.metadata, how='inner')

        X_latest = X_latest[self.model_bundle['feature_columns']]
        preds = self.model_bundle['model'].predict(X_latest)
        preds = pd.Series(preds, index=X_latest.index).map(INV_LABEL_MAP)

        decisions: list[Decision] = []
        for symbol, label in preds.items():
            has_position = symbol in portfolio.positions
            if label == 1 and not has_position:
                decisions.append(OpenOrder(symbol=symbol, side=OrderSide.BUY, weight=self.position_weight))
            elif label == -1 and has_position:
                decisions.append(ClosePosition(symbol=symbol))
        return decisions
