"""
strategy_cross_sectional_momentum.py — example cross-asset Strategy.

Not a real trading strategy — just enough logic (an N-bar return signal
across the universe, ranked into weights via utils.compute_allocations,
diffed into Decisions via strategy_utils.weights_to_decisions) to prove a
strategy can reason about the whole universe at once and still return the
same list[Decision] shape as a single-asset strategy like
strategy_ma_cross.py.
"""

import pandas as pd

from strategy_base import Strategy
from strategy_types import Decision, PortfolioState
from strategy_utils import weights_to_decisions
from utils import compute_allocations


class CrossSectionalMomentumStrategy(Strategy):
    def __init__(self, lookback: int = 30, top_k: int = 3):
        self.lookback, self.top_k = lookback, top_k

    def on_data(self, data: dict[str, pd.DataFrame], portfolio: PortfolioState) -> list[Decision]:
        returns = {
            symbol: df['Close'].iloc[-1] / df['Close'].iloc[-self.lookback] - 1
            for symbol, df in data.items()
            if len(df) >= self.lookback
        }
        if not returns:
            return []

        signals = pd.DataFrame([returns])
        weights = compute_allocations(signals, method='rank', top_k=self.top_k).iloc[-1]
        target_weights = weights[weights != 0].to_dict()

        return weights_to_decisions(target_weights, portfolio)
