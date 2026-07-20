"""
strategy/ma_cross.py — example Strategy implementation.

Not a real trading strategy — just enough logic (a vbt MA crossover,
position-aware branching) to exercise every part of the Strategy/Decision
contract in strategy/base.py / strategy/types.py.
"""

import pandas as pd
from vectorbtpro import vbt

from strategy.base import Strategy
from strategy.types import ClosePosition, Decision, OpenOrder, OrderSide, PortfolioState


class MovingAverageCrossStrategy(Strategy):
    def __init__(self, fast: int = 10, slow: int = 30):
        self.fast, self.slow = fast, slow

    def on_data(self, data: dict[str, pd.DataFrame], portfolio: PortfolioState) -> list[Decision]:
        decisions: list[Decision] = []
        for symbol, df in data.items():
            if len(df) < self.slow:
                continue
            close = df['Close']
            fast_ma = vbt.MA.run(close, self.fast).ma
            slow_ma = vbt.MA.run(close, self.slow).ma
            crossed_above = (fast_ma.iloc[-2] <= slow_ma.iloc[-2]) and (fast_ma.iloc[-1] > slow_ma.iloc[-1])
            crossed_below = (fast_ma.iloc[-2] >= slow_ma.iloc[-2]) and (fast_ma.iloc[-1] < slow_ma.iloc[-1])
            has_position = symbol in portfolio.positions
            if crossed_above and not has_position:
                decisions.append(OpenOrder(symbol=symbol, side=OrderSide.BUY, weight=0.1))
            elif crossed_below and has_position:
                decisions.append(ClosePosition(symbol=symbol))
        return decisions
