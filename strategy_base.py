"""
strategy_base.py — the Strategy interface.

Any implementation takes the same symbol->OHLCV DataFrame shape produced by
LiveDataClient.snapshot() (live_client.py) / utils.get_data(), plus current
portfolio state, and returns a list of Decisions (strategy_types.py). This
keeps every strategy's output the same shape regardless of its internal
logic, so callers don't need to special-case which strategy produced it.
"""

from abc import ABC, abstractmethod

import pandas as pd

from strategy_types import Decision, PortfolioState


class Strategy(ABC):
    @abstractmethod
    def on_data(self, data: dict[str, pd.DataFrame], portfolio: PortfolioState) -> list[Decision]:
        """
        data: {symbol: OHLCV DataFrame}, shaped like LiveDataClient.snapshot()
              (columns Open/High/Low/Close/Volume, datetime index).
        portfolio: current positions/open orders, supplied by the caller —
              the strategy never queries a broker itself.
        Returns zero or more Decisions for this time step.
        """
        ...
