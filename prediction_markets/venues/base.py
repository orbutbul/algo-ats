"""
prediction_markets/venues/base.py — the common interface every
venue client implements. See prediction_markets/models.py for the
normalized return shapes and the module docstring in
prediction_markets/__init__.py for the overall design.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd

from prediction_markets.models import OrderBook


class VenueClient(ABC):
    """Read-only client for one prediction-market venue's public market data.
    No credentials, no persistence, no state — every call hits the venue live
    and returns a fresh pandas DataFrame (or OrderBook, itself convertible
    via .to_dataframe()). Safe to call repeatedly; each client applies its
    own self-imposed rate limit internally."""

    name: str

    @abstractmethod
    def list_markets(self, *, open_only: bool = True) -> pd.DataFrame:
        """Markets currently offered by this venue. One row per market;
        columns match prediction_markets.models.markets_to_df.
        `open_only=True` filters to markets still accepting activity
        (venue-specific definition — Kalshi 'active', Polymarket
        active & not closed, etc)."""

    @abstractmethod
    def get_market(self, market_id: str) -> pd.DataFrame:
        """Single market's current detail, as a one-row DataFrame (same
        columns as list_markets) for a consistent return type across
        methods. Raises if market_id doesn't exist on this venue."""

    @abstractmethod
    def get_price_history(
        self,
        market_id: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        resolution: str = '1h',
    ) -> pd.DataFrame:
        """Time series of a specific market's traded/mid price. One row per
        point, columns: venue, market_id, ts, price. `resolution` is a
        venue-agnostic hint ('1m', '1h', '1d'); each client maps it to its
        own interval/fidelity parameter and documents any venue-specific
        lookback cap in its own docstring (they differ a lot — see
        data/prediction_markets/*.md)."""

    @abstractmethod
    def get_orderbook(self, market_id: str, *, depth: int = 50) -> OrderBook:
        """Current bid/ask ladder for a market, best-first on each side."""
