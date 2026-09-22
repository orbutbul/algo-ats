"""
prediction_markets/models.py — normalized data shapes shared by
every venue client, so callers never have to branch on which venue a
DataFrame came from. Each venue client is responsible for translating its
own raw response shape (Kalshi's `*_dollars` strings, Polymarket.com's
per-outcome token ids, Polymarket.us's `{value, currency}` objects, ...)
into these before returning anything.

Prices are always a float in [0, 1] (probability / dollars-per-$1-contract
either way, since every venue here is a binary or per-outcome market paying
$1 on the winning side). Timestamps are always timezone-aware UTC
`pandas.Timestamp`/`datetime`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd


@dataclass
class Market:
    """One tradeable market/outcome on a venue."""

    venue: str
    market_id: str            # venue-native id: Kalshi ticker, Polymarket slug, predict.fun market id
    title: str
    is_open: bool
    is_resolved: bool
    last_price: float | None = None       # price of the outcome represented (default: "Yes"), in [0, 1]
    close_time: datetime | None = None
    volume: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)   # untouched venue payload, for anything not normalized

    def to_row(self) -> dict[str, Any]:
        return {
            'venue': self.venue,
            'market_id': self.market_id,
            'title': self.title,
            'is_open': self.is_open,
            'is_resolved': self.is_resolved,
            'last_price': self.last_price,
            'close_time': self.close_time,
            'volume': self.volume,
        }


@dataclass
class PricePoint:
    """One point of a market's price/value history."""

    venue: str
    market_id: str
    ts: datetime
    price: float

    def to_row(self) -> dict[str, Any]:
        return {'venue': self.venue, 'market_id': self.market_id, 'ts': self.ts, 'price': self.price}


@dataclass
class OrderBookLevel:
    price: float
    size: float
    side: str   # 'bid' | 'ask'


@dataclass
class OrderBook:
    """Current book snapshot for one market. Levels are NOT assumed sorted;
    each venue client normalizes them into best-first ascending-away-from-mid
    order before returning."""

    venue: str
    market_id: str
    ts: datetime
    bids: list[OrderBookLevel]
    asks: list[OrderBookLevel]
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        rows = [
            {'venue': self.venue, 'market_id': self.market_id, 'ts': self.ts,
             'side': lvl.side, 'price': lvl.price, 'size': lvl.size}
            for lvl in (*self.bids, *self.asks)
        ]
        return pd.DataFrame(rows, columns=['venue', 'market_id', 'ts', 'side', 'price', 'size'])

    @property
    def best_bid(self) -> float | None:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return self.asks[0].price if self.asks else None


def markets_to_df(markets: list[Market]) -> pd.DataFrame:
    cols = ['venue', 'market_id', 'title', 'is_open', 'is_resolved', 'last_price', 'close_time', 'volume']
    return pd.DataFrame([m.to_row() for m in markets], columns=cols)


def price_points_to_df(points: list[PricePoint]) -> pd.DataFrame:
    cols = ['venue', 'market_id', 'ts', 'price']
    df = pd.DataFrame([p.to_row() for p in points], columns=cols)
    if not df.empty:
        df['ts'] = pd.to_datetime(df['ts'], utc=True)
        df = df.sort_values('ts').reset_index(drop=True)
    return df
