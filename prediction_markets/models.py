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

Sports game markets additionally carry a venue-agnostic description of the
proposition their YES/long side pays on (league, game, market_type, outcome,
line, negated) so prediction_markets/matching.py can pair equivalent bets
across venues without knowing either venue's ticker/slug grammar. Those
fields stay None for anything a client doesn't recognize as a full-game
winner/spread/total market.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
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
    best_bid: float | None = None         # top-of-book for the same outcome as last_price, from the list payload
    best_ask: float | None = None         # (None where the venue's list payload doesn't carry quotes)

    # --- sports game markets only (None otherwise) ---------------------------
    # The YES/long side pays iff the proposition below is true (or false, if
    # `negated`). Propositions: 'winner' -> `outcome` wins; 'spread' ->
    # `outcome` wins by more than `line`; 'total' -> combined score > `line`.
    league: str | None = None             # 'nfl', 'cfb', 'mlb', 'nba', 'wnba', 'nhl', 'ufc'
    game_id: str | None = None            # venue-native id shared by every market on the same game
    event_date: date | None = None        # scheduled game date (US Eastern calendar date on both venues)
    market_type: str | None = None        # 'winner' | 'spread' | 'total'
    outcome: str | None = None            # venue-native team/fighter code the proposition is about
    line: float | None = None             # spread margin (always > 0) or total threshold
    negated: bool = False                 # YES/long pays when the proposition is FALSE
    teams: dict[str, tuple[str, ...]] = field(default_factory=dict)   # team code -> display names this market mentions

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
            'best_bid': self.best_bid,
            'best_ask': self.best_ask,
            'league': self.league,
            'game_id': self.game_id,
            'event_date': self.event_date,
            'market_type': self.market_type,
            'outcome': self.outcome,
            'line': self.line,
            'negated': self.negated,
            'teams': self.teams,
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
    cols = ['venue', 'market_id', 'title', 'is_open', 'is_resolved', 'last_price', 'close_time', 'volume',
            'best_bid', 'best_ask', 'league', 'game_id', 'event_date', 'market_type', 'outcome', 'line', 'negated', 'teams']
    return pd.DataFrame([m.to_row() for m in markets], columns=cols)


def price_points_to_df(points: list[PricePoint]) -> pd.DataFrame:
    cols = ['venue', 'market_id', 'ts', 'price']
    df = pd.DataFrame([p.to_row() for p in points], columns=cols)
    if not df.empty:
        df['ts'] = pd.to_datetime(df['ts'], utc=True)
        df = df.sort_values('ts').reset_index(drop=True)
    return df
