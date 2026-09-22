"""
prediction_markets/venues/polymarket_com.py — Polymarket.com
venue client (global, on-chain, Polygon — NOT Polymarket.us, see
polymarket_us.py).

All endpoints used here are public per data/prediction_markets/
polymarket_com.md (recon 2026-09-20): Gamma GET /markets, /markets/slug/{s};
CLOB GET /prices-history, /book. A key is only needed for the wallet-signed
trading endpoints and reading one's own /data/trades — not touched here.
Reads work from any IP even though this machine's IP is geoblocked for
order placement (confirmed in recon).

Each market has **two token ids, one per outcome** (`clobTokenIds`, JSON-
encoded string; index 0 = first outcome, conventionally "Yes"). Every
price/book method here defaults to outcome index 0 unless told otherwise —
pass `outcome_index=1` for the other side.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from prediction_markets.http import RateLimiter, get_json
from prediction_markets.models import (
    Market,
    OrderBook,
    OrderBookLevel,
    PricePoint,
    markets_to_df,
    price_points_to_df,
)
from prediction_markets.venues.base import VenueClient

GAMMA_BASE_URL = 'https://gamma-api.polymarket.com'
CLOB_BASE_URL = 'https://clob.polymarket.com'
CALLS_PER_SECOND = 8  # conservative; Gamma listings allow 900/10s, CLOB /book 1500/10s
PAGE_LIMIT = 100

# resolution hint -> (interval, fidelity_minutes). interval sets the window
# CLOB looks back over; fidelity is point spacing in minutes. See
# polymarket_com.md section 3 for the confirmed pairs/caveats (explicit
# startTs/endTs windows longer than ~15-20 days at 60 min fidelity are
# rejected -- pass start/end cautiously, or prefer coarser resolution).
_RESOLUTION_PARAMS = {'1m': ('1d', 1), '1h': ('max', 60), '1d': ('max', 1440)}


def _parse_json_field(raw: str | None, default):
    if not raw:
        return default
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return default


class PolymarketComClient(VenueClient):
    name = 'polymarket_com'

    def __init__(self):
        self._limiter = RateLimiter(CALLS_PER_SECOND)

    def _get_gamma(self, path: str, params: dict | None = None):
        return get_json(f'{GAMMA_BASE_URL}{path}', params=params, limiter=self._limiter)

    def _get_clob(self, path: str, params: dict | None = None):
        return get_json(f'{CLOB_BASE_URL}{path}', params=params, limiter=self._limiter)

    def _market_from_raw(self, m: dict) -> Market:
        outcome_prices = _parse_json_field(m.get('outcomePrices'), [])
        last_price = float(outcome_prices[0]) if outcome_prices else None
        return Market(
            venue=self.name,
            market_id=m['slug'],
            title=m.get('question', ''),
            is_open=bool(m.get('active')) and not m.get('closed', False),
            is_resolved=bool(m.get('closed', False)),
            last_price=last_price,
            close_time=pd.to_datetime(m['endDate'], utc=True) if m.get('endDate') else None,
            volume=float(m['volumeNum']) if m.get('volumeNum') is not None else None,
            raw=m,
        )

    def _token_id(self, market_id: str, outcome_index: int = 0) -> str:
        m = self._get_gamma(f'/markets/slug/{market_id}')
        token_ids = _parse_json_field(m.get('clobTokenIds'), [])
        if outcome_index >= len(token_ids):
            raise ValueError(f'{market_id} has no outcome index {outcome_index} (has {len(token_ids)} outcomes)')
        return token_ids[outcome_index]

    def list_markets(self, *, open_only: bool = True) -> pd.DataFrame:
        markets: list[Market] = []
        offset = 0
        params = {'limit': PAGE_LIMIT}
        if open_only:
            params.update({'active': 'true', 'closed': 'false'})
        while True:
            page = self._get_gamma('/markets', {**params, 'offset': offset})
            if not page:
                break
            markets.extend(self._market_from_raw(m) for m in page)
            offset += len(page)
            if len(page) < PAGE_LIMIT:
                break
        return markets_to_df(markets)

    def get_market(self, market_id: str) -> pd.DataFrame:
        m = self._get_gamma(f'/markets/slug/{market_id}')
        return markets_to_df([self._market_from_raw(m)])

    def get_price_history(
        self,
        market_id: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        resolution: str = '1h',
        outcome_index: int = 0,
    ) -> pd.DataFrame:
        if resolution not in _RESOLUTION_PARAMS:
            raise ValueError(f'resolution must be one of {list(_RESOLUTION_PARAMS)}, got {resolution!r}')
        token_id = self._token_id(market_id, outcome_index)
        interval, fidelity = _RESOLUTION_PARAMS[resolution]
        params: dict = {'market': token_id, 'fidelity': fidelity}
        if start is not None and end is not None:
            params['startTs'] = int(start.timestamp())
            params['endTs'] = int(end.timestamp())
        else:
            params['interval'] = interval

        data = self._get_clob('/prices-history', params)
        points = [
            PricePoint(venue=self.name, market_id=market_id,
                       ts=pd.to_datetime(pt['t'], unit='s', utc=True), price=float(pt['p']))
            for pt in data.get('history', [])
        ]
        return price_points_to_df(points)

    def get_orderbook(self, market_id: str, *, depth: int = 50, outcome_index: int = 0) -> OrderBook:
        token_id = self._token_id(market_id, outcome_index)
        data = self._get_clob('/book', {'token_id': token_id})
        # Polymarket returns bids ascending / asks descending, best price LAST on each side.
        bids = [OrderBookLevel(price=float(lvl['price']), size=float(lvl['size']), side='bid')
                for lvl in reversed(data.get('bids', []))][:depth]
        asks = [OrderBookLevel(price=float(lvl['price']), size=float(lvl['size']), side='ask')
                for lvl in reversed(data.get('asks', []))][:depth]
        ts = pd.to_datetime(int(data['timestamp']), unit='ms', utc=True) if data.get('timestamp') else datetime.now(timezone.utc)
        return OrderBook(venue=self.name, market_id=market_id, ts=ts, bids=bids, asks=asks, raw=data)
