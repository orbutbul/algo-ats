"""
prediction_markets/venues/polymarket_us.py — Polymarket.us venue
client (CFTC-regulated US exchange, separate stack from Polymarket.com —
see polymarket_com.py; do not conflate the two, they share no infra/ids).

Endpoints used here are all on the public `gateway.polymarket.us` host and
need no key, per data/prediction_markets/polymarket_us.md (recon
2026-09-20): GET /v1/markets (list, and filtered by `?slug=` for one
market's detail -- **/v1/markets/{slug} alone 404s, confirmed live**;
there's no dedicated single-market detail path), /v1/price-history,
/v1/markets/{slug}/book. A key is only needed for the authenticated
`api.polymarket.us` host (own portfolio, orders, the private WebSockets,
and the key-gated TRADE stream) — none of which this client touches, since
.us has no public trade tape at all (price history here is book-derived,
not real trades).

Notable quirks baked in: the "ask" side of the book is named `offers`, not
`asks`; a resolved market can show `active:true` AND `closed:true`
simultaneously (use `closed`, not `active`, for is_resolved/is_open);
prices are `{value, currency}` objects, not bare numbers.
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

GATEWAY_BASE_URL = 'https://gateway.polymarket.us'
CALLS_PER_SECOND = 8  # documented limit is 20 req/s per IP; stay well under it
PAGE_LIMIT = 100

# resolution hint -> (fixedInterval, fidelity_minutes), restricted to the
# combos actually confirmed in polymarket_us.md's table -- .us doesn't
# accept arbitrary fidelity values the way Kalshi/Polymarket.com do.
_RESOLUTION_PARAMS = {'1m': ('INTERVAL_6H', 1), '1h': ('INTERVAL_1D', 5), '1d': ('INTERVAL_ALL', 180)}


def _coerce_list(value, default=None):
    """.us fields like outcomes/outcomePrices may come back as a JSON-encoded
    string (Gamma-style) or an already-parsed list -- handle either."""
    if value is None:
        return default if default is not None else []
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default if default is not None else []
    return value


def _price_value(obj) -> float | None:
    """Unwraps a {'value': '0.69', 'currency': 'USD'} price object, or
    passes through a bare numeric/string price."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        obj = obj.get('value')
    return float(obj) if obj not in (None, '') else None


class PolymarketUSClient(VenueClient):
    name = 'polymarket_us'

    def __init__(self):
        self._limiter = RateLimiter(CALLS_PER_SECOND)

    def _get(self, path: str, params: dict | None = None):
        return get_json(f'{GATEWAY_BASE_URL}{path}', params=params, limiter=self._limiter)

    def _market_from_raw(self, m: dict) -> Market:
        outcome_prices = _coerce_list(m.get('outcomePrices'))
        last_price = _price_value(outcome_prices[0]) if outcome_prices else None
        closed = bool(m.get('closed', False))
        return Market(
            venue=self.name,
            market_id=m['slug'],
            title=m.get('question') or m.get('title', ''),
            is_open=bool(m.get('active')) and not closed,
            is_resolved=closed,
            last_price=last_price,
            close_time=pd.to_datetime(m['endDate'], utc=True) if m.get('endDate') else None,
            volume=None,  # not exposed on .us market objects (confirmed absent in recon)
            raw=m,
        )

    def list_markets(self, *, open_only: bool = True) -> pd.DataFrame:
        markets: list[Market] = []
        cursor = None
        while True:
            params: dict = {'limit': PAGE_LIMIT}
            if cursor:
                params['cursor'] = cursor
            data = self._get('/v1/markets', params)
            page = data.get('markets', [])
            for m in page:
                mk = self._market_from_raw(m)
                if not open_only or mk.is_open:
                    markets.append(mk)
            if data.get('eof', True) or not data.get('nextCursor'):
                break
            cursor = data['nextCursor']
        return markets_to_df(markets)

    def get_market(self, market_id: str) -> pd.DataFrame:
        # There's no /v1/markets/{slug} detail path (confirmed 404 live) --
        # the gateway only exposes per-slug suffixed endpoints (bbo/book/
        # settlement) plus this filtered list call.
        data = self._get('/v1/markets', {'slug': market_id})
        markets = data.get('markets', [])
        if not markets:
            raise ValueError(f'market {market_id!r} not found on polymarket.us')
        return markets_to_df([self._market_from_raw(markets[0])])

    def get_price_history(
        self,
        market_id: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        resolution: str = '1h',
    ) -> pd.DataFrame:
        if resolution not in _RESOLUTION_PARAMS:
            raise ValueError(f'resolution must be one of {list(_RESOLUTION_PARAMS)}, got {resolution!r}')
        params: dict = {'symbol': market_id}
        if start is not None and end is not None:
            params['timestamp.startTimestamp'] = int(start.timestamp())
            params['timestamp.endTimestamp'] = int(end.timestamp())
            params['fidelity'] = 1
        else:
            fixed_interval, fidelity = _RESOLUTION_PARAMS[resolution]
            params['fixedInterval'] = fixed_interval
            params['fidelity'] = fidelity

        data = self._get('/v1/price-history', params)
        points = [
            PricePoint(venue=self.name, market_id=market_id,
                       ts=pd.to_datetime(pt['timestamp'], unit='s', utc=True),
                       price=float(pt['longPrice']))
            for pt in data.get('history', [])
        ]
        return price_points_to_df(points)

    def get_orderbook(self, market_id: str, *, depth: int = 50) -> OrderBook:
        data = self._get(f'/v1/markets/{market_id}/book')['marketData']
        bids = [OrderBookLevel(price=_price_value(lvl['px']), size=float(lvl['qty']), side='bid')
                for lvl in data.get('bids', [])][:depth]
        asks = [OrderBookLevel(price=_price_value(lvl['px']), size=float(lvl['qty']), side='ask')
                for lvl in data.get('offers', [])][:depth]  # .us calls the ask side "offers"
        return OrderBook(venue=self.name, market_id=market_id, ts=datetime.now(timezone.utc),
                          bids=bids, asks=asks, raw=data)
