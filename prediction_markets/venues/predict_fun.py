"""
prediction_markets/venues/predict_fun.py — predict.fun venue
client (onchain, BNB Chain).

predict.fun wasn't part of the 2026-09-20 recon; it was scouted in this
session (2026-09-22) by hitting `api-testnet.predict.fun` live (no key
needed on testnet). list_markets/get_market/get_orderbook below are
confirmed against real testnet responses. get_price_history is **still
unverified**: `/v1/markets/{id}/timeseries` requires a `metric` query
param whose enum value wasn't discoverable (every guess returned
`400 failed to parse parameter "Metric"`, and no reachable OpenAPI/Swagger
JSON was found at the usual paths under either host) — fix `_METRIC` once
the right value is known (check https://api.predict.fun/docs in a browser,
or a captured request from predict.fun's own frontend).

Confirmed testnet shapes (2026-09-22):
- `GET /v1/markets?limit=&after=` -> `{"success", "cursor", "data": [market, ...]}`.
  The pagination param is **`after`, not `cursor`** — passing `cursor` back
  is silently ignored (server just re-returns page 1 forever; caught by
  actually running this against testnet, see list_markets' docstring).
  `limit` also appears capped at 20/page server-side regardless of the
  value sent. There is **no server-side open/status filter** on this
  endpoint (unlike Kalshi/Polymarket) — testnet alone has 2,000+ markets,
  so `list_markets(open_only=True)` must page through everything to filter
  client-side; see `max_pages`. Market fields actually used here: `id`
  (int), `title`, `question`, `status` (e.g. "REGISTERED"), `tradingStatus`
  (e.g. "OPEN"), `outcomes` (list of `{name, bestBid: {price,size},
  bestAsk: {price,size}, ...}`), `stats` (null on quiet markets; assumed to
  hold volume when populated — unconfirmed since no sample had it
  non-null). No close/end-date field was present on any sampled market;
  `close_time` is always None here until one is found.
- `GET /v1/markets/{id}` -> `{"success", "data": market}` (same market shape).
- `GET /v1/markets/{id}/orderbook` -> `{"success", "data": {"bids": [[price,size],...],
  "asks": [[price,size],...], "marketId", "updateTimestampMs", "lastOrderSettled"}}`.
  Both sides already arrive best-first (bids descending, asks ascending) —
  unlike Kalshi/Polymarket, no reversal needed.

Base URLs: mainnet `https://api.predict.fun` (needs `PREDICT_FUN_API_KEY`
for ALL reads, unlike the other three venues — not verified live since we
have no mainnet key), testnet `https://api-testnet.predict.fun` (no key,
240 req/min) — defaults to testnet so this client works out of the box.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv

from prediction_markets.http import RateLimiter, VenueHTTPError, get_json
from prediction_markets.models import (
    Market,
    OrderBook,
    OrderBookLevel,
    PricePoint,
    markets_to_df,
    price_points_to_df,
)
from prediction_markets.venues.base import VenueClient

load_dotenv()

MAINNET_BASE_URL = 'https://api.predict.fun'
TESTNET_BASE_URL = 'https://api-testnet.predict.fun'
CALLS_PER_SECOND = 4  # documented 240 req/min = 4/s
PAGE_LIMIT = 100       # requested; server appears to cap actual page size at 20 (confirmed live)
DEFAULT_MAX_PAGES = 50 # ~1,000 markets at the real 20/page cap -- there's no server-side open
                        # filter here (see list_markets), and testnet alone has 2,000+ markets, so
                        # this bounds an on-demand call by default; pass max_pages=None for no cap

# resolution hint -> guessed `metric`/interval query values for /timeseries.
# UNCONFIRMED -- see module docstring. Placeholder until the real `metric`
# enum is found; calling get_price_history will raise until then.
_RESOLUTION_INTERVAL = {'1m': '1m', '1h': '1h', '1d': '1d'}


def _first_outcome_mid(outcomes: list[dict]) -> float | None:
    """predict.fun markets have no single last-traded-price field on the
    market object itself (confirmed absent from every sampled market) --
    approximate it as the mid of the first outcome's best bid/ask."""
    if not outcomes:
        return None
    bid = (outcomes[0].get('bestBid') or {}).get('price')
    ask = (outcomes[0].get('bestAsk') or {}).get('price')
    if bid is None and ask is None:
        return None
    if bid is None or ask is None:
        return float(bid if bid is not None else ask)
    return round((float(bid) + float(ask)) / 2, 6)


class PredictFunClient(VenueClient):
    name = 'predict_fun'

    def __init__(self, env: str = 'testnet'):
        if env not in ('testnet', 'mainnet'):
            raise ValueError(f"env must be 'testnet' or 'mainnet', got {env!r}")
        self.env = env
        self.base_url = TESTNET_BASE_URL if env == 'testnet' else MAINNET_BASE_URL
        self._limiter = RateLimiter(CALLS_PER_SECOND)
        self._api_key = os.getenv('PREDICT_FUN_API_KEY')
        if env == 'mainnet' and not self._api_key:
            raise ValueError(
                'predict.fun mainnet reads require PREDICT_FUN_API_KEY (set it in .env); '
                'use env="testnet" (the default) if you just want to read public test data.'
            )

    def _get(self, path: str, params: dict | None = None) -> dict:
        headers = {'X-API-Key': self._api_key} if self._api_key else None  # header name unconfirmed (mainnet, untested)
        return get_json(f'{self.base_url}{path}', params=params, headers=headers, limiter=self._limiter)

    def _market_from_raw(self, m: dict) -> Market:
        is_open = m.get('tradingStatus') == 'OPEN'
        is_resolved = m.get('status') in ('RESOLVED', 'SETTLED') or m.get('resolution') is not None
        stats = m.get('stats') or {}
        return Market(
            venue=self.name,
            market_id=str(m['id']),
            title=m.get('title') or m.get('question', ''),
            is_open=is_open,
            is_resolved=is_resolved,
            last_price=_first_outcome_mid(m.get('outcomes', [])),
            close_time=None,  # no close/end-date field found on any sampled market (see module docstring)
            volume=float(stats['volume']) if isinstance(stats, dict) and stats.get('volume') not in (None, '') else None,
            raw=m,
        )

    def list_markets(self, *, open_only: bool = True, max_pages: int | None = DEFAULT_MAX_PAGES) -> pd.DataFrame:
        """`max_pages` bounds how many ~20-row pages this walks (there's no
        server-side open/status filter on predict.fun's /v1/markets, so
        filtering to open markets means paging through everything -- see
        module docstring). Pass `max_pages=None` to walk the full catalog."""
        markets: list[Market] = []
        cursor = None
        pages = 0
        while max_pages is None or pages < max_pages:
            params: dict = {'limit': PAGE_LIMIT}
            if cursor:
                params['after'] = cursor
            data = self._get('/v1/markets', params)
            page = data.get('data', [])
            for m in page:
                mk = self._market_from_raw(m)
                if not open_only or mk.is_open:
                    markets.append(mk)
            pages += 1
            cursor = data.get('cursor')
            if not cursor or not page:
                break
        return markets_to_df(markets)

    def get_market(self, market_id: str) -> pd.DataFrame:
        try:
            data = self._get(f'/v1/markets/{market_id}')
            return markets_to_df([self._market_from_raw(data['data'])])
        except VenueHTTPError:
            all_markets = self.list_markets(open_only=False)
            match = all_markets[all_markets['market_id'] == str(market_id)]
            if match.empty:
                raise ValueError(f'market {market_id!r} not found on predict.fun ({self.env})')
            return match.reset_index(drop=True)

    def get_price_history(
        self,
        market_id: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        resolution: str = '1h',
    ) -> pd.DataFrame:
        raise NotImplementedError(
            "predict.fun's /v1/markets/{id}/timeseries requires a `metric` query parameter whose "
            "valid enum value could not be determined (see this module's docstring) -- fix "
            '_RESOLUTION_INTERVAL and this method once the real parameter is known.'
        )

    def get_orderbook(self, market_id: str, *, depth: int = 50) -> OrderBook:
        data = self._get(f'/v1/markets/{market_id}/orderbook')['data']
        # predict.fun already returns both sides best-first (bids desc, asks asc) -- no reversal needed.
        bids = [OrderBookLevel(price=float(p), size=float(s), side='bid') for p, s in data.get('bids', [])][:depth]
        asks = [OrderBookLevel(price=float(p), size=float(s), side='ask') for p, s in data.get('asks', [])][:depth]
        ts = (pd.to_datetime(data['updateTimestampMs'], unit='ms', utc=True)
              if data.get('updateTimestampMs') else datetime.now(timezone.utc))
        return OrderBook(venue=self.name, market_id=market_id, ts=ts, bids=bids, asks=asks, raw=data)
