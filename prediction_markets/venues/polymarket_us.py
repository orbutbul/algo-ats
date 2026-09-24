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
prices are `{value, currency}` objects, not bare numbers; /v1/markets
paginates by `offset` (no cursor/eof keys come back, confirmed live
2026-09-24) and defaults to oldest-first, so `closed=false` is sent
server-side or the first pages are all settled markets; `outcomes` order is
shuffled per market and `outcomePrices` are per-side buy quotes (not
always long-first), so the long side comes from `marketSides[long=true]`
and prices from `bestBidQuote`/`bestAskQuote` (which are the long side's).

Sports game markets (full-game winner/spread/total, identified by
`sportsMarketType`) also get the normalized proposition fields on Market.
Slugs are `<prefix>-<league>-<team a>-<team b>-<YYYY-MM-DD>[-...]`, shared by
every market on the game. A spread's long side is always the named team: a
negative `line` means "team wins by more than |line|" (same proposition as
Kalshi's YES), a positive one means "team doesn't lose by more than line",
i.e. the negation of "other team wins by more than line".
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import date, datetime, timezone

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


def _price_value(obj) -> float | None:
    """Unwraps a {'value': '0.69', 'currency': 'USD'} price object, or
    passes through a bare numeric/string price."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        obj = obj.get('value')
    return float(obj) if obj not in (None, '') else None


_SLUG_GAME_RE = re.compile(r'^[a-z]+-([a-z0-9]+)-([a-z0-9]+)-([a-z0-9]+)-(\d{4}-\d{2}-\d{2})')

# marketTypes values that hold full-game winner/spread/total markets -- pass
# as list_markets(market_types=...) for cross-venue matching.
SPORTS_MARKET_TYPES = ('moneyline', 'spreads', 'totals')


def _sports_market_type(sports_market_type: str | None) -> str | None:
    """'winner' / 'spread' / 'total' for full-game markets only -- not halves,
    quarters, or single-team totals ('football_team_points_full_game_total')."""
    smt = sports_market_type or ''
    if smt.endswith('_team_full_game_winner') or smt == 'ufc_fight_winner':
        return 'winner'
    if smt.endswith('_team_full_game_spread'):
        return 'spread'
    if smt.endswith('_team_full_game_total'):
        return 'total'
    return None


def _long_side(m: dict) -> dict | None:
    return next((s for s in m.get('marketSides') or [] if s.get('long')), None)


def _sports_fields(m: dict) -> dict:
    """Normalized proposition fields (see models.Market) for a full-game
    winner/spread/total market; {} for anything else."""
    market_type = _sports_market_type(m.get('sportsMarketType'))
    slug_match = _SLUG_GAME_RE.match(m.get('slug', ''))
    long_side = _long_side(m)
    if market_type is None or slug_match is None or long_side is None:
        return {}
    league, team_a, team_b, game_date = slug_match.groups()

    teams: dict[str, tuple[str, ...]] = {team_a: (), team_b: ()}
    for side in m.get('marketSides') or []:
        team = side.get('team') or {}
        if team.get('abbreviation') in teams:
            names = (team.get('name'), team.get('alias'), team.get('safeName'))
            teams[team['abbreviation']] = tuple(dict.fromkeys(n for n in names if n))

    long_team = (long_side.get('team') or {}).get('abbreviation')
    fields = {
        'league': league,
        'game_id': f'{league}-{team_a}-{team_b}-{game_date}',
        'event_date': date.fromisoformat(game_date),
        'market_type': market_type,
        'teams': teams,
    }
    if market_type == 'winner':
        if long_team not in teams:
            return {}
        fields['outcome'] = long_team
    elif market_type == 'spread':
        line = m.get('line')
        if long_team not in teams or not line:   # line 0 (pick'em) has no Kalshi equivalent
            return {}
        if line < 0:
            fields.update(outcome=long_team, line=-float(line))
        else:
            other_team = team_b if long_team == team_a else team_a
            fields.update(outcome=other_team, line=float(line), negated=True)
    else:  # total
        if m.get('line') is None:
            return {}
        fields.update(line=float(m['line']), negated=long_side.get('description') == 'Under')
    return fields


class PolymarketUSClient(VenueClient):
    name = 'polymarket_us'

    def __init__(self):
        self._limiter = RateLimiter(CALLS_PER_SECOND)

    def _get(self, path: str, params: dict | None = None):
        return get_json(f'{GATEWAY_BASE_URL}{path}', params=params, limiter=self._limiter)

    def _market_from_raw(self, m: dict) -> Market:
        # The list payload has no last-trade price: outcomePrices/marketSides
        # prices are per-side *buy* quotes (long = best ask, short = 1 - best
        # bid), and with one side of the book empty outcomePrices[0] is the
        # short side's. So last_price is the long side's bid/ask midpoint,
        # falling back to whichever quote exists.
        best_bid = _price_value(m.get('bestBidQuote'))
        best_ask = _price_value(m.get('bestAskQuote'))
        quotes = [q for q in (best_bid, best_ask) if q is not None]
        last_price = round(sum(quotes) / len(quotes), 4) if quotes else None
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
            best_bid=best_bid,
            best_ask=best_ask,
            **_sports_fields(m),
        )

    def list_markets(
        self,
        *,
        open_only: bool = True,
        market_types: str | Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """`market_types` filters server-side on the market's `marketType`
        ('moneyline', 'spreads', 'totals', 'futures', 'props', ...), e.g.
        SPORTS_MARKET_TYPES for cross-venue matching (~22k markets, vs
        paging through every futures/props market too)."""
        base_params: dict = {'limit': PAGE_LIMIT}
        if open_only:
            base_params['closed'] = 'false'
        if isinstance(market_types, str):
            base_params['marketTypes'] = market_types
        elif market_types is not None:
            base_params['marketTypes'] = list(market_types)

        markets: list[Market] = []
        offset = 0
        while True:
            data = self._get('/v1/markets', {**base_params, 'offset': offset})
            page = data.get('markets', [])
            for m in page:
                mk = self._market_from_raw(m)
                if not open_only or mk.is_open:
                    markets.append(mk)
            if len(page) < PAGE_LIMIT:
                break
            offset += PAGE_LIMIT
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
