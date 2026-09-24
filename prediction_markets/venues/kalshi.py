"""
prediction_markets/venues/kalshi.py — Kalshi venue client.

All endpoints used here are public (no headers) per data/prediction_markets/
kalshi.md (recon 2026-09-20): GET /markets, /markets/{ticker},
/markets/{ticker}/orderbook, /series/{s}/markets/{t}/candlesticks. A key is
only needed for the WebSocket, own portfolio, or trading — none of which
this client touches.

Prices arrive as `*_dollars` strings (4dp, e.g. "0.7400") — converted to
float here. Candlesticks need a series_ticker, which isn't on the market
object itself; it's derived from event_ticker's first '-'-separated segment
(e.g. event_ticker "KXNFLGAME-26SEP20LVLAC" -> series "KXNFLGAME"), per the
ticker convention documented in kalshi.md. If Kalshi ever breaks that
convention this will raise via a 404 from the candlesticks call, not
silently return wrong data.

Sports game markets (SPORTS_SERIES below) also get the normalized
proposition fields on Market. The game is the event_ticker's second segment
("26SEP28PHICHI", or "26SEP271520BALNYY" with a start time for MLB), shared
across a game's GAME/SPREAD/TOTAL series; the team a market is about is the
ticker's last segment, with a spread's strike digits stripped ("PHI8" ->
"PHI"). The full market list is 130k+ markets (~7 min to page through), so
pass `series_ticker=SPORTS_SERIES` to list_markets when only sports matter.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone

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

BASE_URL = 'https://external-api.kalshi.com/trade-api/v2'
DEMO_BASE_URL = 'https://external-api.demo.kalshi.co/trade-api/v2'
CALLS_PER_SECOND = 5  # conservative self-imposed floor; Basic tier is ~20 req/s unauthenticated
PAGE_LIMIT = 200

# resolution hint -> Kalshi candlestick period_interval (minutes), and the
# default lookback window used when start/end aren't given.
_RESOLUTION_MINUTES = {'1m': 1, '1h': 60, '1d': 1440}
_DEFAULT_LOOKBACK = {'1m': timedelta(hours=6), '1h': timedelta(days=14), '1d': timedelta(days=180)}


# series_ticker -> (league, market_type). Full-game series only; YES means
# "<team> wins", "<team> wins by > floor_strike", or "combined score >
# floor_strike" respectively.
SPORTS_SERIES = {
    'KXNFLGAME': ('nfl', 'winner'), 'KXNFLSPREAD': ('nfl', 'spread'), 'KXNFLTOTAL': ('nfl', 'total'),
    'KXNCAAFGAME': ('cfb', 'winner'), 'KXNCAAFSPREAD': ('cfb', 'spread'), 'KXNCAAFTOTAL': ('cfb', 'total'),
    'KXMLBGAME': ('mlb', 'winner'), 'KXMLBSPREAD': ('mlb', 'spread'), 'KXMLBTOTAL': ('mlb', 'total'),
    'KXNBAGAME': ('nba', 'winner'), 'KXNBASPREAD': ('nba', 'spread'), 'KXNBATOTAL': ('nba', 'total'),
    'KXWNBAGAME': ('wnba', 'winner'), 'KXWNBASPREAD': ('wnba', 'spread'), 'KXWNBATOTAL': ('wnba', 'total'),
    'KXNHLGAME': ('nhl', 'winner'), 'KXNHLSPREAD': ('nhl', 'spread'), 'KXNHLTOTAL': ('nhl', 'total'),
    'KXUFCFIGHT': ('ufc', 'winner'),
}
_GAME_DATE_RE = re.compile(r'^(\d{2}[A-Z]{3}\d{2})')


def _sports_fields(m: dict) -> dict:
    """Normalized proposition fields (see models.Market) for a SPORTS_SERIES
    market; {} for anything else, leaving those fields at their defaults."""
    series, _, game = m.get('event_ticker', '').partition('-')
    spec = SPORTS_SERIES.get(series)
    date_match = _GAME_DATE_RE.match(game)
    if spec is None or date_match is None:
        return {}
    league, market_type = spec
    fields = {
        'league': league,
        'game_id': f'{league}:{game}',   # series-free, so a game's GAME/SPREAD/TOTAL markets share it
        'event_date': datetime.strptime(date_match.group(1), '%y%b%d').date(),
    }
    suffix = m['ticker'].rsplit('-', 1)[-1]
    if market_type == 'winner':
        fields.update(market_type='winner', outcome=suffix, teams={suffix: (m.get('yes_sub_title') or '',)})
    elif m.get('strike_type') == 'greater' and m.get('floor_strike') is not None:
        fields.update(market_type=market_type, line=float(m['floor_strike']))
        if market_type == 'spread':
            code = suffix.rstrip('0123456789')
            name = (m.get('yes_sub_title') or '').split(' wins by')[0]   # "PHI Eagles wins by over 7.5 points"
            fields.update(outcome=code, teams={code: (name,)})
    return fields


def _f(dollar_str: str | None) -> float | None:
    return float(dollar_str) if dollar_str not in (None, '') else None


class KalshiClient(VenueClient):
    name = 'kalshi'

    def __init__(self, demo: bool = False):
        self.base_url = DEMO_BASE_URL if demo else BASE_URL
        self._limiter = RateLimiter(CALLS_PER_SECOND)

    def _get(self, path: str, params: dict | None = None) -> dict:
        return get_json(f'{self.base_url}{path}', params=params, limiter=self._limiter)

    def _market_from_raw(self, m: dict) -> Market:
        return Market(
            venue=self.name,
            market_id=m['ticker'],
            title=m.get('title', ''),
            is_open=m.get('status') == 'active',
            is_resolved=m.get('status') == 'finalized',
            last_price=_f(m.get('last_price_dollars')),
            close_time=pd.to_datetime(m['close_time'], utc=True) if m.get('close_time') else None,
            volume=_f(m.get('volume_fp')),
            raw=m,
            best_bid=_f(m.get('yes_bid_dollars')),   # 0.0000 / 1.0000 when that side of the book is empty
            best_ask=_f(m.get('yes_ask_dollars')),
            **_sports_fields(m),
        )

    def list_markets(
        self,
        *,
        open_only: bool = True,
        series_ticker: str | Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """`series_ticker` restricts to one or more series (one paginated
        /markets walk per series), e.g. SPORTS_SERIES for cross-venue
        matching instead of paging through every market on the exchange."""
        params = {'limit': PAGE_LIMIT, 'mve_filter': 'exclude'}
        if open_only:
            params['status'] = 'open'  # /markets' status filter vocabulary ('open'/'closed'/'settled')
                                        # differs from a market object's own status field ('active'/'finalized')
        if series_ticker is None:
            series_list: list[str | None] = [None]
        elif isinstance(series_ticker, str):
            series_list = [series_ticker]
        else:
            series_list = list(series_ticker)

        markets: list[Market] = []
        for series in series_list:
            cursor = None
            while True:
                page_params = {**params,
                               **({'series_ticker': series} if series else {}),
                               **({'cursor': cursor} if cursor else {})}
                data = self._get('/markets', page_params)
                markets.extend(self._market_from_raw(m) for m in data.get('markets', []))
                cursor = data.get('cursor')
                if not cursor:
                    break
        return markets_to_df(markets)

    def get_market(self, market_id: str) -> pd.DataFrame:
        data = self._get(f'/markets/{market_id}')
        return markets_to_df([self._market_from_raw(data['market'])])

    def get_price_history(
        self,
        market_id: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        resolution: str = '1h',
    ) -> pd.DataFrame:
        if resolution not in _RESOLUTION_MINUTES:
            raise ValueError(f'resolution must be one of {list(_RESOLUTION_MINUTES)}, got {resolution!r}')
        end = end or datetime.now(timezone.utc)
        start = start or (end - _DEFAULT_LOOKBACK[resolution])

        market = self._get(f'/markets/{market_id}')['market']
        event_ticker = market['event_ticker']
        series_ticker = event_ticker.split('-')[0]

        data = self._get(
            f'/series/{series_ticker}/markets/{market_id}/candlesticks',
            {
                'period_interval': _RESOLUTION_MINUTES[resolution],
                'start_ts': int(start.timestamp()),
                'end_ts': int(end.timestamp()),
            },
        )
        points = [
            PricePoint(
                venue=self.name,
                market_id=market_id,
                ts=pd.to_datetime(c['end_period_ts'], unit='s', utc=True),
                price=_f(c['price']['close_dollars']),
            )
            for c in data.get('candlesticks', [])
            if c.get('price', {}).get('close_dollars') not in (None, '')
        ]
        return price_points_to_df(points)

    def get_orderbook(self, market_id: str, *, depth: int = 50) -> OrderBook:
        data = self._get(f'/markets/{market_id}/orderbook', {'depth': depth})['orderbook_fp']
        # Kalshi returns YES-side bids and NO-side bids; a YES ask = 1 - best NO bid.
        # We surface the YES book: bids = yes_dollars levels (best price last -> reverse),
        # asks = 1 - no_dollars levels (best NO bid = highest NO price = tightest YES ask).
        yes_bids = [OrderBookLevel(price=float(p), size=float(s), side='bid')
                    for p, s in reversed(data.get('yes_dollars', []))]
        no_bids = data.get('no_dollars', [])
        yes_asks = [OrderBookLevel(price=round(1 - float(p), 4), size=float(s), side='ask')
                    for p, s in reversed(no_bids)]
        return OrderBook(
            venue=self.name, market_id=market_id, ts=datetime.now(timezone.utc),
            bids=yes_bids, asks=yes_asks, raw=data,
        )
