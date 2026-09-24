"""
Unit tests for prediction_markets/matching.py. Fixtures are trimmed live
payloads (2026-09-24) run through each client's real _market_from_raw, so
these also cover the Kalshi / Polymarket.us sports-field parsing the
matcher depends on -- no network calls.
"""

import pandas as pd
import pytest

from prediction_markets.matching import MATCH_COLUMNS, match_markets, team_similarity
from prediction_markets.models import markets_to_df
from prediction_markets.venues.kalshi import KalshiClient
from prediction_markets.venues.polymarket_us import PolymarketUSClient


def _kalshi(ticker, yes_sub_title, bid, ask, *, strike_type='structured', floor_strike=None):
    return {
        'ticker': ticker, 'event_ticker': ticker.rsplit('-', 1)[0], 'title': yes_sub_title,
        'status': 'active', 'yes_sub_title': yes_sub_title, 'strike_type': strike_type,
        'floor_strike': floor_strike, 'last_price_dollars': f'{bid:.4f}',
        'yes_bid_dollars': f'{bid:.4f}', 'yes_ask_dollars': f'{ask:.4f}',
    }


def _us_side(desc, long, abbr=None, name=None):
    side = {'description': desc, 'long': long}
    if abbr:
        side['team'] = {'abbreviation': abbr, 'name': name, 'alias': name, 'safeName': name}
    return side


def _us(slug, smt, sides, bid, ask, line=None):
    return {
        'slug': slug, 'question': slug, 'active': True, 'closed': False,
        'sportsMarketType': smt, 'marketSides': sides, 'line': line,
        'bestBidQuote': {'value': f'{bid:.4f}', 'currency': 'USD'},
        'bestAskQuote': {'value': f'{ask:.4f}', 'currency': 'USD'},
    }


ATL = ('atl', 'Atlanta Falcons')
GB = ('gb', 'Green Bay Packers')

KALSHI = [
    _kalshi('KXNFLGAME-26SEP24ATLGB-ATL', 'Atlanta', 0.30, 0.31),
    _kalshi('KXNFLGAME-26SEP24ATLGB-GB', 'Green Bay', 0.69, 0.70),
    _kalshi('KXNFLSPREAD-26SEP24ATLGB-GB4', 'GB Packers wins by over 3.5 points', 0.55, 0.56,
            strike_type='greater', floor_strike=3.5),
    _kalshi('KXNFLTOTAL-26SEP24ATLGB-45', 'Over 44.5 points scored', 0.52, 0.53,
            strike_type='greater', floor_strike=44.5),
    # a futures-style market with no proposition fields -- must be ignored
    _kalshi('KXSB-27-GB', 'Green Bay', 0.10, 0.11),
]
US = [
    _us('aec-nfl-atl-gb-2026-09-24', 'football_team_full_game_winner',
        [_us_side('Falcons', True, *ATL), _us_side('Packers', False, *GB)], 0.305, 0.31),
    # "ATL +3.5" long == NOT "GB wins by > 3.5"
    _us('asc-nfl-atl-gb-2026-09-24-pos-3pt5', 'football_team_full_game_spread',
        [_us_side('+3.50', True, *ATL), _us_side('-3.50', False, *GB)], 0.40, 0.41, line=3.5),
    # "GB -3.5" long == "GB wins by > 3.5"
    _us('asc-nfl-atl-gb-2026-09-24-neg-3pt5', 'football_team_full_game_spread',
        [_us_side('-3.50', True, *GB), _us_side('+3.50', False, *ATL)], 0.58, 0.59, line=-3.5),
    _us('tsc-nfl-atl-gb-2026-09-24-total-44pt5', 'football_team_full_game_total',
        [_us_side('Over', True), _us_side('Under', False)], 0.51, 0.52, line=44.5),
    # first-half total: not a full-game market, must be ignored
    _us('tsc-nfl-atl-gb-2026-09-24-1h-total-21pt5', 'football_team_first_half_total',
        [_us_side('Over', True), _us_side('Under', False)], 0.5, 0.51, line=21.5),
]


@pytest.fixture
def frames():
    kc, uc = KalshiClient(), PolymarketUSClient()
    return (markets_to_df([kc._market_from_raw(m) for m in KALSHI]),
            markets_to_df([uc._market_from_raw(m) for m in US]))


def test_kalshi_sports_fields(frames):
    k, _ = frames
    spread = k.set_index('market_id').loc['KXNFLSPREAD-26SEP24ATLGB-GB4']
    assert spread['league'] == 'nfl'
    assert spread['game_id'] == 'nfl:26SEP24ATLGB'
    assert str(spread['event_date']) == '2026-09-24'
    assert (spread['market_type'], spread['outcome'], spread['line']) == ('spread', 'GB', 3.5)
    assert spread['teams'] == {'GB': ('GB Packers',)}
    assert pd.isna(k.set_index('market_id').loc['KXSB-27-GB', 'market_type'])


def test_polymarket_us_sports_fields(frames):
    _, u = frames
    u = u.set_index('market_id')
    pos = u.loc['asc-nfl-atl-gb-2026-09-24-pos-3pt5']
    assert (pos['outcome'], pos['line'], bool(pos['negated'])) == ('gb', 3.5, True)
    neg = u.loc['asc-nfl-atl-gb-2026-09-24-neg-3pt5']
    assert (neg['outcome'], neg['line'], bool(neg['negated'])) == ('gb', 3.5, False)
    assert u.loc['aec-nfl-atl-gb-2026-09-24', 'last_price'] == pytest.approx(0.3075)   # bid/ask midpoint
    assert pd.isna(u.loc['tsc-nfl-atl-gb-2026-09-24-1h-total-21pt5', 'market_type'])


def test_match_markets_pairs_and_orientation(frames):
    m = match_markets(*frames)
    assert list(m.columns) == MATCH_COLUMNS
    got = {(r.left_market_id, r.right_market_id): r.orientation for r in m.itertuples()}
    assert got == {
        ('KXNFLGAME-26SEP24ATLGB-ATL', 'aec-nfl-atl-gb-2026-09-24'): 'same',
        ('KXNFLGAME-26SEP24ATLGB-GB', 'aec-nfl-atl-gb-2026-09-24'): 'inverted',
        ('KXNFLSPREAD-26SEP24ATLGB-GB4', 'asc-nfl-atl-gb-2026-09-24-neg-3pt5'): 'same',
        ('KXNFLSPREAD-26SEP24ATLGB-GB4', 'asc-nfl-atl-gb-2026-09-24-pos-3pt5'): 'inverted',
        ('KXNFLTOTAL-26SEP24ATLGB-45', 'tsc-nfl-atl-gb-2026-09-24-total-44pt5'): 'same',
    }


def test_inverted_prices_are_aligned_and_arb_edge(frames):
    m = match_markets(*frames).set_index(['left_market_id', 'right_market_id'])
    row = m.loc[('KXNFLSPREAD-26SEP24ATLGB-GB4', 'asc-nfl-atl-gb-2026-09-24-pos-3pt5')]
    # .us ATL+3.5 bid/ask 0.40/0.41 -> GB-by->3.5 bid/ask 0.59/0.60
    assert (row['right_bid_aligned'], row['right_ask_aligned']) == (0.59, 0.60)
    # Kalshi ask 0.56 < aligned .us bid 0.59: buy Kalshi YES + .us ATL+3.5 for 0.56 + 0.41 = 0.97
    assert row['arb_edge'] == pytest.approx(0.03)


def test_no_match_across_dates(frames):
    k, u = frames
    u = u.assign(event_date=pd.Timestamp('2026-09-25').date())
    assert match_markets(k, u).empty


def test_ufc_matches_on_names_when_codes_differ():
    kc, uc = KalshiClient(), PolymarketUSClient()
    k = markets_to_df([kc._market_from_raw(m) for m in [
        _kalshi('KXUFCFIGHT-26SEP26ROSBAR-ROS', 'Raul Rosas Jr', 0.59, 0.60),
        _kalshi('KXUFCFIGHT-26SEP26ROSBAR-BAR', 'Raoni Barcelos', 0.40, 0.41),
    ]])
    u = markets_to_df([uc._market_from_raw(_us(
        'aec-ufc-rauros-raobar-2026-09-26', 'ufc_fight_winner',
        [_us_side('Raul Rosas Jr.', True, 'rauros', 'Raul Rosas Jr.'),
         _us_side('Raoni Barcelos', False, 'raobar', 'Raoni Barcelos')], 0.59, 0.60))])
    m = match_markets(k, u)
    assert dict(zip(m['left_market_id'], m['orientation'])) == {
        'KXUFCFIGHT-26SEP26ROSBAR-ROS': 'same', 'KXUFCFIGHT-26SEP26ROSBAR-BAR': 'inverted'}


@pytest.mark.parametrize('a, b, expected_min', [
    ('Kansas St.', 'Kansas State', 1.0),
    ('UConn', 'Connecticut', 1.0),
    ('New England', 'New England Patriots', 0.9),
    ('Los Angeles C', 'Los Angeles Chargers', 0.85),
])
def test_team_similarity_matches(a, b, expected_min):
    assert team_similarity('X', (a,), 'y', (b,)) >= expected_min


def test_team_similarity_rejects_different_schools():
    assert team_similarity('X', ('Michigan',), 'y', ('Eastern Michigan',)) < 1.0
    assert team_similarity('X', ('Duke',), 'y', ('Duquesne',)) < 0.8
