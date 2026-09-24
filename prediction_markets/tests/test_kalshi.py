"""
Unit tests for prediction_markets/venues/kalshi.py, mocked with
requests_mock against payload shapes captured live in
data/prediction_markets/kalshi.md (recon 2026-09-20) -- no network calls.
"""

from prediction_markets.venues.kalshi import BASE_URL, KalshiClient

MARKET = {
    'ticker': 'KXNFLGAME-26SEP20LVLAC-LAC',
    'event_ticker': 'KXNFLGAME-26SEP20LVLAC',
    'title': 'Los Angeles C wins',
    'status': 'active',
    'close_time': '2026-09-22T20:05:00Z',
    'last_price_dollars': '0.7400',
    'volume_fp': '323783.07',
}


def test_list_markets_single_page(requests_mock):
    requests_mock.get(f'{BASE_URL}/markets', json={'markets': [MARKET], 'cursor': ''})
    df = KalshiClient().list_markets()
    assert len(df) == 1
    row = df.iloc[0]
    assert row['venue'] == 'kalshi'
    assert row['market_id'] == 'KXNFLGAME-26SEP20LVLAC-LAC'
    assert bool(row['is_open']) is True
    assert bool(row['is_resolved']) is False
    assert row['last_price'] == 0.74


def test_list_markets_paginates(requests_mock):
    page1 = {'markets': [MARKET], 'cursor': 'abc'}
    page2 = {'markets': [{**MARKET, 'ticker': 'OTHER-TICKER'}], 'cursor': ''}
    requests_mock.get(
        f'{BASE_URL}/markets',
        [{'json': page1}, {'json': page2}],
    )
    df = KalshiClient().list_markets()
    assert len(df) == 2
    assert set(df['market_id']) == {'KXNFLGAME-26SEP20LVLAC-LAC', 'OTHER-TICKER'}


def test_list_markets_series_ticker_walks_each_series(requests_mock):
    requests_mock.get(f'{BASE_URL}/markets', json={'markets': [MARKET], 'cursor': ''})
    df = KalshiClient().list_markets(series_ticker=['KXNFLGAME', 'KXNFLTOTAL'])
    assert len(df) == 2
    assert [r.qs['series_ticker'] for r in requests_mock.request_history] == [['kxnflgame'], ['kxnfltotal']]


def test_sports_fields_on_winner_market():
    row = KalshiClient()._market_from_raw({**MARKET, 'yes_sub_title': 'Los Angeles C'})
    assert (row.league, row.game_id, row.market_type, row.outcome) == ('nfl', 'nfl:26SEP20LVLAC', 'winner', 'LAC')
    assert str(row.event_date) == '2026-09-20'


def test_get_market(requests_mock):
    requests_mock.get(f'{BASE_URL}/markets/KXNFLGAME-26SEP20LVLAC-LAC', json={'market': MARKET})
    df = KalshiClient().get_market('KXNFLGAME-26SEP20LVLAC-LAC')
    assert len(df) == 1
    assert df.iloc[0]['title'] == 'Los Angeles C wins'


def test_get_price_history(requests_mock):
    requests_mock.get(f'{BASE_URL}/markets/KXNFLGAME-26SEP20LVLAC-LAC', json={'market': MARKET})
    requests_mock.get(
        f'{BASE_URL}/series/KXNFLGAME/markets/KXNFLGAME-26SEP20LVLAC-LAC/candlesticks',
        json={'candlesticks': [
            {'end_period_ts': 1789923780, 'price': {'close_dollars': '0.7400'}},
            {'end_period_ts': 1789923840, 'price': {'close_dollars': '0.7300'}},
        ]},
    )
    df = KalshiClient().get_price_history('KXNFLGAME-26SEP20LVLAC-LAC', resolution='1h')
    assert list(df['price']) == [0.74, 0.73]
    assert df['ts'].is_monotonic_increasing


def test_get_price_history_rejects_bad_resolution():
    import pytest
    with pytest.raises(ValueError):
        KalshiClient().get_price_history('X', resolution='5m')


def test_get_orderbook(requests_mock):
    requests_mock.get(
        f'{BASE_URL}/markets/KXNFLGAME-26SEP20LVLAC-LAC/orderbook',
        json={'orderbook_fp': {
            'yes_dollars': [['0.7100', '62604.34'], ['0.7200', '292630.36'], ['0.7300', '138449.20']],
            'no_dollars': [['0.2400', '1126658.28'], ['0.2500', '2122847.47'], ['0.2600', '2623161.94']],
        }},
    )
    book = KalshiClient().get_orderbook('KXNFLGAME-26SEP20LVLAC-LAC')
    assert book.best_bid == 0.73          # highest yes bid, since we reverse to best-first
    assert book.best_ask == round(1 - 0.26, 4)  # 1 - best (highest) no bid
    df = book.to_dataframe()
    assert set(df['side']) == {'bid', 'ask'}
