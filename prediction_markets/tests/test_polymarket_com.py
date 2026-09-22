"""
Unit tests for prediction_markets/venues/polymarket_com.py, mocked
against payload shapes captured live in data/prediction_markets/
polymarket_com.md (recon 2026-09-20) -- no network calls.
"""

import pytest

from prediction_markets.venues.polymarket_com import (
    CLOB_BASE_URL,
    GAMMA_BASE_URL,
    PolymarketComClient,
)

MARKET = {
    'id': '1130012',
    'question': 'Will United Russia (ER) gain the most seats?',
    'slug': 'will-united-russia-er-gain-the-most-seats',
    'outcomes': '["Yes", "No"]',
    'outcomePrices': '["0.955", "0.045"]',
    'clobTokenIds': '["20915769520649892253891152116814645067070024223185517956799957803974344024878", "1153510755"]',
    'endDate': '2026-09-30T00:00:00Z',
    'active': True,
    'closed': False,
    'volumeNum': 18448682.096538983,
}


def test_list_markets(requests_mock):
    requests_mock.get(f'{GAMMA_BASE_URL}/markets', [{'json': [MARKET]}, {'json': []}])
    df = PolymarketComClient().list_markets()
    assert len(df) == 1
    row = df.iloc[0]
    assert row['market_id'] == MARKET['slug']
    assert row['last_price'] == 0.955
    assert bool(row['is_open']) is True


def test_get_market(requests_mock):
    requests_mock.get(f'{GAMMA_BASE_URL}/markets/slug/{MARKET["slug"]}', json=MARKET)
    df = PolymarketComClient().get_market(MARKET['slug'])
    assert df.iloc[0]['title'] == MARKET['question']


def test_get_price_history_default_outcome(requests_mock):
    requests_mock.get(f'{GAMMA_BASE_URL}/markets/slug/{MARKET["slug"]}', json=MARKET)
    requests_mock.get(
        f'{CLOB_BASE_URL}/prices-history',
        json={'history': [{'t': 1789844593, 'p': 0.775}, {'t': 1789844653, 'p': 0.78}]},
    )
    df = PolymarketComClient().get_price_history(MARKET['slug'])
    assert list(df['price']) == [0.775, 0.78]
    # token id for outcome 0 was passed through
    last_request = requests_mock.request_history[-1]
    assert last_request.qs['market'][0] == '20915769520649892253891152116814645067070024223185517956799957803974344024878'


def test_get_price_history_rejects_bad_resolution():
    with pytest.raises(ValueError):
        PolymarketComClient().get_price_history('x', resolution='5m')


def test_get_orderbook(requests_mock):
    requests_mock.get(f'{GAMMA_BASE_URL}/markets/slug/{MARKET["slug"]}', json=MARKET)
    requests_mock.get(
        f'{CLOB_BASE_URL}/book',
        json={
            'timestamp': '1789930979994',
            'bids': [{'price': '0.94', 'size': '78215.63'}, {'price': '0.95', 'size': '162069.55'}],
            'asks': [{'price': '0.98', 'size': '194715.39'}, {'price': '0.97', 'size': '18590.27'}],
        },
    )
    book = PolymarketComClient().get_orderbook(MARKET['slug'])
    assert book.best_bid == 0.95
    assert book.best_ask == 0.97
