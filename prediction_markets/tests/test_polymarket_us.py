"""
Unit tests for prediction_markets/venues/polymarket_us.py, mocked
against payload shapes captured live in data/prediction_markets/
polymarket_us.md (recon 2026-09-20) -- no network calls.
"""

import pytest

from prediction_markets.venues.polymarket_us import (
    GATEWAY_BASE_URL,
    PolymarketUSClient,
)

MARKET = {
    'id': '1',
    'slug': 'aec-nfl-lac-ten-2025-11-02',
    'question': 'Los Angeles vs. Tennessee',
    'active': True,
    'closed': True,          # confirmed quirk: resolved market can have both true
    'endDate': '2025-11-03T13:00:00Z',
    'outcomePrices': '["1", "0"]',
}


def test_list_markets_excludes_closed_by_default(requests_mock):
    requests_mock.get(f'{GATEWAY_BASE_URL}/v1/markets', json={'markets': [MARKET], 'eof': True, 'nextCursor': None})
    df = PolymarketUSClient().list_markets(open_only=True)
    assert df.empty   # closed=True -> is_open False -> filtered out


def test_list_markets_open_only_false_includes_it(requests_mock):
    requests_mock.get(f'{GATEWAY_BASE_URL}/v1/markets', json={'markets': [MARKET], 'eof': True, 'nextCursor': None})
    df = PolymarketUSClient().list_markets(open_only=False)
    assert len(df) == 1
    assert bool(df.iloc[0]['is_resolved']) is True


def test_get_market(requests_mock):
    # confirmed live: there's no /v1/markets/{slug} detail path (404) -- the
    # gateway only supports the filtered list call for a single market's detail.
    requests_mock.get(f'{GATEWAY_BASE_URL}/v1/markets', json={'markets': [MARKET]})
    df = PolymarketUSClient().get_market(MARKET['slug'])
    assert df.iloc[0]['title'] == MARKET['question']
    assert requests_mock.last_request.qs['slug'] == [MARKET['slug']]


def test_get_market_not_found(requests_mock):
    requests_mock.get(f'{GATEWAY_BASE_URL}/v1/markets', json={'markets': []})
    with pytest.raises(ValueError):
        PolymarketUSClient().get_market('no-such-slug')


def test_get_price_history(requests_mock):
    requests_mock.get(
        f'{GATEWAY_BASE_URL}/v1/price-history',
        json={'history': [{'timestamp': 1789844400, 'longPrice': 0.76, 'shortPrice': 0.245}]},
    )
    df = PolymarketUSClient().get_price_history(MARKET['slug'], resolution='1d')
    assert list(df['price']) == [0.76]


def test_get_price_history_rejects_bad_resolution():
    with pytest.raises(ValueError):
        PolymarketUSClient().get_price_history('x', resolution='5m')


def test_get_orderbook(requests_mock):
    requests_mock.get(
        f'{GATEWAY_BASE_URL}/v1/markets/{MARKET["slug"]}/book',
        json={'marketData': {
            'bids': [{'px': {'value': '0.6850', 'currency': 'USD'}, 'qty': '131728.6400'}],
            'offers': [{'px': {'value': '0.6450', 'currency': 'USD'}, 'qty': '4276.2300'}],
        }},
    )
    book = PolymarketUSClient().get_orderbook(MARKET['slug'])
    assert book.best_bid == 0.685
    assert book.best_ask == 0.645
