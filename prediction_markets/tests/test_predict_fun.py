"""
Unit tests for prediction_markets/venues/predict_fun.py, mocked
against payload shapes confirmed live against api-testnet.predict.fun on
2026-09-22 (see the module's docstring) -- no network calls here.

get_price_history is intentionally untested beyond the NotImplementedError
check: the real `metric` query param for /timeseries wasn't determined (see
module docstring), so it can't be given a confirmed fixture yet.
"""

import pytest

from prediction_markets.venues.predict_fun import (
    TESTNET_BASE_URL,
    PredictFunClient,
)

MARKET = {
    'id': 437,
    'title': 'David Malpass',
    'question': 'Will Trump nominate David Malpass as the next Fed chair?',
    'status': 'REGISTERED',
    'tradingStatus': 'OPEN',
    'resolution': None,
    'stats': None,
    'outcomes': [
        {'name': 'Definitely', 'bestBid': {'price': 0.33, 'size': 123197.6925},
         'bestAsk': {'price': 0.37, 'size': 796428.2433}},
        {'name': 'Maybe', 'bestBid': {'price': 0.63, 'size': 796428.2433},
         'bestAsk': {'price': 0.67, 'size': 123197.6925}},
    ],
}


def test_defaults_to_testnet_no_key_required():
    client = PredictFunClient()
    assert client.base_url == TESTNET_BASE_URL


def test_mainnet_requires_api_key(monkeypatch):
    monkeypatch.delenv('PREDICT_FUN_API_KEY', raising=False)
    with pytest.raises(ValueError):
        PredictFunClient(env='mainnet')


def test_list_markets(requests_mock):
    requests_mock.get(f'{TESTNET_BASE_URL}/v1/markets', json={'success': True, 'cursor': None, 'data': [MARKET]})
    df = PredictFunClient().list_markets()
    assert len(df) == 1
    row = df.iloc[0]
    assert row['market_id'] == '437'
    assert row['title'] == 'David Malpass'
    assert row['last_price'] == pytest.approx(0.35)   # mid of first outcome's 0.33/0.37
    assert bool(row['is_open']) is True


def test_list_markets_paginates_on_after_param(requests_mock):
    # Confirmed live: the pagination param is `after`, not `cursor` --
    # passing `cursor` back is silently ignored (see predict_fun.py).
    page1 = {'success': True, 'cursor': 'NDUx', 'data': [MARKET]}
    page2 = {'success': True, 'cursor': None, 'data': [{**MARKET, 'id': 438}]}
    requests_mock.get(f'{TESTNET_BASE_URL}/v1/markets', [{'json': page1}, {'json': page2}])
    df = PredictFunClient().list_markets()
    assert set(df['market_id']) == {'437', '438'}
    assert requests_mock.request_history[1].qs['after'] == ['ndux']  # requests_mock lowercases qs values


def test_list_markets_respects_max_pages(requests_mock):
    # Same cursor every time (simulating the real bug this guards against) --
    # max_pages must still stop the loop instead of hanging forever.
    requests_mock.get(f'{TESTNET_BASE_URL}/v1/markets', json={'success': True, 'cursor': 'same', 'data': [MARKET]})
    df = PredictFunClient().list_markets(max_pages=3)
    assert len(df) == 3


def test_get_market(requests_mock):
    requests_mock.get(f'{TESTNET_BASE_URL}/v1/markets/437', json={'success': True, 'data': MARKET})
    df = PredictFunClient().get_market('437')
    assert df.iloc[0]['title'] == 'David Malpass'


def test_get_price_history_not_implemented():
    with pytest.raises(NotImplementedError):
        PredictFunClient().get_price_history('437')


def test_get_orderbook(requests_mock):
    requests_mock.get(
        f'{TESTNET_BASE_URL}/v1/markets/437/orderbook',
        json={'success': True, 'data': {
            'marketId': 437,
            'updateTimestampMs': 1790062303973,
            'bids': [[0.33, 123197.6925], [0.17, 6.0764]],
            'asks': [[0.37, 796428.2433], [0.45, 7.5211]],
        }},
    )
    book = PredictFunClient().get_orderbook('437')
    assert book.best_bid == 0.33
    assert book.best_ask == 0.37
