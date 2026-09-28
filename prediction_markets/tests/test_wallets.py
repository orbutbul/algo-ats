"""
Unit tests for prediction_markets/wallets.py, mocked against Data API
payload shapes captured live 2026-09-24 -- no network calls.
"""

import math

import pandas as pd
import pytest

from prediction_markets.wallets import (
    DATA_BASE_URL,
    GAMMA_BASE_URL,
    PolymarketWalletScanner,
    score_closed_positions,
    wilson_lower_bound,
)

W1 = '0x2a69660046d7acc4ab204d7cc5ba78b0776cd2f7'
W2 = '0x0f6f76ced62a911bccef92f50faaff143854d977'


def _pos(cond, outcome_index, avg_price, bought, pnl, ts):
    return {'proxyWallet': W1, 'conditionId': cond, 'outcomeIndex': outcome_index,
            'avgPrice': avg_price, 'totalBought': bought, 'realizedPnl': pnl,
            'curPrice': 1 if pnl > 0 else 0, 'timestamp': ts}


def test_wilson_lower_bound():
    assert math.isnan(wilson_lower_bound(0, 0))
    assert wilson_lower_bound(14, 14) < 0.8        # small perfect record is weak evidence
    assert wilson_lower_bound(140, 200) > 0.63


def test_score_counts_both_sides_as_one_market():
    df = pd.DataFrame([
        _pos('0xa', 0, 0.4758, 34128.97, -16238.67, 100),   # hedged market: both outcomes held
        _pos('0xa', 1, 0.4713, 30832.07, 16298.08, 100),
        _pos('0xb', 0, 0.50, 1000, 500, 200),
        _pos('0xc', 1, 0.60, 1000, -600, 300),
    ])
    s = score_closed_positions(df)
    assert s['n_positions'] == 4
    assert s['n_markets'] == 3
    assert s['wins'] == 2                          # 0xa nets +59, 0xb wins
    assert s['hedged_share'] == pytest.approx(1 / 3)
    assert s['implied_win_rate'] == pytest.approx(0.55)  # one-sided markets only
    assert s['edge'] == pytest.approx(0.5 - 0.55)
    assert s['last_closed'] == pd.Timestamp(300, unit='s', tz='UTC')


def test_score_empty():
    assert score_closed_positions(pd.DataFrame())['n_markets'] == 0


def test_edge_z_ranks_price_beaters_above_favourite_buyers():
    # 19/20 wins buying at 0.98 is exactly what the prices implied (19.6 expected)
    fav = pd.DataFrame([_pos(f'0x{i}', 0, 0.98, 100, 2 if i < 19 else -98, i) for i in range(20)])
    # 12/20 wins buying at 0.40 beats the 8 the prices implied
    dog = pd.DataFrame([_pos(f'0x{i}', 0, 0.40, 100, 60 if i < 12 else -40, i) for i in range(20)])
    fav_s, dog_s = score_closed_positions(fav), score_closed_positions(dog)
    assert fav_s['win_rate_lb'] > dog_s['win_rate_lb']   # the old sort key got this backwards
    assert fav_s['edge_z'] == pytest.approx((19 - 19.6) / math.sqrt(20 * 0.98 * 0.02))
    assert dog_s['edge_z'] == pytest.approx((12 - 8) / math.sqrt(20 * 0.4 * 0.6))
    assert dog_s['edge_z'] > 1.5 > 0 > fav_s['edge_z']


def test_leaderboard_pages_at_50(requests_mock):
    page1 = [{'rank': str(i), 'proxyWallet': f'0x{i}', 'userName': f'u{i}', 'vol': 1.0, 'pnl': 2.0}
             for i in range(1, 51)]
    page2 = [{'rank': '51', 'proxyWallet': '0x51', 'userName': 'u51', 'vol': 1.0, 'pnl': 2.0}]
    requests_mock.get(f'{DATA_BASE_URL}/v1/leaderboard', [{'json': page1}, {'json': page2}])
    df = PolymarketWalletScanner(calls_per_second=1000).leaderboard('WEEK', top_n=51)
    assert len(df) == 51
    assert df['rank'].dtype.kind == 'i'
    assert requests_mock.request_history[1].qs['offset'] == ['50']
    assert requests_mock.request_history[1].qs['limit'] == ['1']


def test_closed_positions_stops_at_cutoff(requests_mock):
    page = [_pos(f'0x{i}', 0, 0.5, 10, 1, 2_000_000_000 - i * 1000) for i in range(50)]
    requests_mock.get(f'{DATA_BASE_URL}/closed-positions', json=page)
    since = pd.Timestamp(2_000_000_000 - 10_500, unit='s', tz='UTC').to_pydatetime()
    df = PolymarketWalletScanner(calls_per_second=1000).closed_positions(W1, since)
    assert len(df) == 11                           # rows 0..10 are inside the window
    assert len(requests_mock.request_history) == 1  # oldest row was past cutoff: no 2nd page


def test_scan_merges_periods(requests_mock):
    week = [{'rank': '1', 'proxyWallet': W1, 'userName': 'a', 'vol': 10.0, 'pnl': 5.0}]
    month = [{'rank': '1', 'proxyWallet': W2, 'userName': 'b', 'vol': 20.0, 'pnl': 9.0},
             {'rank': '2', 'proxyWallet': W1, 'userName': 'a', 'vol': 30.0, 'pnl': 7.0}]
    requests_mock.get(f'{DATA_BASE_URL}/v1/leaderboard', [{'json': week}, {'json': month}])
    requests_mock.get(f'{DATA_BASE_URL}/closed-positions', json=[])
    requests_mock.get(f'{DATA_BASE_URL}/positions', json=[])
    df = PolymarketWalletScanner(calls_per_second=1000).scan(('WEEK', 'MONTH'), top_n=2)
    assert set(df['wallet']) == {W1, W2}
    row = df.set_index('wallet').loc[W1]
    assert row['rank_week'] == 1 and row['rank_month'] == 2
    assert df.set_index('wallet').loc[W2, 'user_name'] == 'b'


def test_settled_positions_includes_unredeemed_losers(requests_mock):
    # winner was redeemed (closed-positions); loser never was (positions, redeemable)
    requests_mock.get(f'{DATA_BASE_URL}/closed-positions',
                      json=[_pos('0xa', 0, 0.5, 1000, 500, 2_000_000_000)])
    loser = {'proxyWallet': W1, 'conditionId': '0xb', 'outcomeIndex': 0, 'avgPrice': 0.4841,
             'totalBought': 120000, 'initialValue': 58099.8942, 'currentValue': 0,
             'cashPnl': -58099.8942, 'realizedPnl': -132.2658, 'curPrice': 0,
             'redeemable': True, 'endDate': '2033-05-18'}
    undated = {**loser, 'conditionId': '0xc', 'endDate': '1970-01-01'}
    requests_mock.get(f'{DATA_BASE_URL}/positions', json=[loser, undated])
    requests_mock.get(f'{GAMMA_BASE_URL}/markets', json=[])
    scanner = PolymarketWalletScanner(calls_per_second=1000)
    pos = scanner.settled_positions(W1, since=pd.Timestamp('2020-01-01', tz='UTC').to_pydatetime())
    assert list(pos['source']) == ['closed', 'unredeemed']
    assert pos['realizedPnl'].iloc[1] == pytest.approx(-58232.16)
    # no closedTime -> endDate fallback, clipped: a resolved market can't resolve in 2033
    assert pos['timestamp'].iloc[1] <= pd.Timestamp.now(tz='UTC').timestamp()
    s = score_closed_positions(pos)
    assert s['n_markets'] == 2 and s['wins'] == 1
    positions_call = next(r for r in requests_mock.request_history if r.path == '/positions')
    assert positions_call.qs['redeemable'] == ['true']


def _unredeemed(cond, end_date):
    return {'proxyWallet': W1, 'conditionId': cond, 'outcomeIndex': 0, 'avgPrice': 0.3,
            'totalBought': 100, 'cashPnl': -30, 'realizedPnl': 0, 'curPrice': 0,
            'redeemable': True, 'endDate': end_date}


def test_unredeemed_dated_by_gamma_closed_time(requests_mock):
    # "by Dec 31 2027" market that actually resolved in Sept 2026
    requests_mock.get(f'{DATA_BASE_URL}/positions',
                      json=[_unredeemed('0xa', '2027-12-31'), _unredeemed('0xb', '2026-01-01')])
    requests_mock.get(f'{GAMMA_BASE_URL}/markets', json=[
        {'conditionId': '0xa', 'closedTime': '2026-09-21 16:52:14+00'},
        {'conditionId': '0xb', 'closedTime': '2026-01-01 04:00:00+00'},
    ])
    scanner = PolymarketWalletScanner(calls_per_second=1000)
    df = scanner.unredeemed_positions(W1, since=pd.Timestamp('2026-09-01', tz='UTC').to_pydatetime())
    assert list(df['conditionId']) == ['0xa']
    assert df['timestamp'].iloc[0] == pd.Timestamp('2026-09-21 16:52:14', tz='UTC').timestamp()
    gamma = requests_mock.request_history[-1]
    assert gamma.qs['closed'] == ['true'] and gamma.qs['condition_ids'] == ['0xa', '0xb']


def test_capped_wallet_cuts_unredeemed_to_closed_span(requests_mock):
    # 2 closed rows = the cap; oldest at t=2_000_000_000. The unredeemed loser
    # resolved before that, in the part of the window the closed rows never reached.
    requests_mock.get(f'{DATA_BASE_URL}/closed-positions', json=[
        _pos('0xa', 0, 0.5, 10, 5, 2_000_000_100), _pos('0xb', 0, 0.5, 10, 5, 2_000_000_000)])
    requests_mock.get(f'{DATA_BASE_URL}/positions', json=[_unredeemed('0xc', '2030-01-01')])
    requests_mock.get(f'{GAMMA_BASE_URL}/markets', json=[
        {'conditionId': '0xc', 'closedTime': str(pd.Timestamp(1_999_999_000, unit='s', tz='UTC'))}])
    scanner = PolymarketWalletScanner(calls_per_second=1000)
    since = pd.Timestamp(1_900_000_000, unit='s', tz='UTC').to_pydatetime()
    pos = scanner.settled_positions(W1, since, max_positions=2)
    assert list(pos['source']) == ['closed', 'closed']
