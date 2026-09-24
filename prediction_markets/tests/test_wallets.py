"""
Unit tests for prediction_markets/wallets.py, mocked against Data API
payload shapes captured live 2026-09-24 -- no network calls.
"""

import math

import pandas as pd
import pytest

from prediction_markets.wallets import (
    DATA_BASE_URL,
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
    scanner = PolymarketWalletScanner(calls_per_second=1000)
    pos = scanner.settled_positions(W1, since=pd.Timestamp('2033-01-01', tz='UTC').to_pydatetime())
    assert list(pos['source']) == ['closed', 'unredeemed']
    assert pos['realizedPnl'].iloc[1] == pytest.approx(-58232.16)
    s = score_closed_positions(pos)
    assert s['n_markets'] == 2 and s['wins'] == 1
    assert requests_mock.request_history[-1].qs['redeemable'] == ['true']
