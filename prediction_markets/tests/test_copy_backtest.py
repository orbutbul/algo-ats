"""
Unit tests for prediction_markets/copy_backtest.py: synthetic price paths for
the simulation, and /activity + Gamma payload shapes captured live 2026-09-30
for the fetchers -- no network calls.
"""

import numpy as np
import pandas as pd
import pytest

from prediction_markets.copy_backtest import (
    DATA_BASE_URL,
    GAMMA_BASE_URL,
    WalletCopyBacktest,
    asof_price,
    backtest,
    daily_marks,
    wallet_events,
)

W = '0xeb34b86ca3ca64eb3cee6c9a0cce668385df7268'
T0 = 1_790_000_000 - 1_790_000_000 % 86400 + 3600   # 01:00 UTC, so a daily mark falls inside


def _trade(ts, token, cond, side, size, price):
    return {'type': 'TRADE', 'timestamp': ts, 'asset': token, 'conditionId': cond, 'side': side,
            'size': size, 'price': price, 'usdcSize': size * price, 'transactionHash': f'0x{ts}{side}'}


def _leg(kind, ts, cond, size):
    return {'type': kind, 'timestamp': ts, 'asset': '', 'conditionId': cond, 'side': '',
            'size': size, 'price': 0, 'usdcSize': size, 'transactionHash': f'0x{ts}{kind}'}


TOKENS = pd.DataFrame([
    {'token': 'yes', 'condition_id': '0xc', 'outcome_index': 0, 'outcome': 'Yes', 'question': 'Q?',
     'payout': 1.0, 'resolved_ts': T0 + 2 * 86400},
    {'token': 'no', 'condition_id': '0xc', 'outcome_index': 1, 'outcome': 'No', 'question': 'Q?',
     'payout': 0.0, 'resolved_ts': T0 + 2 * 86400},
])


def _prices(token, points):
    return pd.DataFrame([{'token': token, 'ts': T0 + dt, 'price': p} for dt, p in points])


def test_asof_price_never_looks_ahead():
    prices = _prices('yes', [(0, 0.4), (60, 0.5)])
    got = asof_price(prices, pd.Series(['yes'] * 4), pd.Series([T0 - 1, T0, T0 + 59, T0 + 60]))
    assert np.isnan(got.iloc[0])
    assert list(got.iloc[1:]) == [0.4, 0.4, 0.5]


def test_wallet_events_sell_fraction_and_legs():
    act = pd.DataFrame([
        _trade(T0, 'yes', '0xc', 'BUY', 100, 0.4),
        _trade(T0 + 10, 'yes', '0xc', 'SELL', 25, 0.5),
        _leg('SPLIT', T0 + 20, '0xc', 50),             # +50 yes, +50 no
        _trade(T0 + 30, 'no', '0xc', 'SELL', 50, 0.4),  # sells the split's No leg
        _leg('MERGE', T0 + 40, '0xc', 10),
        _trade(T0 + 50, 'gone', '0xz', 'BUY', 1, 0.5),  # market Gamma didn't return
    ])
    ev = wallet_events(act, TOKENS)
    assert list(ev['kind']) == ['TRADE', 'TRADE', 'SPLIT', 'SPLIT', 'TRADE', 'MERGE', 'MERGE']
    sells = ev[ev['side'] == 'SELL'].set_index(['ts', 'token'])['sell_frac']
    assert sells[(T0 + 10, 'yes')] == pytest.approx(0.25)
    assert sells[(T0 + 30, 'no')] == pytest.approx(1.0)
    assert sells[(T0 + 40, 'yes')] == pytest.approx(10 / 125)   # 100 - 25 + 50
    assert sells[(T0 + 40, 'no')] == pytest.approx(1.0)         # nothing left: clipped


def test_backtest_fill_and_lag_pnl():
    # buy 100 yes at 0.40; a minute later it is 0.50; sell half at 0.60; resolves Yes
    act = pd.DataFrame([_trade(T0, 'yes', '0xc', 'BUY', 100, 0.4),
                        _trade(T0 + 3600, 'yes', '0xc', 'SELL', 50, 0.6)])
    prices = _prices('yes', [(-60, 0.40), (0, 0.40), (60, 0.50), (3600, 0.60), (3660, 0.70)])
    ev = wallet_events(act, TOKENS)
    res = backtest(W, ev, TOKENS, prices, lags=[60], slippage=0.0, stake=None, end_ts=T0 + 3 * 86400)
    s = res.summary.set_index('scenario')

    # fill: $40 buys 100 sh; 50 sold at 0.60 (+$30), 50 settle at $1 (+$50) -> +$40
    assert s.at['wallet fill', 'invested'] == pytest.approx(40)
    assert s.at['wallet fill', 'pnl'] == pytest.approx(40)
    assert s.at['wallet fill', 'capital_needed'] == pytest.approx(40)
    # lag 1m: $40 buys 80 sh at 0.50; 40 sold at 0.70 (+$28), 40 settle (+$40) -> +$28
    assert s.at['lag 1m', 'pnl'] == pytest.approx(28)
    assert s.at['lag 1m', 'pnl_vs_fill'] == pytest.approx(0.7)
    assert s.at['lag 1m', 'buys_copied'] == 1 and s.at['lag 1m', 'buys_skipped'] == 0
    assert res.equity['lag 1m'].iloc[-1] == pytest.approx(28)

    d = res.drift.iloc[0]
    assert d['drift_1m'] == pytest.approx(0.1)
    assert d['drift_1d'] == pytest.approx(0.3)     # last price before T0+1d is 0.70
    assert d['drift_final'] == pytest.approx(0.6)  # resolved at 1


def test_backtest_skips_buys_after_resolution_and_marks_open_markets():
    tokens = TOKENS.assign(resolved_ts=[T0 + 30, np.nan], payout=[1.0, np.nan])
    act = pd.DataFrame([_trade(T0, 'yes', '0xc', 'BUY', 100, 0.9),
                        _trade(T0, 'no', '0xc', 'BUY', 100, 0.2)])
    prices = pd.concat([_prices('yes', [(0, 0.9)]), _prices('no', [(0, 0.2), (86400, 0.3)])])
    ev = wallet_events(act, tokens)
    res = backtest(W, ev, tokens, prices, lags=[60], slippage=0.01, stake=10, end_ts=T0 + 2 * 86400)
    s = res.summary.set_index('scenario')
    assert s.at['wallet fill', 'pnl'] == pytest.approx(10 / 0.9 * 0.1 + 10 / 0.2 * 0.1)
    # yes resolved 30s in, so the 1m follower never got it; no is open, marked at 0.30
    assert s.at['lag 1m', 'buys_skipped'] == 1
    assert s.at['lag 1m', 'pnl'] == pytest.approx(10 / 0.21 * 0.3 - 10)
    assert s.at['lag 1m', 'pnl_unresolved'] == pytest.approx(s.at['lag 1m', 'pnl'])


def test_activity_pages_past_offset_cap(requests_mock, monkeypatch):
    monkeypatch.setattr('prediction_markets.copy_backtest.ACTIVITY_MAX_OFFSET', 4)
    monkeypatch.setattr('prediction_markets.copy_backtest.ACTIVITY_PAGE_LIMIT', 2)
    rows = [_trade(T0 + i // 2, 'yes', '0xc', 'BUY', 1 + i, 0.5) for i in range(7)]
    # offset 0, 2 from start=T0; cap hit -> restart at the last ts seen (T0+1)
    requests_mock.get(f'{DATA_BASE_URL}/activity', [
        {'json': rows[0:2]}, {'json': rows[2:4]}, {'json': rows[2:4]}, {'json': rows[4:6]}, {'json': rows[6:]}])
    df = WalletCopyBacktest(calls_per_second=1000).activity(W, 'TRADE', T0, T0 + 100)
    assert list(df['size']) == [1, 2, 3, 4, 5, 6, 7]
    qs = [r.qs for r in requests_mock.request_history]
    assert qs[2]['start'] == [str(T0 + 1)] and qs[2]['offset'] == ['0']
    assert qs[0]['sortdirection'] == ['asc'] and qs[0]['type'] == ['trade']


def test_activity_windows_fetch_in_parallel_and_dedupe_boundaries(requests_mock, monkeypatch):
    monkeypatch.setattr('prediction_markets.copy_backtest.ACTIVITY_WINDOW_SECONDS', 50)
    rows = [_trade(T0 + ts, 'yes', '0xc', 'BUY', ts, 0.5) for ts in (0, 20, 50, 70, 100)]

    def server(request, context):   # honours start/end like the real endpoint
        lo, hi = int(request.qs['start'][0]), int(request.qs['end'][0])
        return [r for r in rows if lo <= r['timestamp'] <= hi]

    requests_mock.get(f'{DATA_BASE_URL}/activity', json=server)
    df = WalletCopyBacktest(calls_per_second=1000).activity(W, 'TRADE', T0, T0 + 100)
    assert list(df['size']) == [0, 20, 50, 70, 100]   # T0+50 is in both windows, kept once
    assert len(requests_mock.request_history) == 2
    assert list(df.columns) == ['type', 'transactionHash', 'asset', 'conditionId', 'side', 'size',
                                'price', 'timestamp', 'usdcSize']


def test_market_tokens_parses_gamma(requests_mock):
    closed = {'conditionId': '0xc', 'question': 'Q?', 'closed': True, 'outcomes': '["Yes", "No"]',
              'outcomePrices': '["0", "1"]', 'clobTokenIds': '["111", "222"]',
              'closedTime': '2026-09-21 16:52:14+00', 'endDate': '2026-12-31T00:00:00Z'}
    open_ = {'conditionId': '0xd', 'question': 'R?', 'closed': False, 'outcomes': '["Yes", "No"]',
             'outcomePrices': '["0.3", "0.7"]', 'clobTokenIds': '["333", "444"]', 'endDate': '2026-12-31'}
    requests_mock.get(f'{GAMMA_BASE_URL}/markets',
                      json=lambda req, ctx: [closed] if req.qs['closed'] == ['true'] else [open_])
    df = WalletCopyBacktest(calls_per_second=1000).market_tokens(['0xc', '0xd']).set_index('token')
    assert list(df.index) == ['111', '222', '333', '444']
    assert df.at['222', 'payout'] == 1.0 and df.at['111', 'outcome'] == 'Yes'
    assert df.at['111', 'resolved_ts'] == pd.Timestamp('2026-09-21 16:52:14', tz='UTC').timestamp()
    assert df.loc[['333', '444'], ['payout', 'resolved_ts']].isna().all().all()


def test_split_then_sell_is_copied_at_par():
    # mint 100 sets for $100, sell the No leg: a synthetic Yes buy at 1 - 0.62
    act = pd.DataFrame([_leg('SPLIT', T0, '0xc', 100), _trade(T0 + 60, 'no', '0xc', 'SELL', 100, 0.62)])
    prices = pd.concat([_prices('yes', [(0, 0.40), (60, 0.39), (120, 0.37)]),
                        _prices('no', [(0, 0.60), (60, 0.61), (120, 0.63)])])
    ev = wallet_events(act, TOKENS)
    res = backtest(W, ev, TOKENS, prices, lags=[0, 60], slippage=0.01, stake=None, end_ts=T0 + 3 * 86400)
    s = res.summary.set_index('scenario')
    assert list(s['invested']) == pytest.approx([100] * 3)         # a set always costs $1, no slippage
    assert s.at['wallet fill', 'pnl'] == pytest.approx(62)       # +62 No sale, Yes pays 100
    assert s.at['lag 0s', 'pnl'] == pytest.approx(60)            # sells at 0.61 - 0.01
    assert s.at['lag 1m', 'pnl'] == pytest.approx(62)            # sells at 0.63 - 0.01
    assert s.at['lag 1m', 'buys_copied'] == 2                    # one split = two legs
    per = res.per_token[res.per_token['scenario'] == 'wallet fill'].set_index('token')
    assert per.at['yes', 'invested'] == pytest.approx(40)        # $100 split 0.40 / 0.60
    # the sale was right (No went to 0): drift in the wallet's favour is positive
    d = res.drift.set_index('side').loc['SELL']
    assert d['drift_1m'] == pytest.approx(0.62 - 0.63) and d['drift_final'] == pytest.approx(0.62)


def test_batches_match_one_pass(monkeypatch):
    # the two tokens overlap in time; one batch each must give the same totals
    act = pd.DataFrame([_trade(T0, 'yes', '0xc', 'BUY', 100, 0.4),
                        _trade(T0 + 60, 'no', '0xc', 'BUY', 50, 0.5),
                        _trade(T0 + 3600, 'yes', '0xc', 'SELL', 50, 0.6)])
    prices = pd.concat([_prices('yes', [(0, 0.40), (60, 0.50), (3600, 0.60), (3660, 0.70)]),
                        _prices('no', [(0, 0.60), (60, 0.50), (120, 0.45), (86400, 0.2)])])
    ev = wallet_events(act, TOKENS)
    end_ts = T0 + 3 * 86400
    one = backtest(W, ev, TOKENS, prices, [60], 0.01, None, end_ts)

    bt = WalletCopyBacktest()
    monkeypatch.setattr('prediction_markets.copy_backtest.TOKENS_PER_BATCH', 1)
    monkeypatch.setattr(bt, 'token_prices',
                        lambda e, *a: (prices[prices['token'].isin(e['token'])], {}))
    two = bt.backtest_events(W, ev, TOKENS, [60], 0.01, None, end_ts)
    pd.testing.assert_frame_equal(one.summary, two.summary)
    pd.testing.assert_frame_equal(one.equity, two.equity)
    assert one.summary.set_index('scenario').at['wallet fill', 'capital_needed'] == pytest.approx(65)


def test_daily_marks_only_inside_each_tokens_life():
    orders = pd.DataFrame({'token': ['yes', 'no'], 't': [T0, T0 + 86400]})
    tokens = TOKENS.set_index('token').assign(resolved_ts=[T0 + 2 * 86400, np.nan])
    prices = pd.concat([_prices('yes', [(0, 0.4)]), _prices('no', [(0, 0.6)])])
    m = daily_marks(orders, tokens, prices, end_ts=T0 + 3 * 86400 + 60)
    midnight = T0 - 3600
    assert sorted(zip(m['token'], m['t_ns'] // 10**9)) == [
        ('no', midnight + 2 * 86400), ('no', midnight + 3 * 86400),
        ('yes', midnight + 86400), ('yes', midnight + 2 * 86400)]
