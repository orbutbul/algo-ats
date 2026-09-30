"""
prediction_markets/copy_backtest.py — would copying a Polymarket.com (global)
wallet have made money? Research layer: replays one wallet's trades as a
follower who sees each fill `lag` seconds late, then scores that follower.
Implements features E1 (post-entry drift) and E2 (simulated copy PnL) from
data/prediction_markets/wallet_features.md. Read-only: no orders are placed.

Data (all public, keyless; verified live 2026-09-30):

  GET data-api.polymarket.com/activity?user=&type=&start=&end=&sortDirection=ASC
      The wallet's history. type=TRADE rows are exactly /trades?takerOnly=false
      -- plain /trades defaults to taker-only and silently drops maker fills.
      500 rows/page and offset is capped at 5000, so paging moves `start`
      forward past the cap. SPLIT/MERGE rows carry no asset: they move `size`
      shares of every outcome of the market at once.
  GET gamma-api.polymarket.com/markets?condition_ids=..&closed=true|false
      Outcome token ids, resolution time (`closedTime`) and payout
      (`outcomePrices`). Closed and open markets need separate calls.
  GET clob.polymarket.com/prices-history?market=<token>&startTs=&endTs=&fidelity=1
      One-minute price path per outcome token. Windows over ~15d at fidelity 1
      are rejected, so this pulls 7-day chunks. Markets resolved long ago keep
      only coarse history; those fall back to fidelity 60 then 720 and are
      flagged in `tokens['fidelity']` (lag results on them are rough).

The follower model, per scenario:
  - "wallet fill" replays the wallet's own fill prices, no slippage. It is the
    ceiling a copier cannot beat, and roughly the wallet's own result on the
    trades it made in the window.
  - "lag N" enters each of the wallet's BUYs N seconds later at the market price
    then plus `slippage` (prices-history is a last/mid price, not your fill),
    and mirrors each SELL by selling the same *fraction* of the position,
    N seconds later, minus `slippage`. Fills at or after resolution are skipped.
    "lag 0s" uses the market price at the wallet's own timestamp: the gap
    between it and "wallet fill" is price-series basis, not delay.
  - Whatever is still held settles at the resolution payout (1/0); positions in
    unresolved markets are marked at the last price.
  - Buys are sized at the wallet's own USDC notional (so PnL is directly
    comparable) or at a fixed `stake` per fill.
  - SPLITs and MERGEs are copied as what they are: minting / redeeming full
    sets at exactly $1, no slippage. Many profitable wallets mostly split and
    sell one leg (a synthetic buy of the other), so skipping splits would miss
    most of their book. The $1 is allocated across the legs by their market
    prices, which only moves pnl between the two tokens, never the total.
    Holdings from before the window are invisible, so a sell of them counts as
    selling everything the follower holds in that token.

Simulation is vbt.Portfolio.from_orders: rows = event timestamps (plus a daily
mark), columns = outcome tokens, buys as SizeType.Value, sells as
SizeType.Percent of the current position, init_cash='auto' so the capital a
follower needed falls out of the cash flows. Tokens are independent, so they
are simulated in column chunks to keep the matrices small.

Usage:
    uv run python -m prediction_markets.copy_backtest 0xeb34b86ca3ca64eb3cee6c9a0cce668385df7268 --days 30

    from prediction_markets.copy_backtest import WalletCopyBacktest
    res = WalletCopyBacktest().run('0x...', days=30, lags=(60, 300, 3600))
    res.summary; res.drift_summary; res.equity.plot()
"""

from __future__ import annotations

import argparse
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import vectorbtpro as vbt

from prediction_markets.http import RateLimiter, get_json, log
from prediction_markets.venues.polymarket_com import CLOB_BASE_URL, _parse_json_field
from prediction_markets.wallets import DATA_BASE_URL, GAMMA_BASE_URL, GAMMA_IDS_PER_CALL

CALLS_PER_SECOND = 15              # /activity allows 200/10s, CLOB /prices-history 1000/10s
ACTIVITY_PAGE_LIMIT = 500          # server cap
ACTIVITY_MAX_OFFSET = 5000         # server rejects deeper offsets
PRICE_CHUNK_SECONDS = 7 * 86400
FALLBACK_FIDELITIES = (60, 720)    # minutes; old resolved tokens lose 1-minute history
DEFAULT_LAGS = (0, 60, 300, 3600)
FETCH_THREADS = 8                  # prices-history is latency-bound; the limiter still paces
DRIFT_HORIZONS = (60, 300, 3600, 86400)
MAX_PRICE = 0.999
COLUMNS_PER_SIM = 200
OUT_DIR = Path(__file__).parent.parent / 'data' / 'prediction_markets' / 'copy_backtest'
_ACTIVITY_KEY = ['type', 'transactionHash', 'asset', 'conditionId', 'side', 'size', 'price', 'timestamp']
_VALUE = vbt.pf_enums.SizeType.Value
_PERCENT = vbt.pf_enums.SizeType.Percent


def fmt_seconds(s: int) -> str:
    if s == 0:
        return '0s'
    for unit, n in (('d', 86400), ('h', 3600), ('m', 60)):
        if s % n == 0:
            return f'{s // n}{unit}'
    return f'{s}s'


def asof_price(prices: pd.DataFrame, tokens: pd.Series, times: pd.Series) -> pd.Series:
    """Last known price of each token at or before each time (unix s) -- never
    a later one, so a lagged entry can't peek ahead. NaN before the first point."""
    left = pd.DataFrame({'token': tokens.to_numpy(), 'ts': times.to_numpy().astype('int64'),
                         'i': np.arange(len(times))}).sort_values('ts', kind='stable')
    right = prices[['token', 'ts', 'price']].astype({'ts': 'int64'}).sort_values('ts', kind='stable')
    merged = pd.merge_asof(left, right, on='ts', by='token', direction='backward')
    return pd.Series(merged.sort_values('i')['price'].to_numpy(), index=times.index)


def wallet_events(activity: pd.DataFrame, tokens: pd.DataFrame) -> pd.DataFrame:
    """One row per change in the wallet's holding of one outcome token, in time
    order, with its running `position` and, for sells, `sell_frac` = share of the
    position sold. SPLIT/MERGE become one row per outcome, sharing `leg_of`.
    `activity` = /activity rows; `tokens` = market_tokens()."""
    trades = activity[activity['type'] == 'TRADE']
    legs = activity[activity['type'].isin(['SPLIT', 'MERGE'])].rename_axis('leg_of').reset_index().merge(
        tokens[['condition_id', 'token']], left_on='conditionId', right_on='condition_id')
    ev = pd.concat([
        pd.DataFrame({'ts': trades['timestamp'], 'token': trades['asset'],
                      'condition_id': trades['conditionId'], 'kind': 'TRADE', 'side': trades['side'],
                      'shares': trades['size'], 'price': trades['price'], 'usdc': trades['usdcSize']}),
        pd.DataFrame({'ts': legs['timestamp'], 'token': legs['token'], 'condition_id': legs['condition_id'],
                      'kind': legs['type'], 'side': np.where(legs['type'] == 'SPLIT', 'BUY', 'SELL'),
                      'shares': legs['size'], 'price': np.nan, 'usdc': np.nan, 'leg_of': legs['leg_of']}),
    ], ignore_index=True)
    ev = ev[ev['token'].isin(tokens['token'])].sort_values('ts', kind='stable').reset_index(drop=True)
    delta = ev['shares'].where(ev['side'] == 'BUY', -ev['shares'])
    ev['position'] = delta.groupby(ev['token']).cumsum()
    before = ev['position'] - delta
    frac = (ev['shares'] / before.where(before > 1e-9)).fillna(1.0).clip(upper=1.0)
    ev['sell_frac'] = frac.where(ev['side'] == 'SELL')
    return ev


def copy_orders(events: pd.DataFrame, tokens: pd.DataFrame, prices: pd.DataFrame, lag: int | None,
                slippage: float, stake: float | None, end_ts: int) -> pd.DataFrame:
    """The follower's orders for one scenario (lag=None: the wallet's own fill
    prices), plus a settlement sell of everything left at each resolution.
    `tokens` is indexed by token id."""
    ev = events
    t = ev['ts'] + (lag or 0)
    live = (t < ev['token'].map(tokens['resolved_ts']).fillna(np.inf)) & (t <= end_ts)
    ev, t = ev[live], t[live]
    mkt = asof_price(prices, ev['token'], t)
    buy = (ev['side'] == 'BUY').to_numpy()
    if lag is None:
        price = ev['price']
    else:
        price = (mkt + np.where(buy, slippage, -slippage)).clip(0, MAX_PRICE)
    # split/merge legs trade at par: market prices rescaled to sum to $1 per set
    leg = ev['kind'] != 'TRADE'
    sets = ev.loc[leg, 'leg_of']
    unpriced = mkt[leg].isna().groupby(sets).transform('any')
    price = price.where(~leg, (mkt[leg] / mkt[leg].groupby(sets).transform('sum')).where(~unpriced))
    # a fixed stake buys one full set's worth when the wallet splits
    notional = stake * price.where(leg, 1.0) if stake else ev['usdc'].where(~leg, ev['shares'] * price)
    orders = pd.DataFrame({
        't': t, 'token': ev['token'], 'side': ev['side'], 'kind': ev['kind'],
        'size': np.where(buy, notional, -ev['sell_frac']),
        'size_type': np.where(buy, _VALUE, _PERCENT), 'price': price, 'mark': mkt.fillna(price),
    })
    held = tokens[tokens.index.isin(orders['token']) & (tokens['resolved_ts'] <= end_ts)]
    settle = pd.DataFrame({'t': held['resolved_ts'].astype('int64'), 'token': held.index, 'side': 'SELL',
                           'kind': 'RESOLVE', 'size': -1.0, 'size_type': _PERCENT,
                           'price': held['payout'], 'mark': held['payout']})
    orders = pd.concat([orders, settle], ignore_index=True).dropna(subset=['price'])
    orders = orders.sort_values('t', kind='stable').reset_index(drop=True)
    # one order per (time, token) cell: same-second fills get 1ns apart, in order
    orders['t_ns'] = orders['t'] * 10**9 + orders.groupby(['token', 't']).cumcount()
    return orders


def daily_marks(orders: pd.DataFrame, tokens: pd.DataFrame, prices: pd.DataFrame, end_ts: int) -> pd.DataFrame:
    """Midnight-UTC mark price per token from its first order until it resolves
    (or `end_ts`), so the equity curve moves between the wallet's own trades."""
    if orders.empty:
        return pd.DataFrame(columns=['t_ns', 'token', 'mark'])
    span = orders.groupby('token')['t'].min().to_frame('first')
    span['last'] = tokens['resolved_ts'].reindex(span.index).fillna(end_ts).clip(upper=end_ts)
    days = np.arange((span['first'].min() // 86400 + 1) * 86400, end_ts + 1, 86400)
    grid = span.reset_index().merge(pd.DataFrame({'t': days}), how='cross')
    grid = grid[(grid['t'] > grid['first']) & (grid['t'] < grid['last'])]
    grid = grid.assign(mark=asof_price(prices, grid['token'], grid['t']), t_ns=grid['t'] * 10**9)
    return grid.dropna(subset=['mark'])[['t_ns', 'token', 'mark']]


def simulate(orders: pd.DataFrame, marks: pd.DataFrame) -> dict:
    """Run the orders through vbt, one column per token, in column chunks.
    Returns per-token invested/pnl, the summed pnl curve and cash flow."""
    if orders.empty:
        return {'per_token': pd.DataFrame(columns=['invested', 'pnl']),
                'pnl_curve': pd.Series(dtype=float), 'cash_flow': pd.Series(dtype=float)}
    per_token, curves, flows = [], [], []
    tokens = orders.groupby('token')['t_ns'].min().sort_values().index  # neighbours share rows
    for i in range(0, len(tokens), COLUMNS_PER_SIM):
        cols = tokens[i:i + COLUMNS_PER_SIM]
        long = pd.concat([orders[orders['token'].isin(cols)], marks[marks['token'].isin(cols)]])
        long = long.drop_duplicates(['t_ns', 'token'])  # a mark never overrides an order
        wide = long.pivot(index='t_ns', columns='token', values=['mark', 'size', 'size_type', 'price'])
        wide.index = pd.to_datetime(wide.index, utc=True)
        pf = vbt.Portfolio.from_orders(
            close=wide['mark'], size=wide['size'],
            size_type=wide['size_type'].fillna(_VALUE).astype(int),
            price=wide['price'].fillna(wide['mark']),
            direction='longonly', init_cash='auto', ffill_val_price=True,
        )
        cash_flow = pf.cash_flow
        per_token.append(pd.DataFrame({'invested': -cash_flow.clip(upper=0).sum(),
                                       'pnl': pf.total_profit}))
        curves.append((pf.value - pf.init_cash).ffill().fillna(0).sum(axis=1))
        flows.append(cash_flow.sum(axis=1))
    join = lambda parts: pd.concat(parts, axis=1).sort_index()  # noqa: E731
    return {'per_token': pd.concat(per_token),
            'pnl_curve': join(curves).ffill().fillna(0).sum(axis=1),
            'cash_flow': join(flows).fillna(0).sum(axis=1)}


def summarize(name: str, orders: pd.DataFrame, sim: dict, tokens: pd.DataFrame,
              n_wallet_buys: int, end_ts: int) -> dict:
    per = sim['per_token'].join(tokens[['condition_id', 'resolved_ts']])
    markets = per.groupby('condition_id')[['pnl', 'invested']].sum()
    markets = markets[markets['invested'] > 0]
    pnl, invested = per['pnl'].sum(), per['invested'].sum()
    curve = sim['pnl_curve']
    capital = (-sim['cash_flow'].cumsum()).max()
    n_buys = int((orders['side'] == 'BUY').sum())
    sd = markets['pnl'].std()
    return {
        'scenario': name,
        'buys_copied': n_buys,
        'buys_skipped': n_wallet_buys - n_buys,
        'n_markets': len(markets),
        'invested': invested,
        'pnl': pnl,
        'roi': pnl / invested if invested > 0 else np.nan,
        'pnl_unresolved': per.loc[~(per['resolved_ts'] <= end_ts), 'pnl'].sum(),
        'capital_needed': capital,
        'return_on_capital': pnl / capital if capital > 0 else np.nan,
        'max_drawdown': (curve - curve.cummax()).min() if len(curve) else np.nan,
        'market_win_rate': (markets['pnl'] > 0).mean() if len(markets) else np.nan,
        # mean per-market pnl over its standard error: is the total more than luck?
        'pnl_t': markets['pnl'].mean() / (sd / math.sqrt(len(markets))) if len(markets) > 1 and sd > 0 else np.nan,
    }


def post_entry_drift(events: pd.DataFrame, tokens: pd.DataFrame, prices: pd.DataFrame,
                     horizons: Iterable[int], end_ts: int) -> pd.DataFrame:
    """One row per wallet TRADE fill: how far the price moved in the wallet's
    favour after it (up after a BUY, down after a SELL). Positive = the wallet
    was right, and a follower acting later pays that much per share. Past
    resolution the price is the payout; `drift_final` uses the payout (or the
    last price for open markets)."""
    fills = events[events['kind'] == 'TRADE']
    res_ts = fills['token'].map(tokens['resolved_ts'])
    payout = fills['token'].map(tokens['payout'])
    sign = np.where(fills['side'] == 'BUY', 1.0, -1.0)
    out = fills[['ts', 'token', 'condition_id', 'side', 'price', 'usdc']].copy()
    out['question'] = fills['token'].map(tokens['question'])
    out['outcome'] = fills['token'].map(tokens['outcome'])
    for h in horizons:
        t = fills['ts'] + h
        p = asof_price(prices, fills['token'], t).where(~(t >= res_ts), payout)
        out[f'drift_{fmt_seconds(h)}'] = sign * (p - fills['price'])
    last = asof_price(prices, fills['token'], pd.Series(end_ts, index=fills.index))
    out['drift_final'] = sign * (payout.where(res_ts <= end_ts, last) - fills['price'])
    return out


def drift_summary(drift: pd.DataFrame) -> pd.DataFrame:
    """Per side and horizon: USDC-weighted mean drift, and the share of fills
    where the price had moved the wallet's way (against a follower) by then."""
    rows = []
    for side, g in drift.groupby('side'):
        for c in [c for c in drift.columns if c.startswith('drift_')]:
            d = g.dropna(subset=[c])
            rows.append({'side': side, 'horizon': c.removeprefix('drift_'), 'n': len(d),
                         'mean_drift': np.average(d[c], weights=d['usdc']) if d['usdc'].sum() > 0 else np.nan,
                         'share_favourable': (d[c] > 0).mean()})
    return pd.DataFrame(rows)


@dataclass
class CopyBacktestResult:
    wallet: str
    summary: pd.DataFrame     # one row per scenario
    drift: pd.DataFrame       # one row per wallet TRADE fill
    equity: pd.DataFrame      # follower pnl curve, one column per scenario
    per_token: pd.DataFrame   # invested/pnl per scenario x token
    events: pd.DataFrame
    tokens: pd.DataFrame

    @property
    def drift_summary(self) -> pd.DataFrame:
        return drift_summary(self.drift)


def backtest(wallet: str, events: pd.DataFrame, tokens: pd.DataFrame, prices: pd.DataFrame,
             lags: Iterable[int], slippage: float, stake: float | None, end_ts: int) -> CopyBacktestResult:
    """Offline core of run(): every scenario over already-fetched data.
    `tokens` = market_tokens() output, one row per token id."""
    meta = tokens.set_index('token')
    lags = list(lags)
    n_wallet_buys = int((events['side'] == 'BUY').sum())   # split legs count as buys
    scenarios = {'wallet fill': None} | {f'lag {fmt_seconds(lag)}': lag for lag in lags}
    rows, curves, per_token = [], {}, []
    for name, lag in scenarios.items():
        orders = copy_orders(events, meta, prices, lag, slippage, stake, end_ts)
        sim = simulate(orders, daily_marks(orders, meta, prices, end_ts))
        rows.append(summarize(name, orders, sim, meta, n_wallet_buys, end_ts))
        curves[name] = sim['pnl_curve']
        per_token.append(sim['per_token'].assign(scenario=name))
    summary = pd.DataFrame(rows)
    base = summary['pnl'].iloc[0]
    summary['pnl_vs_fill'] = summary['pnl'] / base if base > 0 else np.nan
    per_token = pd.concat(per_token).join(meta[['question', 'outcome', 'fidelity']]
                                          if 'fidelity' in meta else meta[['question', 'outcome']])
    horizons = sorted(set(lags) | set(DRIFT_HORIZONS))
    return CopyBacktestResult(
        wallet=wallet, summary=summary,
        drift=post_entry_drift(events, meta, prices, horizons, end_ts),
        equity=pd.DataFrame(curves).sort_index().ffill().fillna(0),
        per_token=per_token.rename_axis('token').reset_index(), events=events, tokens=tokens)


class WalletCopyBacktest:
    def __init__(self, calls_per_second: float = CALLS_PER_SECOND):
        self._limiter = RateLimiter(calls_per_second)

    def activity(self, wallet: str, kind: str, start: int, end: int) -> pd.DataFrame:
        """All /activity rows of one `kind` (TRADE, SPLIT, MERGE, ...) with
        start <= timestamp <= end, oldest first."""
        rows: list[dict] = []
        cursor, offset = start, 0
        while True:
            page = get_json(f'{DATA_BASE_URL}/activity', params={
                'user': wallet, 'type': kind, 'start': cursor, 'end': end,
                'sortBy': 'TIMESTAMP', 'sortDirection': 'ASC',
                'limit': ACTIVITY_PAGE_LIMIT, 'offset': offset}, limiter=self._limiter)
            rows.extend(page)
            if len(page) < ACTIVITY_PAGE_LIMIT:
                break
            offset += len(page)
            if offset >= ACTIVITY_MAX_OFFSET:
                # restart from the last second seen; its rows repeat and are deduped below
                cursor, offset = page[-1]['timestamp'], 0
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return df.drop_duplicates(subset=[c for c in _ACTIVITY_KEY if c in df]).reset_index(drop=True)

    def market_tokens(self, condition_ids: Iterable[str]) -> pd.DataFrame:
        """One row per outcome token of the given markets: resolution time
        (unix s, NaN if open) and payout (NaN if open)."""
        ids = list(dict.fromkeys(condition_ids))
        rows: list[dict] = []
        for i in range(0, len(ids), GAMMA_IDS_PER_CALL):
            chunk = ids[i:i + GAMMA_IDS_PER_CALL]
            for closed in ('true', 'false'):
                params = [('condition_ids', c) for c in chunk] + [('closed', closed), ('limit', len(chunk))]
                for m in get_json(f'{GAMMA_BASE_URL}/markets', params=params, limiter=self._limiter):
                    token_ids = _parse_json_field(m.get('clobTokenIds'), [])
                    outcomes = _parse_json_field(m.get('outcomes'), [])
                    payouts = _parse_json_field(m.get('outcomePrices'), [])
                    is_closed = bool(m.get('closed'))
                    for j, tok in enumerate(token_ids):
                        rows.append({
                            'token': tok, 'condition_id': m['conditionId'], 'outcome_index': j,
                            'outcome': outcomes[j] if j < len(outcomes) else None,
                            'question': m.get('question', ''),
                            'payout': float(payouts[j]) if is_closed and j < len(payouts) else np.nan,
                            'resolved_at': (m.get('closedTime') or m.get('endDate')) if is_closed else None,
                        })
        df = pd.DataFrame(rows, columns=['token', 'condition_id', 'outcome_index', 'outcome',
                                         'question', 'payout', 'resolved_at'])
        when = pd.to_datetime(df['resolved_at'], utc=True, errors='coerce')
        df['resolved_ts'] = (when.astype('int64') // 10**9).where(when.notna())
        return df.drop(columns='resolved_at').drop_duplicates('token').reset_index(drop=True)

    def price_history(self, token: str, start: int, end: int) -> tuple[pd.DataFrame, int]:
        """(ts, price) points for one token up to `end`, and the fidelity (minutes) used."""
        def fetch(params):
            data = get_json(f'{CLOB_BASE_URL}/prices-history', params={'market': token, **params},
                            limiter=self._limiter)
            return data.get('history', [])

        fidelity = 1
        pts = [p for s in range(start, end, PRICE_CHUNK_SECONDS)
               for p in fetch({'startTs': s, 'endTs': min(s + PRICE_CHUNK_SECONDS, end), 'fidelity': 1})]
        for fid in FALLBACK_FIDELITIES:
            if pts:
                break
            fidelity = fid
            pts = [p for p in fetch({'interval': 'max', 'fidelity': fid}) if p['t'] <= end]
        df = pd.DataFrame(pts, columns=['t', 'p']).rename(columns={'t': 'ts', 'p': 'price'})
        return df.drop_duplicates('ts').assign(token=token), fidelity

    def token_prices(self, events: pd.DataFrame, tokens: pd.DataFrame, end_ts: int,
                     max_lag: int) -> tuple[pd.DataFrame, dict[str, int]]:
        """Price history for every token the wallet touched, from an hour before
        its first event there to the furthest lag/drift horizon or resolution."""
        horizon = max(max_lag, *DRIFT_HORIZONS) + 60
        span = events.groupby('token')['ts'].agg(['min', 'max'])
        res = tokens.set_index('token')['resolved_ts'].reindex(span.index).fillna(np.inf)
        stop = np.minimum(np.minimum(span['max'] + horizon, res + 60), end_ts).astype('int64')
        with ThreadPoolExecutor(FETCH_THREADS) as pool:
            got = list(pool.map(lambda tok: self.price_history(tok, int(span.at[tok, 'min']) - 3600,
                                                               int(stop[tok])), span.index))
        fidelity = {tok: fid for tok, (_, fid) in zip(span.index, got)}
        prices = (pd.concat([df for df, _ in got], ignore_index=True) if got
                  else pd.DataFrame(columns=['ts', 'price', 'token']))
        return prices, fidelity

    def run(self, wallet: str, days: int = 30, lags: Iterable[int] = DEFAULT_LAGS,
            slippage: float = 0.01, stake: float | None = None,
            end: datetime | None = None) -> CopyBacktestResult:
        """Backtest copying `wallet` over the `days` before `end` (default now)."""
        lags = list(lags)
        end_ts = int((end or datetime.now(timezone.utc)).timestamp())
        start_ts = end_ts - days * 86400
        activity = pd.concat([self.activity(wallet, kind, start_ts, end_ts)
                              for kind in ('TRADE', 'SPLIT', 'MERGE')], ignore_index=True)
        if activity.empty:
            raise ValueError(f'{wallet} has no trades in the {days}d before {end_ts}')
        tokens = self.market_tokens(activity['conditionId'])
        events = wallet_events(activity, tokens)
        log.info('copy backtest %s: %d events on %d tokens', wallet, len(events), events['token'].nunique())
        print(f'{len(events)} wallet events on {events["token"].nunique()} tokens; fetching price history')
        prices, fidelity = self.token_prices(events, tokens, end_ts, max(lags, default=0))
        tokens = tokens.assign(fidelity=tokens['token'].map(fidelity))
        return backtest(wallet, events, tokens, prices, lags, slippage, stake, end_ts)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('wallet', help='proxy wallet address (0x...)')
    ap.add_argument('--days', type=int, default=30, help='lookback window')
    ap.add_argument('--lags', type=int, nargs='+', default=list(DEFAULT_LAGS), help='follower delay, seconds')
    ap.add_argument('--slippage', type=float, default=0.01, help='price units added to buys / taken off sells')
    ap.add_argument('--stake', type=float, default=None,
                    help='fixed USDC per copied buy fill (default: mirror the wallet\'s own size)')
    ap.add_argument('--out-dir', type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)

    res = WalletCopyBacktest().run(args.wallet, args.days, args.lags, args.slippage, args.stake)
    stem = f'{args.wallet[:10]}_{datetime.now(timezone.utc):%Y%m%d_%H%M}'
    args.out_dir.mkdir(parents=True, exist_ok=True)
    res.summary.to_csv(args.out_dir / f'{stem}_summary.csv', index=False)
    res.drift.to_csv(args.out_dir / f'{stem}_drift.csv', index=False)
    res.per_token.to_csv(args.out_dir / f'{stem}_tokens.csv', index=False)

    kinds = res.events['kind'].value_counts().to_dict()
    coarse = res.tokens.loc[res.tokens['fidelity'] > 1, 'token']
    with pd.option_context('display.width', 200, 'display.max_columns', None,
                           'display.float_format', '{:,.3f}'.format):
        print(f'\nWallet {args.wallet}, last {args.days}d, slippage {args.slippage}, '
              f'events {kinds}')
        print(res.summary.to_string(index=False))
        print('\nPost-fill drift in the wallet\'s favour (USDC-weighted, price units; '
              'positive = a later follower pays it):')
        print(res.drift_summary.to_string(index=False))
    if len(coarse):
        print(f'\nWarning: {len(coarse)} tokens only had coarse (>=1h) price history; '
              'lagged fills on them are approximate.')
    print(f'\nCSV output: {args.out_dir / stem}_*.csv')


if __name__ == '__main__':
    main()
