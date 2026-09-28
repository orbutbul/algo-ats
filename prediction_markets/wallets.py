"""
prediction_markets/wallets.py — find Polymarket.com (global) wallets that
are performing well recently.

Two public, keyless Data API endpoints do the work (verified live 2026-09-24,
see data/prediction_markets/polymarket_com.md section 6):

  GET data-api.polymarket.com/v1/leaderboard?timePeriod=&orderBy=&category=&limit=&offset=
      Candidate pool. PnL/volume only -- no win rate. `limit` is capped at 50
      server-side; `offset` pages at least to rank 10,000. Cached ~30 min.
      timePeriod DAY|WEEK|MONTH|ALL, orderBy PNL|VOL, category e.g. POLITICS.
  GET data-api.polymarket.com/closed-positions?user=&sortBy=TIMESTAMP&sortDirection=DESC&limit=&offset=
      Per-wallet realised outcomes, losers included (curPrice 0, negative
      realizedPnl). `timestamp` = when the position closed, so paging newest-
      first until it falls behind the lookback cutoff gives "recent" results.
      Also capped at 50 rows/page; rate limit 150 req/10s.
  GET data-api.polymarket.com/positions?user=&redeemable=true&sizeThreshold=0&limit=&offset=
      Resolved positions the wallet still holds. **Losers almost never get
      redeemed** (a worthless token isn't worth the gas/click), so they sit here
      with curPrice 0 forever and never reach /closed-positions. Scoring on
      /closed-positions alone showed one wallet at 138/138 wins while it held
      322 unredeemed losers worth -$3.5M. These rows are merged in as settled,
      pnl = cashPnl + realizedPnl. Unsorted, so every page is read; 500 rows/page.
  GET gamma-api.polymarket.com/markets?condition_ids=..&closed=true
      Resolution time (`closedTime`) for the unredeemed rows, which carry no
      close timestamp. Their `endDate` is the *scheduled* end and is often far
      off (a "by Dec 31" market that resolved in September), so dating by it
      put resolved losers outside the lookback. Up to 100 ids per call
      (verified 2026-09-28); without closed=true it returns nothing. Rows with
      no closedTime fall back to endDate, clipped to now; the placeholder
      1970-01-01 endDate is dropped.

When a wallet hits the closed-positions cap (market makers close thousands a
month), its closed rows only reach back to the oldest one fetched. The
unredeemed rows are cut to that same start, otherwise losers from the
uncovered part of the window would be scored against wins from only the
covered part. `coverage_days` reports how much of the lookback was scored.

Win rate is computed here, per *market* (conditionId), not per position:
market makers and arbs routinely hold both outcomes of one market, which
would otherwise count as one win + one loss. Raw win rate is also
misleading on its own -- buying at 0.95 wins 95% of the time with no edge --
so each wallet also gets `implied_win_rate` (mean entry price on one-sided
markets) and `edge` = win rate minus that. `win_rate_lb` is the Wilson 95%
lower bound, which penalizes small samples (a 14-for-14 week is weak evidence).
Wallets are ranked by `edge_z`: wins on one-sided markets minus the wins their
entry prices implied, over the binomial standard deviation of that count. It
is ~N(0,1) for a wallet with no edge whatever prices it buys at, so a wallet
buying 0.98 favourites no longer outranks one beating the price at 0.40
(sorting by win_rate_lb did exactly that: #1 on 2026-09-24 made 0.06% ROI).

Usage:
    uv run python -m prediction_markets.wallets --periods WEEK MONTH --top 100 --days 30

    from prediction_markets.wallets import PolymarketWalletScanner
    scanner = PolymarketWalletScanner()
    df = scanner.scan(periods=('WEEK',), top_n=50, lookback_days=14)
"""

from __future__ import annotations

import argparse
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from prediction_markets.http import RateLimiter, get_json, log

DATA_BASE_URL = 'https://data-api.polymarket.com'
GAMMA_BASE_URL = 'https://gamma-api.polymarket.com'
CALLS_PER_SECOND = 10         # /closed-positions allows 150/10s; stay under it
PAGE_LIMIT = 50               # server-side cap on both endpoints
MAX_POSITIONS_PER_WALLET = 5000  # 100 pages (~10s); market makers close thousands/month
POSITIONS_PAGE_LIMIT = 500
MAX_UNREDEEMED_PER_WALLET = 10000  # 20 pages
GAMMA_IDS_PER_CALL = 50
OUT_DIR = Path(__file__).parent.parent / 'data' / 'prediction_markets' / 'wallets'
_RATIO_METRICS = ('win_rate', 'win_rate_lb', 'implied_win_rate', 'edge', 'edge_z', 'realized_pnl',
                  'cost_basis', 'roi', 'profit_factor', 'top_market_share', 'hedged_share',
                  'median_entry')


def wilson_lower_bound(wins: int, n: int, z: float = 1.96) -> float:
    """Lower bound of the Wilson score interval for a binomial proportion."""
    if n == 0:
        return float('nan')
    p = wins / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - margin) / denom


def score_closed_positions(positions: pd.DataFrame) -> dict:
    """Aggregate one wallet's closed positions into performance metrics.
    `positions` needs columns conditionId, outcomeIndex, avgPrice,
    totalBought, realizedPnl, timestamp."""
    if positions.empty:
        return {'n_positions': 0, 'n_markets': 0, 'wins': 0,
                **dict.fromkeys(_RATIO_METRICS, float('nan')), 'last_closed': pd.NaT}

    pos = positions.assign(cost=positions['avgPrice'] * positions['totalBought'])
    markets = pos.groupby('conditionId').agg(
        pnl=('realizedPnl', 'sum'),
        cost=('cost', 'sum'),
        n_sides=('outcomeIndex', 'nunique'),
        entry=('avgPrice', 'mean'),
    )
    n = len(markets)
    wins = int((markets['pnl'] > 0).sum())
    gains = markets.loc[markets['pnl'] > 0, 'pnl'].sum()
    losses = -markets.loc[markets['pnl'] < 0, 'pnl'].sum()
    one_sided = markets[markets['n_sides'] == 1]
    win_rate = wins / n
    implied = one_sided['entry'].mean() if len(one_sided) else float('nan')
    one_sided_wr = (one_sided['pnl'] > 0).mean() if len(one_sided) else float('nan')
    expected_var = (one_sided['entry'] * (1 - one_sided['entry'])).sum()
    edge_z = (((one_sided['pnl'] > 0).sum() - one_sided['entry'].sum()) / math.sqrt(expected_var)
              if expected_var > 0 else float('nan'))
    cost = markets['cost'].sum()

    return {
        'n_positions': len(pos),
        'n_markets': n,
        'wins': wins,
        'win_rate': win_rate,
        'win_rate_lb': wilson_lower_bound(wins, n),
        'implied_win_rate': implied,
        'edge': one_sided_wr - implied,
        'edge_z': edge_z,
        'realized_pnl': markets['pnl'].sum(),
        'cost_basis': cost,
        'roi': markets['pnl'].sum() / cost if cost > 0 else float('nan'),
        'profit_factor': gains / losses if losses > 0 else float('inf'),
        'top_market_share': markets['pnl'].max() / gains if gains > 0 else float('nan'),
        'hedged_share': 1 - len(one_sided) / n,
        'median_entry': pos['avgPrice'].median(),
        'last_closed': pd.to_datetime(pos['timestamp'].max(), unit='s', utc=True),
    }


class PolymarketWalletScanner:
    def __init__(self, calls_per_second: float = CALLS_PER_SECOND):
        self._limiter = RateLimiter(calls_per_second)

    def _get(self, path: str, params: dict):
        return get_json(f'{DATA_BASE_URL}{path}', params=params, limiter=self._limiter)

    def leaderboard(
        self,
        period: str = 'WEEK',
        order_by: str = 'PNL',
        top_n: int = 100,
        category: str | None = None,
    ) -> pd.DataFrame:
        """Top `top_n` wallets for one leaderboard. One row per wallet."""
        rows: list[dict] = []
        while len(rows) < top_n:
            params = {'timePeriod': period, 'orderBy': order_by,
                      'limit': min(PAGE_LIMIT, top_n - len(rows)), 'offset': len(rows)}
            if category:
                params['category'] = category
            page = self._get('/v1/leaderboard', params)
            if not page:
                break
            rows.extend(page)
            if len(page) < params['limit']:
                break
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df['rank'] = df['rank'].astype(int)
        return df.rename(columns={'proxyWallet': 'wallet', 'userName': 'user_name'})

    def closed_positions(
        self,
        wallet: str,
        since: datetime | None = None,
        max_positions: int = MAX_POSITIONS_PER_WALLET,
    ) -> pd.DataFrame:
        """Closed positions for `wallet`, newest first, stopping at `since`."""
        cutoff = since.timestamp() if since else None
        rows: list[dict] = []
        while len(rows) < max_positions:
            page = self._get('/closed-positions', {
                'user': wallet, 'sortBy': 'TIMESTAMP', 'sortDirection': 'DESC',
                'limit': PAGE_LIMIT, 'offset': len(rows),
            })
            if not page:
                break
            rows.extend(page)
            if len(page) < PAGE_LIMIT or (cutoff and page[-1]['timestamp'] < cutoff):
                break
        df = pd.DataFrame(rows)
        if cutoff and not df.empty:
            df = df[df['timestamp'] >= cutoff]
        return df.reset_index(drop=True)

    def unredeemed_positions(
        self,
        wallet: str,
        since: datetime | None = None,
        max_positions: int = MAX_UNREDEEMED_PER_WALLET,
    ) -> pd.DataFrame:
        """Resolved-but-still-held positions, reshaped to the closed_positions
        columns (realizedPnl = total settled pnl, timestamp = endDate)."""
        rows: list[dict] = []
        while len(rows) < max_positions:
            page = self._get('/positions', {
                'user': wallet, 'redeemable': 'true', 'sizeThreshold': 0,
                'limit': POSITIONS_PAGE_LIMIT, 'offset': len(rows),
            })
            if not page:
                break
            rows.extend(page)
            if len(page) < POSITIONS_PAGE_LIMIT:
                break
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        end = pd.to_datetime(df['endDate'], errors='coerce', utc=True)
        resolved = df['conditionId'].map(self.resolution_times(df['conditionId']))
        resolved = pd.to_datetime(resolved, errors='coerce', utc=True)
        if resolved.isna().any():
            log.info('%s: %d unredeemed rows have no closedTime; using endDate',
                     wallet, int(resolved.isna().sum()))
        when = resolved.fillna(end.clip(upper=pd.Timestamp.now(tz='UTC')))
        df = df.assign(realizedPnl=df['cashPnl'] + df['realizedPnl'],
                       timestamp=when.astype('int64') // 10**9)
        keep = when > pd.Timestamp('2000-01-01', tz='UTC')
        if since:
            keep &= when >= pd.Timestamp(since)
        return df[keep].reset_index(drop=True)

    def resolution_times(self, condition_ids: Iterable[str]) -> dict[str, str]:
        """conditionId -> Gamma `closedTime` for resolved markets."""
        ids = list(dict.fromkeys(condition_ids))
        out: dict[str, str] = {}
        for i in range(0, len(ids), GAMMA_IDS_PER_CALL):
            chunk = ids[i:i + GAMMA_IDS_PER_CALL]
            params = [('condition_ids', c) for c in chunk] + [('closed', 'true'), ('limit', len(chunk))]
            markets = get_json(f'{GAMMA_BASE_URL}/markets', params=params, limiter=self._limiter)
            out.update({m['conditionId']: m['closedTime'] for m in markets if m.get('closedTime')})
        return out

    def settled_positions(self, wallet: str, since: datetime | None = None,
                          max_positions: int = MAX_POSITIONS_PER_WALLET) -> pd.DataFrame:
        """Closed + resolved-unredeemed positions, with a `source` column.
        If the closed rows hit `max_positions`, the unredeemed rows are cut to
        the oldest closed row so both cover the same span."""
        closed = self.closed_positions(wallet, since, max_positions).assign(source='closed')
        if len(closed) >= max_positions:
            since = pd.Timestamp(closed['timestamp'].min(), unit='s', tz='UTC').to_pydatetime()
        held = self.unredeemed_positions(wallet, since).assign(source='unredeemed')
        return pd.concat([closed, held], ignore_index=True)

    def candidates(
        self,
        periods: Iterable[str] = ('WEEK', 'MONTH'),
        top_n: int = 100,
        category: str | None = None,
    ) -> pd.DataFrame:
        """Union of the PnL leaderboards for `periods`, one row per wallet with
        rank/pnl/vol columns per period (e.g. rank_week, pnl_week)."""
        merged: pd.DataFrame | None = None
        for period in periods:
            lb = self.leaderboard(period, 'PNL', top_n, category)
            if lb.empty:
                continue
            p = period.lower()
            lb = lb[['wallet', 'user_name', 'rank', 'pnl', 'vol']].rename(
                columns={'rank': f'rank_{p}', 'pnl': f'pnl_{p}', 'vol': f'vol_{p}'})
            if merged is None:
                merged = lb
            else:
                merged = merged.merge(lb, on='wallet', how='outer', suffixes=('', '_dup'))
                merged['user_name'] = merged['user_name'].fillna(merged.pop('user_name_dup'))
        return merged if merged is not None else pd.DataFrame(columns=['wallet', 'user_name'])

    def scan(
        self,
        periods: Iterable[str] = ('WEEK', 'MONTH'),
        top_n: int = 100,
        lookback_days: int = 30,
        category: str | None = None,
        max_positions: int = MAX_POSITIONS_PER_WALLET,
    ) -> pd.DataFrame:
        """Score every leaderboard candidate on its closed positions from the
        last `lookback_days`. One row per wallet, sorted by edge_z."""
        cands = self.candidates(periods, top_n, category)
        now = datetime.now(timezone.utc)
        since = now - timedelta(days=lookback_days)
        log.info('wallet scan: %d candidates, lookback %dd', len(cands), lookback_days)
        stats = []
        for i, wallet in enumerate(cands['wallet'], 1):
            try:
                pos = self.settled_positions(wallet, since, max_positions)
            except Exception as e:  # one bad wallet shouldn't sink the scan
                log.warning('position fetch failed for %s: %s', wallet, e)
                pos = pd.DataFrame(columns=['source'])
            closed = pos[pos['source'] == 'closed']
            capped = len(closed) >= max_positions
            start = (pd.Timestamp(closed['timestamp'].min(), unit='s', tz='UTC')
                     if capped else pd.Timestamp(since))
            stats.append({'wallet': wallet, **score_closed_positions(pos),
                          'n_unredeemed': len(pos) - len(closed),
                          'hit_position_cap': capped,
                          'coverage_days': (pd.Timestamp(now) - start).total_seconds() / 86400})
            if i % 25 == 0:
                print(f'  scored {i}/{len(cands)} wallets')
        out = cands.merge(pd.DataFrame(stats), on='wallet', how='left')
        return out.sort_values(['edge_z', 'roi'], ascending=False).reset_index(drop=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--periods', nargs='+', default=['WEEK', 'MONTH'],
                    choices=['DAY', 'WEEK', 'MONTH', 'ALL'])
    ap.add_argument('--top', type=int, default=100, help='wallets per leaderboard')
    ap.add_argument('--days', type=int, default=30, help='closed-position lookback')
    ap.add_argument('--category', default=None, help='e.g. POLITICS, SPORTS, CRYPTO')
    ap.add_argument('--min-markets', type=int, default=20,
                    help='hide wallets with fewer closed markets in the window')
    ap.add_argument('--max-positions', type=int, default=MAX_POSITIONS_PER_WALLET)
    ap.add_argument('--out', type=Path, default=None, help='CSV path (default: data/prediction_markets/wallets/)')
    args = ap.parse_args(argv)

    df = PolymarketWalletScanner().scan(args.periods, args.top, args.days,
                                        args.category, args.max_positions)
    out = args.out or OUT_DIR / f'scan_{datetime.now(timezone.utc):%Y%m%d_%H%M}.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    shown = df[df['n_markets'] >= args.min_markets]
    cols = ['user_name', 'wallet', 'n_markets', 'win_rate', 'win_rate_lb', 'edge', 'edge_z',
            'roi', 'realized_pnl', 'profit_factor', 'hedged_share', 'top_market_share',
            'n_unredeemed', 'coverage_days']
    with pd.option_context('display.width', 200, 'display.max_columns', None,
                           'display.float_format', '{:,.3f}'.format):
        print(shown[cols].head(30).to_string(index=False))
    print(f'\n{len(shown)}/{len(df)} wallets with >= {args.min_markets} closed markets '
          f'in {args.days}d. Full results: {out}')


if __name__ == '__main__':
    main()
