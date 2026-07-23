"""
gap_detection.py — shared gap-detection module for data/ohlcv_1min.parquet.

Computes, per (ticker, day), how many 1-minute bars are present versus how
many are expected (NYSE session length for equities/ETFs, 1440/day for
24/7 crypto), over the last LOOKBACK_DAYS. Detection only — no fetching, no
writes to the OHLCV file itself. fill_gaps.py consumes this to decide what
to refetch; dashboard.py consumes it to visualize data health.
"""

from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pandas_market_calendars as mcal

from utils import OHLCV_PATH, _tracked_symbols

GAP_REPORT_PATH = Path(__file__).parent / 'data' / 'gap_report.parquet'
LOOKBACK_DAYS = 365

# Below this completeness fraction, a (ticker, day) is flagged for refetch.
# Deliberately conservative — well below typical illiquid-name sparsity — so
# normal thin trading doesn't trigger constant re-pulls and burn API quota.
# Tune after eyeballing the dashboard's completeness distribution.
FILL_THRESHOLD = 0.90


def equity_valid_days(cutoff_start: date, cutoff_end: date) -> pd.DataFrame:
    """NYSE schedule over the window, indexed by date, with market_open/market_close
    columns — lets callers compute exact expected-minute counts, half-days included."""
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date=cutoff_start.isoformat(), end_date=cutoff_end.isoformat())
    schedule.index = schedule.index.date
    return schedule


def crypto_valid_days(cutoff_start: date, cutoff_end: date) -> list:
    """24/7 — every calendar day in the window."""
    return list(pd.date_range(cutoff_start, cutoff_end, freq='D').date)


def actual_minute_counts(df: pd.DataFrame) -> pd.Series:
    """{(ticker, date): count of bars present that day}, indexed as a MultiIndex Series."""
    counts = df.groupby([
        df.index.get_level_values('ticker'),
        df.index.get_level_values('datetime').date,
    ]).size()
    counts.index.names = ['ticker', 'date']
    return counts


def build_gap_report(df: pd.DataFrame | None = None, tracked: dict | None = None,
                      lookback_days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """
    Full whole-day + intra-day gap picture in one pass.

    Returns a DataFrame indexed by MultiIndex (ticker, date) with columns:
      asset_class, expected_minutes, actual_minutes, completeness (0-1),
      is_whole_day_gap (bool), needs_refetch (bool, completeness < FILL_THRESHOLD).

    Every tracked ticker is scored across the full lookback window, including
    days before it was first recorded, so a newly-tracked ticker's pre-listing
    days show up as legitimate low completeness rather than being silently
    excluded — dashboards can distinguish "new listing" from "actually broken"
    via first/last-present-date.
    """
    if df is None:
        df = pd.read_parquet(OHLCV_PATH)
    if tracked is None:
        tracked = _tracked_symbols()

    today = datetime.now(timezone.utc).date()
    cutoff_start = today - pd.Timedelta(days=lookback_days)
    cutoff_start = cutoff_start if isinstance(cutoff_start, date) else cutoff_start.date()
    cutoff_end = today

    eq_schedule = equity_valid_days(cutoff_start, cutoff_end)
    eq_expected = ((pd.to_datetime(eq_schedule['market_close']) - pd.to_datetime(eq_schedule['market_open']))
                   .dt.total_seconds() // 60).astype(int)
    eq_expected = eq_expected.to_dict()  # {date: expected_minutes}

    cr_days = crypto_valid_days(cutoff_start, cutoff_end)
    cr_expected = {d: 1440 for d in cr_days}

    actual = actual_minute_counts(df)

    equity_tickers = tracked['equities'] + tracked['etfs']
    crypto_tickers = tracked['cryptos']

    rows = []
    for ticker in equity_tickers:
        for d, expected in eq_expected.items():
            act = actual.get((ticker, d), 0)
            rows.append((ticker, d, 'equities', expected, act))
    for ticker in crypto_tickers:
        for d, expected in cr_expected.items():
            act = actual.get((ticker, d), 0)
            rows.append((ticker, d, 'cryptos', expected, act))

    report = pd.DataFrame(rows, columns=['ticker', 'date', 'asset_class', 'expected_minutes', 'actual_minutes'])
    report['completeness'] = (report['actual_minutes'] / report['expected_minutes']).clip(upper=1.0)
    report['is_whole_day_gap'] = report['actual_minutes'] == 0
    report['needs_refetch'] = report['completeness'] < FILL_THRESHOLD
    report = report.set_index(['ticker', 'date']).sort_index()
    return report


def load_or_build_gap_report(max_age_hours: float = 6.0, force: bool = False) -> pd.DataFrame:
    """Cached wrapper around build_gap_report(), backed by GAP_REPORT_PATH."""
    if not force and GAP_REPORT_PATH.exists():
        mtime = datetime.fromtimestamp(GAP_REPORT_PATH.stat().st_mtime, tz=timezone.utc)
        age_hours = (datetime.now(timezone.utc) - mtime).total_seconds() / 3600
        if age_hours < max_age_hours:
            return pd.read_parquet(GAP_REPORT_PATH)

    report = build_gap_report()
    GAP_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_parquet(GAP_REPORT_PATH)
    return report


def summarize(gap_report: pd.DataFrame) -> dict:
    """Per-asset-class aggregate rollup for a dashboard summary panel."""
    summary = {}
    for asset_class, group in gap_report.groupby('asset_class'):
        by_ticker = group.groupby(level='ticker')
        summary[asset_class] = {
            'tickers': by_ticker.ngroups,
            'tickers_with_whole_day_gap': int((by_ticker['is_whole_day_gap'].sum() > 0).sum()),
            'tickers_below_threshold': int((by_ticker['needs_refetch'].sum() > 0).sum()),
            'mean_completeness': float(group['completeness'].mean()),
            'median_completeness': float(group['completeness'].median()),
        }
    generated_at = None
    if GAP_REPORT_PATH.exists():
        generated_at = datetime.fromtimestamp(GAP_REPORT_PATH.stat().st_mtime, tz=timezone.utc)
    summary['generated_at'] = generated_at
    return summary
