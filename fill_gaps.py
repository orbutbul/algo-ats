"""
fill_gaps.py — one-off maintenance script, run manually: `python fill_gaps.py`.

Scans the last year of data/ohlcv_1min.parquet for gaps per tracked symbol —
both entire missing trading/calendar days and days whose intra-day minute
coverage falls below gap_detection.FILL_THRESHOLD — including days before a
symbol was first tracked, so every currently tracked symbol ends up with a
full year of history. Backfills exactly those (ticker, day)s, refetching the
whole day and letting the existing dedupe-on-index-keep-last merge fill in
whatever was actually missing, via Alpaca for equities/ETFs and Binance for
crypto.
"""

from datetime import datetime, time, timedelta, timezone

import pandas as pd

from vectorbtpro import vbt

from extraction.ohlcv import OHLCV_PATH, OHLCV_COLS, BATCH_SIZE, _fetch_alpaca_bars
from utils import _tracked_symbols
from gap_detection import load_or_build_gap_report, LOOKBACK_DAYS, crypto_valid_days


def _group_contiguous(missing: dict[str, list], valid_days: list) -> list[tuple]:
    """
    Merges valid_days that share an identical missing-ticker set into
    (start_date, end_date, [tickers]) ranges, so a systemic outage affecting
    the same tickers over several days becomes one backfill request per range
    instead of one per day.
    """
    day_to_tickers = {}
    for ticker, days in missing.items():
        for d in days:
            day_to_tickers.setdefault(d, set()).add(ticker)

    groups = []
    current_set, current_start, current_end = None, None, None
    for d in valid_days:
        s = day_to_tickers.get(d)
        if s is not None and s == current_set:
            current_end = d
        else:
            if current_set:
                groups.append((current_start, current_end, sorted(current_set)))
            current_set, current_start, current_end = s, d, d
    if current_set:
        groups.append((current_start, current_end, sorted(current_set)))
    return groups


def _fetch_crypto_range(symbols: list[str], start: datetime, end: datetime) -> pd.DataFrame | None:
    vbt.BinanceData.set_custom_settings(client_config=dict(tld='us'))
    try:
        crypto_data = vbt.BinanceData.pull(
            symbols, start=start, end=end, timeframe='1m',
            execute_kwargs=dict(engine='threadpool', engine_config=dict(init_kwargs=dict(max_workers=min(len(symbols), 30)))),
            show_progress=False,
        )
    except Exception as e:
        print(f'  Binance range fetch failed: {e}')
        return None
    if not crypto_data.data:
        return None
    crypto_df = pd.concat(crypto_data.data).rename_axis(['ticker', 'datetime'])
    crypto_df = crypto_df.swaplevel().sort_index()
    crypto_df.columns = crypto_df.columns.str.lower()
    return crypto_df[OHLCV_COLS]


def fill_gaps() -> None:
    if not OHLCV_PATH.exists():
        print(f'{OHLCV_PATH} does not exist — nothing to gap-fill.')
        return

    df = pd.read_parquet(OHLCV_PATH)
    if list(df.columns) != OHLCV_COLS:
        df = df[OHLCV_COLS]  # drops the stray all-NaN 'adj close' column, if present
        df.to_parquet(OHLCV_PATH)

    today = datetime.now(timezone.utc).date()
    cutoff_start = today - timedelta(days=LOOKBACK_DAYS)

    # Manual maintenance script — always recompute fresh rather than risk
    # acting on a stale cached report (contrast with dashboard.py's default
    # cache reuse).
    gap_report = load_or_build_gap_report(force=True)
    to_refetch = gap_report[gap_report['needs_refetch']].reset_index()

    equity_missing = (
        to_refetch[to_refetch['asset_class'].isin(['equities', 'etfs'])]
        .groupby('ticker')['date'].apply(list).to_dict()
    )
    crypto_missing = (
        to_refetch[to_refetch['asset_class'] == 'cryptos']
        .groupby('ticker')['date'].apply(list).to_dict()
    )
    whole_day_gaps = int(to_refetch['is_whole_day_gap'].sum())
    intraday_gaps = len(to_refetch) - whole_day_gaps

    crypto_days = crypto_valid_days(cutoff_start, today)

    # Equities are batched by symbol (not by contiguous identical-missing-set
    # day ranges): each batch fetches the full min->max span of its gaps in
    # one call. This keeps the batch count stable at ~len(missing)/BATCH_SIZE
    # even after a partial run leaves per-symbol gaps unevenly distributed
    # (contiguous-range grouping fragments badly in that case — a single
    # symbol filled out of a hundred splits one big range into many tiny
    # single-day ones).
    equity_gapped = sorted(equity_missing)
    equity_batches = [equity_gapped[i:i + BATCH_SIZE] for i in range(0, len(equity_gapped), BATCH_SIZE)]
    crypto_groups = _group_contiguous(crypto_missing, crypto_days)

    print(f'{len(equity_missing)} equity/ETF symbols with gaps, {len(equity_batches)} batch(es)')
    print(f'{len(crypto_missing)} crypto symbols with gaps, {len(crypto_groups)} backfill range(s)')
    print(f'{whole_day_gaps} whole-day gap(s), {intraday_gaps} additional (ticker, day)s refetched due to low intra-day completeness')

    total_new_rows = 0

    def _save(frame: pd.DataFrame) -> None:
        nonlocal df, total_new_rows
        combined = pd.concat([df, frame])
        combined = combined[~combined.index.duplicated(keep='last')]
        combined.sort_index(inplace=True)
        combined.to_parquet(OHLCV_PATH)
        total_new_rows += len(frame)
        df = combined
        print(f'    saved ({len(combined):,} rows total)')

    for batch in equity_batches:
        batch_min = min(min(equity_missing[t]) for t in batch)
        batch_max = max(max(equity_missing[t]) for t in batch)
        start_dt = datetime.combine(batch_min, time.min, tzinfo=timezone.utc)
        end_dt = datetime.combine(batch_max + timedelta(days=1), time.min, tzinfo=timezone.utc)
        print(f'  Backfilling {len(batch)} equity/ETF symbol(s) {batch_min} -> {batch_max}...')
        raw = _fetch_alpaca_bars(batch, start_dt, end_dt)
        if raw is not None and not raw.empty:
            batch_df = raw.rename_axis(['ticker', 'datetime']).swaplevel().sort_index()
            _save(batch_df[OHLCV_COLS])

    for start_day, end_day, symbols in crypto_groups:
        start_dt = datetime.combine(start_day, time.min, tzinfo=timezone.utc)
        end_dt = datetime.combine(end_day + timedelta(days=1), time.min, tzinfo=timezone.utc)
        print(f'  Backfilling {len(symbols)} crypto symbol(s) {start_day} -> {end_day}...')
        frame = _fetch_crypto_range(symbols, start_dt, end_dt)
        if frame is not None:
            _save(frame)

    if total_new_rows == 0:
        print('No gaps needed backfilling.')
        return

    print(f'Done. {total_new_rows:,} new rows fetched, {len(df):,} rows in {OHLCV_PATH}')


if __name__ == '__main__':
    fill_gaps()
