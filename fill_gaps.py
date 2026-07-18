"""
fill_gaps.py — one-off maintenance script, run manually: `python fill_gaps.py`.

Scans the last year of data/ohlcv_1min.parquet, from 365 days ago through
now, for entire missing trading/calendar days per symbol — including days
before a symbol was first tracked, so every currently tracked symbol ends
up with a full year of history — and backfills exactly those gaps, via
Alpaca for equities/ETFs and Binance for crypto.
"""

from datetime import datetime, time, timedelta, timezone

import pandas as pd
import pandas_market_calendars as mcal

from vectorbtpro import vbt

from data_ohlcv import OHLCV_PATH, OHLCV_COLS, BATCH_SIZE, _fetch_alpaca_bars
from utils import _tracked_symbols

LOOKBACK_DAYS = 365


def _present_dates(df: pd.DataFrame) -> dict[str, set]:
    """{ticker: set of dates with data}."""
    present = pd.DataFrame({
        'ticker': df.index.get_level_values('ticker'),
        'date': df.index.get_level_values('datetime').date,
    }).drop_duplicates()
    return present.groupby('ticker')['date'].apply(set).to_dict()


def _missing_days(tickers, valid_days, present_by_ticker) -> dict[str, list]:
    """Missing days for every tracked symbol across the full valid_days range,
    including days before the symbol's first recorded date, so every symbol
    ends up with a full year of history rather than just gap-free coverage
    within whatever range it already has."""
    missing = {}
    for ticker in tickers:
        present = present_by_ticker.get(ticker, set())
        gaps = [d for d in valid_days if d not in present]
        if gaps:
            missing[ticker] = gaps
    return missing


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
    present_by_ticker = _present_dates(df)

    tracked = _tracked_symbols()
    equity_tickers = tracked['equities'] + tracked['etfs']
    crypto_tickers = tracked['cryptos']

    today = datetime.now(timezone.utc).date()
    cutoff_start = today - timedelta(days=LOOKBACK_DAYS)
    cutoff_end = today  # today included — catches a gap even if daily_run.py hasn't run yet today

    nyse = mcal.get_calendar('NYSE')
    equity_valid_days = list(nyse.schedule(start_date=cutoff_start.isoformat(), end_date=cutoff_end.isoformat()).index.date)
    crypto_valid_days = list(pd.date_range(cutoff_start, cutoff_end, freq='D').date)

    equity_missing = _missing_days(equity_tickers, equity_valid_days, present_by_ticker)
    crypto_missing = _missing_days(crypto_tickers, crypto_valid_days, present_by_ticker)

    # Equities are batched by symbol (not by contiguous identical-missing-set
    # day ranges): each batch fetches the full min->max span of its gaps in
    # one call. This keeps the batch count stable at ~len(missing)/BATCH_SIZE
    # even after a partial run leaves per-symbol gaps unevenly distributed
    # (contiguous-range grouping fragments badly in that case — a single
    # symbol filled out of a hundred splits one big range into many tiny
    # single-day ones).
    equity_gapped = sorted(equity_missing)
    equity_batches = [equity_gapped[i:i + BATCH_SIZE] for i in range(0, len(equity_gapped), BATCH_SIZE)]
    crypto_groups = _group_contiguous(crypto_missing, crypto_valid_days)

    print(f'{len(equity_missing)} equity/ETF symbols with gaps, {len(equity_batches)} batch(es)')
    print(f'{len(crypto_missing)} crypto symbols with gaps, {len(crypto_groups)} backfill range(s)')

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
