"""
extraction/ohlcv_massive.py — one-off historical backfill of 1-min equity/ETF
OHLCV bars from Massive (formerly Polygon.io), replacing the sparse
Alpaca/IEX-sourced history. Rate-limited to the free tier's 5 calls/min,
resumable via a per-ticker progress ledger, checkpointed to a staging
parquet. Run manually: `python -m extraction.ohlcv_massive`.

Does NOT touch crypto (already ~100% complete via Binance) or the daily
incremental Alpaca pull in extraction/ohlcv.py — this only rebuilds
equity/ETF *history*. Promote the result into data/ohlcv_1min.parquet with
`python -m extraction.ohlcv_massive --promote` once the backfill finishes.

"""

import argparse
import json
import os
import shutil
import time
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from extraction.ohlcv import _to_alpaca_symbol, OHLCV_COLS
from utils import OHLCV_PATH, _tracked_symbols

load_dotenv()

MASSIVE_API_BASE = 'https://api.massive.com'
MASSIVE_RATE_LIMIT_CALLS_PER_MIN = 5  # free tier
MASSIVE_MAX_BARS_PER_CALL = 50_000
STAGING_PATH = Path(__file__).parent.parent / 'data' / 'ohlcv_1min_massive.parquet'
PROGRESS_PATH = Path(__file__).parent.parent / 'data' / 'ohlcv_massive_progress.json'
DEFAULT_START = date(2024, 7, 23)
DEFAULT_END = date(2026, 7, 23)
CHECKPOINT_EVERY = 25  # tickers per flush to the staging parquet


class _RateLimiter:
    """Steady-interval throttle (not a bursty window) — keeps spacing between
    calls at >= 60/calls_per_minute seconds."""

    def __init__(self, calls_per_minute: int):
        self.interval = 60.0 / calls_per_minute
        self._last_call = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self._last_call = time.monotonic()


class _NotFound(Exception):
    pass


def _get(url: str, params: dict | None, limiter: _RateLimiter, retries: int = 5) -> dict:
    for attempt in range(retries):
        limiter.wait()
        resp = requests.get(url, params=params)
        if resp.status_code == 404:
            raise _NotFound()
        if resp.status_code == 429 or resp.status_code >= 500:
            wait = 15 * (attempt + 1)
            print(f'  {resp.status_code} on {url} — retrying in {wait}s')
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError(f'exceeded retries fetching {url}')


def _fetch_ticker_bars(ticker: str, start: date, end: date, api_key: str, limiter: _RateLimiter) -> pd.DataFrame:
    """
    Paginates GET /v2/aggs/ticker/{ticker}/range/1/minute/{start}/{end} via
    next_url (re-appending apiKey each time, since Massive's next_url omits
    it) until exhausted. Returns a DataFrame indexed (datetime, ticker) with
    OHLCV_COLS, empty if no data. Raises _NotFound if the ticker isn't
    covered by Massive at all.
    """
    massive_symbol = _to_alpaca_symbol(ticker)  # same dot-notation share-class convention as Alpaca
    url = f'{MASSIVE_API_BASE}/v2/aggs/ticker/{massive_symbol}/range/1/minute/{start.isoformat()}/{end.isoformat()}'
    params = {'adjusted': 'true', 'sort': 'asc', 'limit': MASSIVE_MAX_BARS_PER_CALL, 'apiKey': api_key}

    results = []
    while url:
        data = _get(url, params, limiter)
        results.extend(data.get('results') or [])
        url = data.get('next_url')
        params = {'apiKey': api_key} if url else None

    if not results:
        empty = pd.DataFrame(columns=OHLCV_COLS)
        empty.index = pd.MultiIndex.from_arrays([[], []], names=['datetime', 'ticker'])
        return empty

    df = pd.DataFrame(results)
    df['datetime'] = pd.to_datetime(df['t'], unit='ms', utc=True)
    df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
    df['ticker'] = ticker  # store under the internal (hyphenated) ticker, not Massive's dot notation
    return df.set_index(['datetime', 'ticker'])[OHLCV_COLS].sort_index()


def _load_progress() -> dict:
    if PROGRESS_PATH.exists():
        return json.loads(PROGRESS_PATH.read_text())
    return {}


def _save_progress(progress: dict) -> None:
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_PATH.write_text(json.dumps(progress, indent=2, default=str))


def build_massive_ohlcv(tickers: list[str] | None = None, start: date = DEFAULT_START, end: date = DEFAULT_END,
                         force: bool = False, checkpoint_every: int = CHECKPOINT_EVERY) -> None:
    """
    Backfills 1-min bars for `tickers` (default: tracked equities + ETFs)
    from Massive, respecting the free tier's 5 calls/min. Resumable: skips
    tickers already marked 'done' in PROGRESS_PATH unless force=True.
    """
    api_key = os.environ['MASSIVE_API_KEY']

    if tickers is None:
        tracked = _tracked_symbols()
        tickers = sorted(set(tracked['equities']) | set(tracked['etfs']))

    progress = _load_progress()
    remaining = [t for t in tickers if force or progress.get(t, {}).get('status') != 'done']
    print(f'{len(tickers) - len(remaining)}/{len(tickers)} already done, {len(remaining)} remaining')

    limiter = _RateLimiter(MASSIVE_RATE_LIMIT_CALLS_PER_MIN)
    pending_frames: list[pd.DataFrame] = []
    pending_tickers: list[str] = []

    def _flush() -> None:
        nonlocal pending_frames, pending_tickers
        if not pending_tickers:
            return
        non_empty = [f for f in pending_frames if len(f)]
        if non_empty:
            new_data = pd.concat(non_empty)
            if STAGING_PATH.exists():
                existing = pd.read_parquet(STAGING_PATH)
                combined = pd.concat([existing, new_data])
                combined = combined[~combined.index.duplicated(keep='last')]
            else:
                combined = new_data
            combined.sort_index(inplace=True)
            STAGING_PATH.parent.mkdir(parents=True, exist_ok=True)
            combined.to_parquet(STAGING_PATH)
            n_rows = len(combined)
        else:
            n_rows = len(pd.read_parquet(STAGING_PATH)) if STAGING_PATH.exists() else 0
        # Persist progress only after any new data is safely on disk, so a
        # crash between flush and progress-save can't mark a ticker 'done'
        # for data that isn't actually there.
        _save_progress(progress)
        print(f'  checkpoint: {len(pending_tickers)} tickers flushed, {n_rows:,} rows in staging file')
        pending_frames = []
        pending_tickers = []

    for i, ticker in enumerate(remaining, 1):
        try:
            df = _fetch_ticker_bars(ticker, start, end, api_key, limiter)
            pending_frames.append(df)
            dates = df.index.get_level_values('datetime')
            progress[ticker] = {
                'status': 'done',
                'n_bars': len(df),
                'first': dates.min().isoformat() if len(df) else None,
                'last': dates.max().isoformat() if len(df) else None,
                'error': None,
                'fetched_at': datetime.now(timezone.utc).isoformat(),
            }
            print(f'[{i}/{len(remaining)}] {ticker}: {len(df)} bars')
        except _NotFound:
            progress[ticker] = {
                'status': 'not_found', 'n_bars': 0, 'first': None, 'last': None,
                'error': None, 'fetched_at': datetime.now(timezone.utc).isoformat(),
            }
            print(f'[{i}/{len(remaining)}] {ticker}: not found on Massive')
        except Exception as e:
            progress[ticker] = {
                'status': 'error', 'n_bars': 0, 'first': None, 'last': None,
                'error': str(e), 'fetched_at': datetime.now(timezone.utc).isoformat(),
            }
            print(f'[{i}/{len(remaining)}] {ticker}: FAILED — {e}')
        pending_tickers.append(ticker)

        if len(pending_tickers) >= checkpoint_every or i == len(remaining):
            _flush()

    n_done = sum(1 for v in progress.values() if v['status'] == 'done')
    n_not_found = sum(1 for v in progress.values() if v['status'] == 'not_found')
    n_error = sum(1 for v in progress.values() if v['status'] == 'error')
    print(f'Done. {n_done} succeeded, {n_not_found} not found, {n_error} failed. '
          f'See {PROGRESS_PATH} for per-ticker detail.')


def promote_to_main() -> None:
    """
    Backs up the current OHLCV_PATH, then replaces its equities/ETF rows with
    STAGING_PATH's contents while preserving its crypto rows (still
    Binance-sourced, untouched).
    """
    if not STAGING_PATH.exists():
        raise FileNotFoundError(f'{STAGING_PATH} not found — run build_massive_ohlcv() first')
    massive_df = pd.read_parquet(STAGING_PATH)

    if OHLCV_PATH.exists():
        backup_path = OHLCV_PATH.with_name(
            f'{OHLCV_PATH.stem}_backup_{datetime.now():%Y%m%d_%H%M%S}{OHLCV_PATH.suffix}'
        )
        shutil.copy2(OHLCV_PATH, backup_path)
        print(f'Backed up existing OHLCV to {backup_path}')

        existing = pd.read_parquet(OHLCV_PATH)
        crypto_tickers = set(_tracked_symbols()['cryptos'])
        crypto_mask = existing.index.get_level_values('ticker').isin(crypto_tickers)
        crypto_only = existing[crypto_mask]
    else:
        crypto_only = pd.DataFrame(columns=OHLCV_COLS)

    combined = pd.concat([crypto_only, massive_df]).sort_index()
    combined.to_parquet(OHLCV_PATH)
    print(f'Promoted {len(massive_df):,} equity/ETF rows + {len(crypto_only):,} crypto rows -> {OHLCV_PATH}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Backfill 1-min equity/ETF OHLCV from Massive (rate-limited, resumable).')
    p.add_argument('--start', type=date.fromisoformat, default=DEFAULT_START)
    p.add_argument('--end', type=date.fromisoformat, default=DEFAULT_END)
    p.add_argument('--force', action='store_true', help='Re-fetch tickers already marked done.')
    p.add_argument('--promote', action='store_true',
                    help='Skip fetching; just promote the existing staging file into ohlcv_1min.parquet.')
    args = p.parse_args()

    if args.promote:
        promote_to_main()
    else:
        build_massive_ohlcv(start=args.start, end=args.end, force=args.force)
