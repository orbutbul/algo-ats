import os
import yfinance as yf
from yfinance import EquityQuery
import pandas as pd
import requests
import json
from datetime import datetime, timedelta, timezone
import time
from pathlib import Path
from dotenv import load_dotenv
from vectorbtpro import *
from tradingview_screener import Query, col
import pandas_market_calendars as mcal
import re
from tqdm import tqdm
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed, Adjustment
from utils import *

load_dotenv()

# --- Paths ---
OHLCV_PATH = Path('data/ohlcv_1min.parquet')       # 1-minute OHLCV bars for all asset classes
OHLCV_COLS = ['open', 'high', 'low', 'close', 'volume']
BATCH_SIZE = 100                                    # tickers per data-provider call
LAST_RUN_PATH = Path('data/ohlcv_last_run.txt')     # timestamp of the last successful 1-min OHLCV pull

_alpaca_client = StockHistoricalDataClient(os.environ['ALPACA_API_KEY'], os.environ['ALPACA_SECRET_KEY'])


def _read_last_run() -> datetime | None:
    if not LAST_RUN_PATH.exists():
        return None
    return datetime.fromisoformat(LAST_RUN_PATH.read_text().strip())


def _write_last_run(ts: datetime) -> None:
    LAST_RUN_PATH.parent.mkdir(parents=True, exist_ok=True)
    LAST_RUN_PATH.write_text(ts.isoformat())



_INVALID_SYMBOL_RE = re.compile(r'invalid symbol: ([^"]+)')
_SHARE_CLASS_RE = re.compile(r'^([A-Z0-9]+)-([A-Z]{1,2})$')


def _to_alpaca_symbol(symbol: str) -> str:
    """
    Alpaca uses dot notation for share classes (e.g. 'BF.A'); our internal
    tickers use hyphens (e.g. 'BF-A') for yfinance/fundamentals compatibility.
    """
    m = _SHARE_CLASS_RE.match(symbol)
    return f'{m.group(1)}.{m.group(2)}' if m else symbol


def _fetch_alpaca_bars(symbols: list[str], start: datetime, end: datetime):
    """
    Fetches 1-min bars for a batch of symbols. One malformed ticker (e.g. a
    share-class symbol Alpaca doesn't recognize) fails the whole batch request,
    so on that error this drops the offending symbol and retries rather than
    losing every other symbol in the batch.
    """
    symbols = list(symbols)
    while symbols:
        alpaca_to_internal = {_to_alpaca_symbol(s): s for s in symbols}
        req = StockBarsRequest(
            symbol_or_symbols=list(alpaca_to_internal),
            timeframe=TimeFrame.Minute,
            start=start,
            end=end,
            feed=DataFeed.IEX,
            adjustment=Adjustment.SPLIT,
        )
        try:
            raw = _alpaca_client.get_stock_bars(req).df
            if raw is not None and not raw.empty:
                raw = raw.rename(index=alpaca_to_internal, level='symbol')
            return raw
        except Exception as e:
            bad = _INVALID_SYMBOL_RE.search(str(e))
            bad_symbol = alpaca_to_internal.get(bad.group(1), bad.group(1)) if bad else None
            if bad_symbol in symbols:
                symbols.remove(bad_symbol)
                continue
            print(f'  Alpaca batch failed: {e}')
            return None
    return None


def download_daily_1min(equity_symbols, crypto_symbols):
    """
    Downloads 1-minute OHLCV bars and appends them to OHLCV_PATH.
    Equities and ETFs are fetched via Alpaca; cryptos via Binance (vectorbtpro).
    Deduplicates on (datetime, ticker) so re-running the same day is safe.

    If the last successful pull was more than 24h ago (e.g. the machine was off),
    backfills from that timestamp to now instead of just the latest session, so a
    missed day doesn't leave a gap. Otherwise pulls the last 24h as usual.
    """
    OHLCV_PATH.parent.mkdir(parents=True, exist_ok=True)
    frames = []

    now = datetime.now(timezone.utc)
    last_run = _read_last_run()
    if last_run is not None and (now - last_run) > timedelta(hours=24):
        start = last_run
        print(f'  Last run was {now - start} ago — backfilling from {start} to now.')
    else:
        start = now - timedelta(days=1)

    # Alpaca: download in batches to keep request URLs a reasonable size
    for i in range(0, len(equity_symbols), BATCH_SIZE):
        batch = equity_symbols[i:i + BATCH_SIZE]
        raw = _fetch_alpaca_bars(batch, start, now)
        if raw is None or raw.empty:
            continue
        batch_df = raw.rename_axis(['ticker', 'datetime']).swaplevel().sort_index()
        frames.append(batch_df[OHLCV_COLS])

    # Binance US crypto data via vectorbtpro — fetched concurrently since
    # Binance pages each symbol sequentially otherwise (~5 min/symbol for a
    # full year of 1-min bars), which dominates runtime when backfilling.
    if crypto_symbols:
        vbt.BinanceData.set_custom_settings(client_config=dict(tld="us"))
        crypto_data = vbt.BinanceData.pull(
            crypto_symbols,
            start=start,
            timeframe='1m',
            execute_kwargs=dict(engine='threadpool', engine_config=dict(init_kwargs=dict(max_workers=min(len(crypto_symbols), 30)))),
            show_progress=False,
        )
        # vbt returns {ticker: df}; concat gives (ticker, datetime) — swap to match equities index order
        crypto_df = pd.concat(crypto_data.data).rename_axis(['ticker', 'datetime'])
        crypto_df = crypto_df.swaplevel().sort_index()
        crypto_df.columns = crypto_df.columns.str.lower()
        crypto_df = crypto_df[OHLCV_COLS]
        frames.append(crypto_df)

    if not frames:
        print('No data fetched.')
        return

    new_data = pd.concat(frames)
    if OHLCV_PATH.exists():
        existing = pd.read_parquet(OHLCV_PATH)
        combined = pd.concat([existing, new_data])
        combined = combined[~combined.index.duplicated(keep='last')]  # keep freshest on overlap
    else:
        combined = new_data

    combined.sort_index(inplace=True)
    combined.to_parquet(OHLCV_PATH)
    _write_last_run(now)
    print(f'Saved {len(combined):,} rows to {OHLCV_PATH}')


DAILY_PATH = Path('data/ohlcv_max.parquet')


def get_max_data(symbols: list[str] | None = None) -> None:

    """
    Download the maximum available daily OHLCV history for equities and ETFs
    and save to data/ohlcv_daily.parquet.

    If the file already exists, only fetches data after the last stored date
    so re-running is fast and safe.

    symbols: explicit list of tickers, or None to use screen_symbols().
    """
    if symbols is None:
        screen = screen_symbols()
        symbols = screen['equities'] + screen['etfs']

    DAILY_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Per-ticker last date so new tickers always get full history
    if DAILY_PATH.exists():
        existing = pd.read_parquet(DAILY_PATH)
        last_per_ticker = (
            existing.groupby(level='ticker')
            .apply(lambda x: x.index.get_level_values('datetime').max().date().isoformat())
            .to_dict()
        )
    else:
        existing = None
        last_per_ticker = {}

    new_tickers  = [s for s in symbols if s not in last_per_ticker]
    old_tickers  = [s for s in symbols if s in last_per_ticker]
    print(f'{len(new_tickers):,} new tickers (full history) + {len(old_tickers):,} existing (incremental)')

    def _download_batches(tickers, extra_kwargs):
        frames = []
        for i in tqdm(range(0, len(tickers), BATCH_SIZE), desc='Downloading'):
            batch = tickers[i:i + BATCH_SIZE]
            try:
                raw = yf.download(
                    tickers=batch, interval='1d',
                    auto_adjust=True, progress=False, threads=True,
                    **extra_kwargs,
                )
            except Exception as e:
                print(f'  Batch {i//BATCH_SIZE + 1} failed: {e}')
                continue
            if raw.empty:
                continue
            df = raw.stack(level='Ticker', future_stack=True).rename_axis(['datetime', 'ticker'])
            df.columns = df.columns.str.lower()
            df = df[[c for c in OHLCV_COLS if c in df.columns]]
            frames.append(df)
        return frames

    frames = []
    if new_tickers:
        frames += _download_batches(new_tickers, {'period': 'max'})
    if old_tickers:
        # All existing tickers share the same incremental start (earliest last date)
        start = min(last_per_ticker[t] for t in old_tickers)
        frames += _download_batches(old_tickers, {'start': start})

    if not frames:
        print('No data fetched.')
        return

    new_data = pd.concat(frames)
    if existing is not None:
        combined = pd.concat([existing, new_data])
        combined = combined[~combined.index.duplicated(keep='last')]
    else:
        combined = new_data

    combined.sort_index(inplace=True)
    combined.to_parquet(DAILY_PATH)
    print(f'Saved {len(combined):,} rows ({combined.index.get_level_values("ticker").nunique():,} tickers) to {DAILY_PATH}')


def upsert_new_symbol(symbol: str, max: bool = False, daily: bool = False) -> None:
    """
    Validate a symbol on yfinance, then optionally fetch and append its data.

    symbol : ticker to add (e.g. 'PLTR', 'ETHUSDT')
    max    : fetch full daily history and append to data/ohlcv_daily.parquet
    daily  : fetch today's 1-min bars and append to data/ohlcv_1min.parquet

    Crypto (quoteType='CRYPTOCURRENCY') is supported for daily (via Binance)
    but not for max (yfinance daily history only covers equities/ETFs).
    """
    info = yf.Ticker(symbol).info
    quote_type = info.get('quoteType')
    if not quote_type:
        print(f"'{symbol}' not found on yfinance — aborting.")
        return

    is_crypto = quote_type == 'CRYPTOCURRENCY'
    print(f"Found {symbol}: {quote_type} — {info.get('longName', info.get('shortName', 'N/A'))}")

    if max:
        if is_crypto:
            print(f"  Skipping max history for {symbol}: crypto daily history not supported (use Binance pull manually).")
        else:
            get_max_data([symbol])

    if daily:
        if is_crypto:
            download_daily_1min([], [symbol])
        else:
            download_daily_1min([symbol], [])

