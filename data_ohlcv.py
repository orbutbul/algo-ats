import yfinance as yf
from yfinance import EquityQuery
import pandas as pd
import requests
import json
from datetime import datetime, timezone
import time
from pathlib import Path
from vectorbtpro import *
from tradingview_screener import Query, col
import pandas_market_calendars as mcal
import re
from tqdm import tqdm
from utils import *

# --- Paths ---
OHLCV_PATH = Path('data/ohlcv_1min.parquet')   # 1-minute OHLCV bars for all asset classes
OHLCV_COLS = ['open', 'high', 'low', 'close', 'volume']
BATCH_SIZE = 100                                # tickers per yfinance download call





def download_daily_1min(yf_symbols, crypto_symbols):
    """
    Downloads today's 1-minute OHLCV bars and appends them to OHLCV_PATH.
    Equities and ETFs are fetched via yfinance; cryptos via Binance (vectorbtpro).
    Deduplicates on (datetime, ticker) so re-running the same day is safe.
    """
    OHLCV_PATH.parent.mkdir(parents=True, exist_ok=True)
    frames = []

    # yfinance: download in batches to avoid rate limits
    for i in range(0, len(yf_symbols), BATCH_SIZE):
        batch = yf_symbols[i:i + BATCH_SIZE]
        raw = yf.download(
            tickers=batch,
            period='1d',
            interval='1m',
            group_by='ticker',
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        if raw.empty:
            continue
        # stack() pivots the ticker column level into the row index
        batch_df = raw.stack(level=0, future_stack=True).rename_axis(['datetime', 'ticker'])
        batch_df.columns = batch_df.columns.str.lower()
        frames.append(batch_df)

    # Binance US crypto data via vectorbtpro
    if crypto_symbols:
        vbt.BinanceData.set_custom_settings(client_config=dict(tld="us"))
        crypto_data = vbt.BinanceData.pull(
            crypto_symbols,
            start='1 day ago',
            timeframe='1m',
        )
        # vbt returns {ticker: df}; concat gives (ticker, datetime) — swap to match equities index order
        crypto_df = pd.concat(crypto_data.data).rename_axis(['ticker', 'datetime'])
        crypto_df = crypto_df.swaplevel().sort_index()
        crypto_df.columns = crypto_df.columns.str.lower()
        crypto_df = crypto_df[['open', 'high', 'low', 'close', 'volume']]
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
    combined.to_csv('mama.csv')
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

