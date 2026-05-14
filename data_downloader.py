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

# --- Paths ---
OHLCV_PATH = Path('data/ohlcv_1min.parquet')   # 1-minute OHLCV bars for all asset classes
FUNDAMENTALS_DIR = Path('data/fundamentals')    # one parquet per {asset_class}_{freq}

BATCH_SIZE = 100                                # tickers per yfinance download call
OHLCV_COLS = ['open', 'high', 'low', 'close', 'volume']

# --- Fundamental field lists ---
# Fields are grouped by how often they meaningfully change.
# All sourced from yfinance Ticker.info (one HTTP call per ticker).

EQUITY_DAILY_FIELDS = [
    "marketCap",            # moves with price every session
    "sharesOutstanding",    # changes on buybacks / new issuances
    "floatShares",          # tradeable shares (excludes insiders / locked-up)
    "shortPercentOfFloat",  # updated bi-monthly by FINRA; yfinance shows latest available
]

EQUITY_WEEKLY_FIELDS = [
    "beta",                 # 5-year rolling beta vs S&P 500; recalculated weekly
]

EQUITY_MONTHLY_FIELDS = [
    "trailingPE",           # price / trailing-12-month EPS
    "forwardPE",            # price / next-12-month consensus EPS
    "enterpriseToEbitda",   # EV / EBITDA valuation multiple
    "revenueGrowth",        # YoY revenue growth (most recent quarter)
    "earningsGrowth",       # YoY EPS growth (most recent quarter)
    "returnOnEquity",       # net income / shareholders' equity
    "grossMargins",         # (revenue - COGS) / revenue
    "debtToEquity",         # total debt / total equity
    "freeCashflow",         # operating CF - capex (absolute $)
]

ETF_DAILY_FIELDS = [
    "totalAssets",                  # AUM in dollars
    "trailingAnnualDividendYield",  # TTM distributions / NAV
]

ETF_WEEKLY_FIELDS = [
    "beta3Year",    # 3-year beta vs benchmark (more stable than 1-year)
    "ytdReturn",    # year-to-date total return
]

ETF_MONTHLY_FIELDS = [
    "threeYearAverageReturn",   # annualized 3-year total return
    "fiveYearAverageReturn",    # annualized 5-year total return
    "annualReportExpenseRatio", # fund expense ratio (almost never changes)
]

# yfinance crypto coverage is thin — no on-chain data, just market stats
CRYPTO_DAILY_FIELDS = [
    "marketCap",            # price * circulating supply
    "circulatingSupply",    
    "volume24Hr",   
]

CRYPTO_MONTHLY_FIELDS = [
    "maxSupply",    
    "totalSupply",  
]

# Master config: maps each asset class to its freq → fields schedule.
# Pass a subset of this dict to fundamentals() to run only what you need.
FUNDAMENTALS_SPECS = {
    'equities': {
        'daily':   EQUITY_DAILY_FIELDS,
        'weekly':  EQUITY_WEEKLY_FIELDS,
        'monthly': EQUITY_MONTHLY_FIELDS,
    },
    'etfs': {
        'daily':   ETF_DAILY_FIELDS,
        'weekly':  ETF_WEEKLY_FIELDS,
        'monthly': ETF_MONTHLY_FIELDS,
    },
    # 'cryptos': {
    #     'daily':   CRYPTO_DAILY_FIELDS,
    #     'monthly': CRYPTO_MONTHLY_FIELDS,
    # },
}

# Minimum age (days) before a cached parquet is considered stale
FREQ_STALENESS_DAYS = {'daily': 1, 'weekly': 7, 'monthly': 30}

# data downloader
# job app
# coursera


def screen_symbols():
    """
    Returns {'cryptos': [...], 'equities': [...], 'etfs': [...]}.
    Pulls the top symbols for each asset class from the TradingView screener.
    """
    # Top 50 BinanceUS cryptos by 24h volume
    resp = requests.post(
        'https://scanner.tradingview.com/crypto/scan',
        headers={'User-Agent': 'Mozilla/5.0'},
        json={
            "filter": [
                {"left": "exchange", "operation": "equal", "right": "BINANCEUS"}
            ],
            "columns": ["name", "volume"],
            "sort": {"sortBy": "volume", "sortOrder": "desc"},
            "range": [0, 50]
        }
    )
    cryptos = [row['d'][0] for row in resp.json()['data']]

    # Top 2000 US equities by market cap, requiring min 100k avg daily volume
    _, equities_df = (Query()
        .set_markets('america')
        .select('name', 'average_volume_60d_calc', 'market_cap_basic')
        .where(
            col('average_volume_60d_calc') >= 100_000,
            col('type') == 'stock',
            col('is_primary') == True,
        )
        .order_by('market_cap_basic', ascending=False)
        .limit(2000)
        .get_scanner_data()
    )
    equities = equities_df['name'].tolist()
    # TradingView uses '.' for share classes (e.g. BRK.B); yfinance expects '-'
    equities = [t.replace('.', '-').strip().upper() for t in equities]

    # Top 200 ETFs by AUM, filtered to $500M+ AUM and 100k+ avg volume
    etf_resp = requests.post(
        'https://scanner.tradingview.com/america/scan',
        headers={'User-Agent': 'Mozilla/5.0'},
        json={
            "filter": [
                {"left": "type", "operation": "equal", "right": "fund"},
                {"left": "typespecs", "operation": "has", "right": ["etf"]},
                {"left": "average_volume_60d_calc", "operation": "greater", "right": 100_000},
                {"left": "aum", "operation": "greater", "right": 500_000_000},
            ],
            "columns": ["name", "aum", "average_volume_60d_calc"],
            "sort": {"sortBy": "aum", "sortOrder": "desc"},
            "range": [0, 200]
        }
    )
    etfs = [row['d'][0] for row in etf_resp.json()['data']]

    return {'cryptos': cryptos, 'equities': equities, 'etfs': etfs}


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
        batch_df = raw.stack(level=0).rename_axis(['datetime', 'ticker'])
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


def get_data(tickers=None, start=None, end=None):
    """
    Reads the OHLCV parquet and returns {ticker: DataFrame} with Title-cased columns.
    All filters are optional — omit to load everything.
    """
    df = pd.read_parquet(OHLCV_PATH)
    if tickers is not None:
        tickers = [tickers] if isinstance(tickers, str) else tickers
        df = df[df.index.get_level_values('ticker').isin(tickers)]
    if start is not None:
        df = df[df.index.get_level_values('datetime') >= pd.Timestamp(start)]
    if end is not None:
        df = df[df.index.get_level_values('datetime') <= pd.Timestamp(end)]

    return {
        ticker: group.droplevel('ticker').rename(columns=str.title)
        for ticker, group in df.groupby(level='ticker')
    }


def _needs_update(path: Path, freq: str) -> bool:
    """True if the parquet is missing or older than the staleness threshold for freq."""
    if not path.exists():
        return True
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).date()
    today = datetime.now(timezone.utc).date()
    return (today - mtime).days >= FREQ_STALENESS_DAYS.get(freq, 1)


def _fetch_fields(tickers: list[str], fields: list[str], delay: float = 0.5, retries: int = 3) -> pd.DataFrame:
    """Calls Ticker.info for each ticker and extracts the requested fields into a DataFrame.
    Sleeps `delay` seconds between calls; retries up to `retries` times on rate-limit errors.
    """
    rows = []
    for ticker in tqdm(tickers, desc='fetching', unit='ticker'):
        for attempt in range(retries):
            try:
                info = yf.Ticker(ticker).info
                row = {'ticker': ticker}
                row.update({f: info.get(f) for f in fields})
                rows.append(row)
                time.sleep(delay)
                break
            except Exception as e:
                if attempt < retries - 1:
                    wait = delay * 2 ** attempt  # exponential backoff: 0.5s, 1s, 2s
                    tqdm.write(f'  {ticker} attempt {attempt + 1} failed, retrying in {wait:.1f}s: {e}')
                    time.sleep(wait)
                else:
                    tqdm.write(f'  skipping {ticker} after {retries} attempts: {e}')

    if not rows:
        return pd.DataFrame(columns=fields)
    return pd.DataFrame(rows).set_index('ticker')


def fundamentals(symbols: dict, specs: dict = FUNDAMENTALS_SPECS, force: bool = False) -> dict:
    """
    Fetches and caches fundamental data per asset class and update frequency.

    symbols : {'equities': [...], 'etfs': [...], 'cryptos': [...]}
    specs   : {'equities': {'daily': [...], 'weekly': [...]}, ...}
               defaults to FUNDAMENTALS_SPECS — pass a subset to run only part of it
    force   : skip the staleness check and always re-fetch from yfinance

    Returns {'equities': {'daily': df, 'weekly': df, ...}, 'etfs': {...}, ...}.
    Each df is indexed by ticker with one column per field.
    Results are cached to data/fundamentals/{asset_class}_{freq}.parquet.
    """
    FUNDAMENTALS_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    for asset_class, freq_map in specs.items():
        tickers = symbols.get(asset_class, [])
        if not tickers:
            continue
        results[asset_class] = {}
        for freq, fields in freq_map.items():
            path = FUNDAMENTALS_DIR / f'{asset_class}_{freq}.parquet'
            if not force and not _needs_update(path, freq):
                results[asset_class][freq] = pd.read_parquet(path)
                print(f'{asset_class}/{freq}: loaded cache ({path})')
                continue
            print(f'{asset_class}/{freq}: fetching {len(fields)} fields for {len(tickers)} tickers...')
            df = _fetch_fields(tickers, fields)
            df.to_parquet(path)
            print(f'{asset_class}/{freq}: saved {len(df)} rows to {path}')
            results[asset_class][freq] = df

    return results


screen = screen_symbols()

# download_daily_1min(screen['equities'] + screen['etfs'], screen['cryptos'])
results = fundamentals(screen)
print(results['equities']['daily'])
