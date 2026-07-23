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
FUNDAMENTALS_DIR = Path('data/fundamentals')    # one parquet per {asset_class}_{freq}

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

# yfinance crypto coverage is too thin to rely on (no on-chain data, spotty
# ticker matches) — crypto fundamentals are sourced from CoinGecko instead
# (see fetch_crypto_fundamentals below), whose native field names are
# snake_case rather than the yfinance-derived camelCase used for
# equities/ETFs above. volume24Hr isn't a CoinGecko /coins/markets field;
# total_volume is the equivalent.
CRYPTO_DAILY_FIELDS = [
    "market_cap",            # price * circulating supply
    "circulating_supply",
    "market_cap_rank",
]

CRYPTO_MONTHLY_FIELDS = [
    "max_supply",
    "total_supply",
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
    'cryptos': {
        'daily':   CRYPTO_DAILY_FIELDS,
        'monthly': CRYPTO_MONTHLY_FIELDS,
    },
}

# Minimum age (days) before a cached parquet is considered stale
FREQ_STALENESS_DAYS = {'daily': 1, 'weekly': 7, 'monthly': 30}





COINGECKO_BASE = 'https://api.coingecko.com/api/v3'
COINGECKO_COINS_LIST_PATH = FUNDAMENTALS_DIR / 'coingecko_coins_list.json'
CRYPTO_UNMATCHED_PATH = FUNDAMENTALS_DIR / 'crypto_unmatched.json'
# CoinGecko free tier is roughly 10-30 calls/min (undocumented precisely) —
# 2.5s between calls keeps us well under that with margin for retries.
COINGECKO_RATE_LIMIT_DELAY = 2.5

# Same quote-asset suffixes utils._tracked_symbols() uses to detect crypto
# tickers, reused here so "what counts as a quote-asset suffix" has exactly
# one definition in the codebase.
_QUOTE_ASSET_RE = re.compile(r'(USDT|BUSD|BTC|ETH|BNB)$')


def _base_asset(binance_symbol: str) -> str:
    """'BTCUSDT' -> 'BTC'."""
    return _QUOTE_ASSET_RE.sub('', binance_symbol)


def _fetch_coingecko_coins_list(force: bool = False) -> pd.DataFrame:
    """GET /coins/list — full id/symbol/name universe (~15k rows), cached
    under the same 'daily' staleness rule as everything else."""
    if not force and not _needs_update(COINGECKO_COINS_LIST_PATH, 'daily'):
        with open(COINGECKO_COINS_LIST_PATH, encoding='utf-8') as f:
            return pd.DataFrame(json.load(f))

    resp = requests.get(f'{COINGECKO_BASE}/coins/list')
    resp.raise_for_status()
    coins = resp.json()

    COINGECKO_COINS_LIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(COINGECKO_COINS_LIST_PATH, 'w', encoding='utf-8') as f:
        json.dump(coins, f)
    return pd.DataFrame(coins)


def _fetch_coingecko_markets(coin_ids: list[str], vs_currency: str = 'usd',
                              delay: float = COINGECKO_RATE_LIMIT_DELAY, retries: int = 3) -> pd.DataFrame:
    """GET /coins/markets, batched at 250 ids/call (CoinGecko's per_page max)."""
    frames = []
    for i in range(0, len(coin_ids), 250):
        batch = coin_ids[i:i + 250]
        for attempt in range(retries):
            try:
                resp = requests.get(f'{COINGECKO_BASE}/coins/markets', params={
                    'vs_currency': vs_currency,
                    'ids': ','.join(batch),
                    'per_page': 250,
                })
                resp.raise_for_status()
                frames.append(pd.DataFrame(resp.json()))
                time.sleep(delay)
                break
            except Exception as e:
                if attempt < retries - 1:
                    wait = delay * 2 ** attempt
                    print(f'  CoinGecko markets batch failed, retrying in {wait:.1f}s: {e}')
                    time.sleep(wait)
                else:
                    print(f'  skipping CoinGecko markets batch after {retries} attempts: {e}')
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _map_symbols_to_coingecko_ids(binance_symbols: list[str], coins_list: pd.DataFrame) -> tuple[dict, list]:
    """
    Maps Binance ticker (e.g. 'BTCUSDT') -> CoinGecko coin id (e.g. 'bitcoin').
    Collisions (multiple coins sharing a symbol) are resolved by picking the
    candidate with the best (lowest) market_cap_rank.
    Returns (symbol_to_id, unmatched_bases).
    """
    coins_list = coins_list.copy()
    coins_list['symbol'] = coins_list['symbol'].str.upper()
    by_symbol = coins_list.groupby('symbol')['id'].apply(list).to_dict()

    base_by_symbol = {s: _base_asset(s) for s in binance_symbols}
    candidates_by_base = {}
    unmatched = []
    for symbol, base in base_by_symbol.items():
        ids = by_symbol.get(base)
        if not ids:
            unmatched.append(base)
        else:
            candidates_by_base[base] = ids

    # Only fetch market data (for rank-based disambiguation) for bases with
    # more than one candidate — the common case (single match) needs no
    # extra API call.
    ambiguous_ids = [cid for ids in candidates_by_base.values() if len(ids) > 1 for cid in ids]
    rank_by_id = {}
    if ambiguous_ids:
        markets = _fetch_coingecko_markets(ambiguous_ids)
        if not markets.empty:
            rank_by_id = markets.set_index('id')['market_cap_rank'].to_dict()

    resolved_by_base = {}
    for base, ids in candidates_by_base.items():
        if len(ids) == 1:
            resolved_by_base[base] = ids[0]
        else:
            resolved_by_base[base] = min(ids, key=lambda cid: rank_by_id.get(cid) or float('inf'))

    symbol_to_id = {
        symbol: resolved_by_base[base]
        for symbol, base in base_by_symbol.items()
        if base in resolved_by_base
    }
    return symbol_to_id, unmatched


def fetch_crypto_fundamentals(binance_symbols: list[str], force: bool = False) -> pd.DataFrame:
    """
    Fetches crypto fundamentals from CoinGecko for a list of Binance tickers
    (e.g. 'BTCUSDT'). Returns a DataFrame indexed by the original Binance
    ticker with columns market_cap, circulating_supply, market_cap_rank,
    max_supply, total_supply, plus coingecko_id for traceability.
    Tickers that can't be matched to a CoinGecko coin are omitted here and
    logged to CRYPTO_UNMATCHED_PATH.
    """
    if not binance_symbols:
        return pd.DataFrame()

    coins_list = _fetch_coingecko_coins_list(force=force)
    symbol_to_id, unmatched = _map_symbols_to_coingecko_ids(binance_symbols, coins_list)

    FUNDAMENTALS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CRYPTO_UNMATCHED_PATH, 'w', encoding='utf-8') as f:
        json.dump({'unmatched_bases': unmatched, 'checked_at': datetime.now(timezone.utc).isoformat()}, f, indent=2)

    if not symbol_to_id:
        return pd.DataFrame()

    markets = _fetch_coingecko_markets(list(set(symbol_to_id.values())))
    if markets.empty:
        return pd.DataFrame()

    markets = markets.set_index('id')
    rows = []
    for symbol, coin_id in symbol_to_id.items():
        if coin_id not in markets.index:
            continue
        m = markets.loc[coin_id]
        rows.append({
            'ticker': symbol,
            'coingecko_id': coin_id,
            'market_cap': m.get('market_cap'),
            'circulating_supply': m.get('circulating_supply'),
            'market_cap_rank': m.get('market_cap_rank'),
            'max_supply': m.get('max_supply'),
            'total_supply': m.get('total_supply'),
        })
    return pd.DataFrame(rows).set_index('ticker')


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
    df = pd.DataFrame(rows).set_index('ticker')
    # Coerce numeric columns and replace inf (e.g. trailingPE on loss-making stocks)
    df = df.apply(pd.to_numeric, errors='ignore')
    df = df.replace([float('inf'), float('-inf')], float('nan'))
    return df


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
            if asset_class == 'cryptos':
                df = fetch_crypto_fundamentals(tickers, force=force)
                df = df[[f for f in fields if f in df.columns]]
            else:
                df = _fetch_fields(tickers, fields)
            df.to_parquet(path)
            print(f'{asset_class}/{freq}: saved {len(df)} rows to {path}')
            results[asset_class][freq] = df

    return results



METADATA_PATH = Path('data/metadata.json')


def get_static_metadata(
    symbols: dict | None = None,
    force: bool = False,
) -> dict:
    """
    Fetch the full yf.Ticker.info dict for all equities and ETFs and save to
    data/metadata.json, keyed by ticker.

    symbols : {'equities': [...], 'etfs': [...]} or None to use screen_symbols().
    force   : re-fetch even if the file already exists.
    """
    if not force and METADATA_PATH.exists():
        print(f'Loading cached metadata from {METADATA_PATH}')
        with open(METADATA_PATH, encoding='utf-8') as f:
            return json.load(f)

    if symbols is None:
        symbols = screen_symbols()

    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    metadata: dict = {}

    for asset_class in ('equities', 'etfs'):
        tickers = symbols.get(asset_class, [])
        if not tickers:
            continue
        print(f'Fetching metadata for {len(tickers)} {asset_class}...')
        for ticker in tqdm(tickers, desc=asset_class, unit='ticker'):
            for attempt in range(3):
                try:
                    info = yf.Ticker(ticker).info
                    info['asset_class'] = asset_class
                    metadata[ticker] = info
                    time.sleep(0.3)
                    break
                except Exception as e:
                    if attempt < 2:
                        time.sleep(0.3 * 2 ** attempt)
                    else:
                        tqdm.write(f'  skipping {ticker}: {e}')

    with open(METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f'Saved metadata for {len(metadata)} tickers to {METADATA_PATH}')
    return metadata


if __name__ == '__main__':
    screen = screen_symbols()
    results = fundamentals(screen)
    print(results['equities']['daily'])
