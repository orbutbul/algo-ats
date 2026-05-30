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
OHLCV_COLS = ['open', 'high', 'low', 'close', 'volume']
BATCH_SIZE = 100                                # tickers per yfinance download call


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



def _tracked_symbols() -> dict:
    """
    Reads ohlcv_1min.parquet and returns unique tickers bucketed by asset class,
    so symbols that drop out of the screener are still maintained.
    Returns {'cryptos': [...], 'equities': [...], 'etfs': [...]} or empty lists if the file doesn't exist.
    """
    if not OHLCV_PATH.exists():
        return {'cryptos': [], 'equities': [], 'etfs': []}

    tickers = pd.read_parquet(OHLCV_PATH, columns=[]).index.get_level_values('ticker').unique().tolist()

    # Crypto tickers on BinanceUS end with USDT/BUSD/BTC/ETH/BNB
    crypto_pattern = re.compile(r'(USDT|BUSD|BTC|ETH|BNB)$')
    cryptos = [t for t in tickers if crypto_pattern.search(t)]
    non_cryptos = [t for t in tickers if not crypto_pattern.search(t)]

    # Split non-cryptos into ETFs vs equities using yfinance quoteType in bulk isn't feasible,
    # so bucket everything non-crypto as equities — ETF distinction only matters for fundamentals,
    # and the screener results will correctly label them there.
    return {'cryptos': cryptos, 'equities': non_cryptos, 'etfs': []}


def screen_symbols():
    """
    Returns {'cryptos': [...], 'equities': [...], 'etfs': [...]}.
    Pulls the top symbols for each asset class from the TradingView screener,
    then merges in any previously tracked symbols from ohlcv_1min.parquet.
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

    tracked = _tracked_symbols()
    etf_set = set(etfs)
    return {
        'cryptos':  list(dict.fromkeys(cryptos  + tracked['cryptos'])),
        'equities': list(dict.fromkeys(equities + [t for t in tracked['equities'] if t not in etf_set])),
        'etfs':     list(dict.fromkeys(etfs)),
    }


def compute_allocations(
    signals: pd.DataFrame,
    method: str = "signal",
    top_k: int | None = None,
    long_only: bool = True,
) -> pd.DataFrame:
    """
    Convert per-asset signal strengths into portfolio weight allocations.

    Parameters
    ----------
    signals : pd.DataFrame
        Rows = timestamps, columns = assets, values = signal strengths.
        NaN values are treated as no signal (zero allocation).
    method : {'equal', 'signal', 'rank', 'inverse_rank'}
        equal        — equal weight among selected assets
        signal       — weight proportional to signal value
        rank         — weight proportional to rank (weakest=1, strongest=N)
        inverse_rank — weight proportional to 1/rank (strongest=rank 1 → weight 1,
                       second=rank 2 → weight 1/2, etc.)
    top_k : int or None
        Restrict allocation to the top-k assets by signal strength.
        If None, all assets with positive signal are eligible.
    long_only : bool
        If True (default), negative signals are floored to zero.
        If False, absolute signal value is used for sizing; sign is preserved
        and weights are normalised by the sum of absolute weights.

    Returns
    -------
    pd.DataFrame
        Same shape as signals. Each row sums to 1 in absolute value
        (or 0 if no valid signals exist for that row).
        Long-only: all values in [0, 1].
        Long/short: values in [-1, 1], abs values sum to 1.
    """
    valid_methods = {"equal", "signal", "rank", "inverse_rank"}
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}, got '{method}'")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be >= 1")

    signals = signals.copy().fillna(0.0)

    def _allocate_row(row: pd.Series) -> pd.Series:
        signs = np.sign(row)

        if long_only:
            eligible = row[row > 0]
        else:
            eligible = row[row != 0].abs()

        if eligible.empty:
            return pd.Series(0.0, index=row.index)

        if top_k is not None and top_k < len(eligible):
            eligible = eligible.nlargest(top_k)

        if method == "equal":
            weights = pd.Series(1.0, index=eligible.index)
        elif method == "signal":
            weights = eligible
        elif method == "rank":
            weights = eligible.rank(method="first")
        elif method == "inverse_rank":
            weights = 1.0 / eligible.rank(method="first", ascending=False)

        weights = weights / weights.sum()

        result = pd.Series(0.0, index=row.index)
        if long_only:
            result[weights.index] = weights
        else:
            result[weights.index] = weights * signs[weights.index]

        return result

    return signals.apply(_allocate_row, axis=1)
