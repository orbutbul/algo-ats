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