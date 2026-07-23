"""
extraction/metadata.py — consolidated, normalized per-ticker metadata across
all asset classes (equities, ETFs, crypto), saved to
data/metadata_consolidated.parquet.

Additive: does not replace data/metadata.json (raw yfinance blobs,
equities/ETFs) or data/fundamentals/{asset_class}_{freq}.parquet (curated,
staleness-tiered numeric fields) — this file sits on top of both, giving one
directly queryable row per tracked ticker regardless of asset class.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from extraction.fundamentals import (
    get_static_metadata,
    _fetch_coingecko_coins_list,
    _fetch_coingecko_markets,
    _map_symbols_to_coingecko_ids,
    _needs_update,
)
from utils import _tracked_symbols

CONSOLIDATED_METADATA_PATH = Path(__file__).parent.parent / 'data' / 'metadata_consolidated.parquet'

CORE_COLUMNS = [
    'asset_class', 'name', 'sector', 'industry', 'market_cap',
    'market_cap_rank', 'currency', 'exchange', 'country', 'source', 'last_updated',
]


def _row_from_yfinance_info(asset_class: str, info: dict, last_updated: datetime) -> dict:
    return {
        'asset_class': asset_class,
        'name': info.get('longName') or info.get('shortName'),
        'sector': info.get('sector'),
        'industry': info.get('industry'),
        'market_cap': info.get('marketCap'),
        'market_cap_rank': None,
        'currency': info.get('currency'),
        'exchange': info.get('fullExchangeName') or info.get('exchange'),
        'country': info.get('country'),
        'source': 'yfinance',
        'last_updated': last_updated,
        'raw': json.dumps(info, default=str),
    }


def _row_from_coingecko_market(market_row: pd.Series, last_updated: datetime) -> dict:
    return {
        'asset_class': 'cryptos',
        'name': market_row.get('name'),
        'sector': None,
        'industry': None,
        'market_cap': market_row.get('market_cap'),
        'market_cap_rank': market_row.get('market_cap_rank'),
        'currency': 'usd',
        'exchange': None,
        'country': None,
        'source': 'coingecko',
        'last_updated': last_updated,
        'raw': market_row.to_json(),
    }


def build_consolidated_metadata(
    equities_etfs_metadata: dict | None = None,
    crypto_symbols: list[str] | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """
    Builds/refreshes data/metadata_consolidated.parquet by combining
    extraction.fundamentals.get_static_metadata() (equities/ETFs) with a
    fuller CoinGecko /coins/markets pull (crypto), normalized into
    CORE_COLUMNS + a raw JSON blob column.
    """
    if not force and not _needs_update(CONSOLIDATED_METADATA_PATH, 'daily'):
        return pd.read_parquet(CONSOLIDATED_METADATA_PATH)

    now = datetime.now(timezone.utc)
    rows = {}

    if equities_etfs_metadata is None:
        equities_etfs_metadata = get_static_metadata(force=force)
    yf_last_updated = now
    if CONSOLIDATED_METADATA_PATH.parent.exists():
        from extraction.fundamentals import METADATA_PATH
        if METADATA_PATH.exists():
            yf_last_updated = datetime.fromtimestamp(METADATA_PATH.stat().st_mtime, tz=timezone.utc)

    for ticker, info in equities_etfs_metadata.items():
        asset_class = info.get('asset_class', 'equities')
        rows[ticker] = _row_from_yfinance_info(asset_class, info, yf_last_updated)

    if crypto_symbols is None:
        crypto_symbols = _tracked_symbols()['cryptos']

    if crypto_symbols:
        coins_list = _fetch_coingecko_coins_list(force=force)
        symbol_to_id, _unmatched = _map_symbols_to_coingecko_ids(crypto_symbols, coins_list)
        if symbol_to_id:
            markets = _fetch_coingecko_markets(list(set(symbol_to_id.values())))
            if not markets.empty:
                markets = markets.set_index('id')
                for symbol, coin_id in symbol_to_id.items():
                    if coin_id in markets.index:
                        rows[symbol] = _row_from_coingecko_market(markets.loc[coin_id], now)

    df = pd.DataFrame.from_dict(rows, orient='index')
    df.index.name = 'ticker'
    df = df[CORE_COLUMNS + ['raw']]

    CONSOLIDATED_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CONSOLIDATED_METADATA_PATH)
    print(f'Saved consolidated metadata for {len(df)} tickers to {CONSOLIDATED_METADATA_PATH}')
    return df


if __name__ == '__main__':
    build_consolidated_metadata()
