"""
relative_strength/io.py — read-only inputs and new RS artifacts.

The only repo price history this adapter reads is data/ohlcv.duckdb table
ohlcv_max_daily, and only when a caller asks. The connection is read-only.
Nothing here creates tables, writes parquet, or touches credentials.

ohlcv_max_daily is the daily bar table written by extraction.ohlcv.get_max_data
via yfinance.download(auto_adjust=True). That adjustment is splits and
dividends together. 1-minute tables are not resampled: they are a different
session definition and a different adjustment policy, and a short incremental
pull is not 253 sessions.

CSV/JSON artifacts are new files under the caller-chosen output directory.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from relative_strength.calculator import PriceSchemaError, RelativeStrengthResult

# Anchored to the repo root (parent of this package), same idea as utils.py,
# without importing utils.py.
REPO_ROOT = Path(__file__).resolve().parents[1]
OHLCV_DUCKDB_PATH = REPO_ROOT / 'data' / 'ohlcv.duckdb'
OHLCV_MAX_TABLE = 'ohlcv_max_daily'
METADATA_PARQUET = REPO_ROOT / 'data' / 'metadata_consolidated.parquet'
METADATA_JSON = REPO_ROOT / 'data' / 'metadata.json'

PRICE_SOURCE_DUCKDB = 'ohlcv.duckdb::ohlcv_max_daily'
ADJUSTMENT_DUCKDB = 'split_and_dividend'
ADJUSTMENT_NOTE_DUCKDB = (
    'Closes are read from data/ohlcv.duckdb table ohlcv_max_daily, written by '
    'extraction.ohlcv.get_max_data via yfinance.download(interval="1d", auto_adjust=True). '
    'yfinance auto_adjust=True adjusts OHLC for splits and dividends. '
    'These are not split-only closes, not raw closes, and not official IBD prices. '
    '1-minute tables are not read: ohlcv_1min_airflow is Alpaca Adjustment.SPLIT and only an '
    'incremental intraday window; massive_1min is requested with adjusted=true (vendor split '
    'adjustment) at 1-minute resolution; robinhood_1min is 1-minute bounds=24_5 with no '
    'adjustment policy recorded in this repo. None of those are turned into session closes here.'
)


@dataclass(frozen=True)
class DailyCoverage:
    """What the read-only daily adapter can see. ``gap`` explains a missing source."""

    available: bool
    duckdb_path: str
    table: str
    gap: str | None
    rows: int | None = None
    tickers: int | None = None
    min_date: str | None = None
    max_date: str | None = None
    null_or_nonfinite_closes: int | None = None
    nonpositive_closes: int | None = None
    tables_present: tuple[str, ...] = ()
    price_source: str | None = None
    adjustment: str | None = None
    adjustment_note: str | None = None

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload['tables_present'] = list(self.tables_present)
        return payload


def _gap_missing_file(path: Path) -> str:
    return (
        f'No daily close history at {path}. Expected DuckDB table {OHLCV_MAX_TABLE}, '
        'written by extraction.ohlcv.get_max_data (yfinance daily bars, auto_adjust=True, '
        'split and dividend adjusted). data/ is gitignored, so a checkout does not include it. '
        'This calculator does not create the file, does not download prices, and does not resample '
        '1-minute tables (ohlcv_1min_airflow, massive_1min, robinhood_1min). '
        'Supply --input prices.csv, or point --duckdb-path at an existing read-only ohlcv.duckdb '
        f'that already contains {OHLCV_MAX_TABLE}.'
    )


def _connect_readonly(path: Path):
    import duckdb

    if not path.exists():
        raise FileNotFoundError(_gap_missing_file(path))
    return duckdb.connect(str(path), read_only=True)


def describe_daily_coverage(path: Path | None = None) -> DailyCoverage:
    """Read-only description of ohlcv_max_daily. Never creates the database."""
    path = Path(path) if path is not None else OHLCV_DUCKDB_PATH
    if not path.exists():
        return DailyCoverage(
            available=False,
            duckdb_path=str(path),
            table=OHLCV_MAX_TABLE,
            gap=_gap_missing_file(path),
            adjustment_note=ADJUSTMENT_NOTE_DUCKDB,
        )
    con = _connect_readonly(path)
    try:
        tables = tuple(row[0] for row in con.execute('SHOW TABLES').fetchall())
        if OHLCV_MAX_TABLE not in tables:
            return DailyCoverage(
                available=False,
                duckdb_path=str(path),
                table=OHLCV_MAX_TABLE,
                gap=(
                    f'{path} has no table {OHLCV_MAX_TABLE}. Tables present: '
                    f'{list(tables) if tables else "(none)"}. '
                    '1-minute tables are not a substitute and were not resampled. '
                    'The calculator does not create ohlcv_max_daily.'
                ),
                tables_present=tables,
                adjustment_note=ADJUSTMENT_NOTE_DUCKDB,
            )
        row = con.execute(f'''
            SELECT
                COUNT(*) AS n_rows,
                COUNT(DISTINCT ticker) AS n_tickers,
                CAST(MIN(datetime) AS DATE) AS min_dt,
                CAST(MAX(datetime) AS DATE) AS max_dt,
                COUNT(*) FILTER (
                    WHERE close IS NULL OR close <> close OR NOT isfinite(close)
                ) AS bad_close,
                COUNT(*) FILTER (WHERE close <= 0) AS nonpositive
            FROM {OHLCV_MAX_TABLE}
        ''').fetchone()
    finally:
        con.close()
    return DailyCoverage(
        available=True,
        duckdb_path=str(path),
        table=OHLCV_MAX_TABLE,
        gap=None,
        rows=int(row[0]),
        tickers=int(row[1]),
        min_date=_date_text(row[2]),
        max_date=_date_text(row[3]),
        null_or_nonfinite_closes=int(row[4]),
        nonpositive_closes=int(row[5]),
        tables_present=tables,
        price_source=PRICE_SOURCE_DUCKDB,
        adjustment=ADJUSTMENT_DUCKDB,
        adjustment_note=ADJUSTMENT_NOTE_DUCKDB,
    )


def load_ohlcv_max_daily(path: Path | None = None, tickers: list[str] | None = None) -> pd.DataFrame:
    """Read daily closes. Raises FileNotFoundError or LookupError with the gap text.

    The frame has columns date, ticker, close. The database is opened read-only.
    """
    path = Path(path) if path is not None else OHLCV_DUCKDB_PATH
    con = _connect_readonly(path)
    try:
        tables = [row[0] for row in con.execute('SHOW TABLES').fetchall()]
        if OHLCV_MAX_TABLE not in tables:
            raise LookupError(
                f'{path} has no table {OHLCV_MAX_TABLE}. Tables present: {tables or "(none)"}. '
                'This calculator does not resample 1-minute tables and does not create the daily table.'
            )
        # VARCHAR, not a pandas/duckdb timestamptz conversion: that path imports pytz,
        # which this package does not depend on. Midnight UTC labels stay midnight strings.
        sql = (
            f'SELECT CAST(datetime AS VARCHAR) AS date, ticker, close FROM {OHLCV_MAX_TABLE}'
        )
        params: list[str] = []
        if tickers:
            placeholders = ', '.join('?' * len(tickers))
            sql += f' WHERE ticker IN ({placeholders})'
            params = [str(t).strip().upper() for t in tickers]
        sql += ' ORDER BY 1, 2'
        rows = con.execute(sql, params).fetchall()
    finally:
        con.close()
    return pd.DataFrame(rows, columns=['date', 'ticker', 'close'])


def load_price_csv(path: Path) -> pd.DataFrame:
    """Load a caller-supplied CSV. The file is only read."""
    return pd.read_csv(path)


def load_asset_class_map(path: Path) -> dict[str, str]:
    """CSV with columns ticker, asset_class. Read-only."""
    frame = pd.read_csv(path)
    lookup = {str(col).strip().lower(): col for col in frame.columns}
    if 'ticker' not in lookup or 'asset_class' not in lookup:
        raise PriceSchemaError('asset-class CSV needs columns ticker and asset_class')
    out: dict[str, str] = {}
    for ticker, asset_class in zip(frame[lookup['ticker']], frame[lookup['asset_class']], strict=True):
        if pd.isna(ticker):
            continue
        out[str(ticker).strip().upper()] = '' if pd.isna(asset_class) else str(asset_class)
    return out


def load_repo_asset_classes(
    parquet_path: Path | None = None,
    json_path: Path | None = None,
) -> dict[str, str]:
    """Read consolidated metadata if it is already on disk. Never fetches it.

    Prefers metadata_consolidated.parquet (asset_class column) and otherwise
    metadata.json objects that already contain asset_class. Missing files yield
    an empty map; the caller decides whether that blocks a stock/ETF split.
    """
    parquet_path = METADATA_PARQUET if parquet_path is None else Path(parquet_path)
    json_path = METADATA_JSON if json_path is None else Path(json_path)
    if parquet_path.exists():
        frame = pd.read_parquet(parquet_path)
        if 'asset_class' not in frame.columns:
            return {}
        tickers = frame.index if frame.index.name == 'ticker' or 'ticker' not in frame.columns else frame['ticker']
        if 'ticker' in frame.columns:
            pairs = zip(frame['ticker'], frame['asset_class'], strict=True)
        else:
            pairs = zip(tickers, frame['asset_class'], strict=True)
        return {
            str(ticker).strip().upper(): str(asset_class)
            for ticker, asset_class in pairs
            if not pd.isna(ticker) and not pd.isna(asset_class)
        }
    if json_path.exists():
        payload = json.loads(json_path.read_text(encoding='utf-8'))
        out: dict[str, str] = {}
        if isinstance(payload, dict):
            for ticker, info in payload.items():
                if isinstance(info, dict) and info.get('asset_class'):
                    out[str(ticker).strip().upper()] = str(info['asset_class'])
        return out
    return {}


def _date_text(value) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        return None
    return ts.date().isoformat()


def _json_scalar(value):
    if value is None:
        return None
    if isinstance(value, (bool,)):
        return bool(value)
    # bool is an int; already returned above.
    if isinstance(value, (int,)):
        return int(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (date, datetime, pd.Timestamp)):
        if pd.isna(value):
            return None
        return pd.Timestamp(value).date().isoformat()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, 'item'):
        return _json_scalar(value.item())
    return value


def records_from_frame(frame: pd.DataFrame) -> list[dict]:
    rows = []
    for record in frame.to_dict(orient='records'):
        rows.append({key: _json_scalar(val) for key, val in record.items()})
    return rows


def result_payload(result: RelativeStrengthResult) -> dict:
    return {
        'summary': result.summary,
        'ratings': records_from_frame(result.ratings),
        'rs_line': records_from_frame(result.rs_line),
    }


def write_result(result: RelativeStrengthResult, out_dir: Path) -> dict[str, Path]:
    """Write a new CSV and JSON. Does not touch the source database or input file."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = result.summary.get('as_of_session') or result.summary.get('as_of_requested') or 'undated'
    csv_path = out_dir / f'rs_proxy_{stamp}.csv'
    json_path = out_dir / f'rs_proxy_{stamp}.json'
    result.ratings.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(result_payload(result), indent=2) + '\n', encoding='utf-8')
    paths = {'csv': csv_path, 'json': json_path}
    if not result.rs_line.empty:
        line_path = out_dir / f'rs_line_{stamp}.csv'
        result.rs_line.to_csv(line_path, index=False)
        paths['rs_line_csv'] = line_path
    return paths
