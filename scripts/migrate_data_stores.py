"""
scripts/migrate_data_stores.py — one-shot local data reorganization.

1. Copy massive_1min + robinhood_1min from data/ohlcv.duckdb into
   data/ohlcv_research.duckdb, then DROP them from ohlcv.duckdb.
2. Import legacy data/fundamentals/*.parquet into database/fundamentals.duckdb.
3. Optionally merge a PE snapshot CSV/CSV.GZ into equities_monthly + equities.

Run from the repo root after pulling this change:

    python scripts/migrate_data_stores.py
    python scripts/migrate_data_stores.py --pe-snapshot path/to/ats_equities_pe_snapshot.csv.gz

Safe to re-run (CREATE OR REPLACE / upsert semantics).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from extraction.fundamentals import (  # noqa: E402
    FUNDAMENTALS_DB_PATH,
    LEGACY_FUNDAMENTALS_DIR,
    _refresh_equities_wide,
    _save_fundamentals_table,
    import_pe_snapshot,
)

OHLCV_PATH = REPO_ROOT / 'data' / 'ohlcv.duckdb'
RESEARCH_PATH = REPO_ROOT / 'data' / 'ohlcv_research.duckdb'
RESEARCH_TABLES = ('massive_1min', 'robinhood_1min')


def migrate_research_ohlcv(drop_from_live: bool = True) -> None:
    if not OHLCV_PATH.exists():
        print(f'skip research migrate: {OHLCV_PATH} not found')
        return

    RESEARCH_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Copy via ATTACH from the destination side only (single connection to source).
    research = duckdb.connect(str(RESEARCH_PATH))
    try:
        research.execute(f"ATTACH '{OHLCV_PATH.as_posix()}' AS live (READ_ONLY)")
        live_tables = {
            r[0] for r in research.execute(
                "SELECT table_name FROM duckdb_tables() WHERE database_name = 'live'"
            ).fetchall()
        }
        for table in RESEARCH_TABLES:
            if table not in live_tables:
                print(f'  {table}: not in ohlcv.duckdb — skip')
                continue
            n = research.execute(f'SELECT COUNT(*) FROM live.{table}').fetchone()[0]
            print(f'  copying {table} ({n:,} rows) -> {RESEARCH_PATH.name}')
            research.execute(
                f'CREATE OR REPLACE TABLE {table} AS SELECT * FROM live.{table}'
            )
        research.execute('DETACH live')
    finally:
        research.close()

    if not drop_from_live:
        return

    live = duckdb.connect(str(OHLCV_PATH))
    try:
        live_tables = {r[0] for r in live.execute('SHOW TABLES').fetchall()}
        for table in RESEARCH_TABLES:
            if table in live_tables:
                live.execute(f'DROP TABLE IF EXISTS {table}')
                print(f'  dropped {table} from ohlcv.duckdb')
    finally:
        live.close()


def migrate_fundamentals_parquet() -> None:
    if not LEGACY_FUNDAMENTALS_DIR.exists():
        print(f'skip fundamentals parquet migrate: {LEGACY_FUNDAMENTALS_DIR} not found')
        return

    import pandas as pd

    parquet_files = sorted(LEGACY_FUNDAMENTALS_DIR.glob('*_*.parquet'))
    if not parquet_files:
        print(f'  no parquet files under {LEGACY_FUNDAMENTALS_DIR}')
        return

    for path in parquet_files:
        stem = path.stem  # e.g. equities_monthly
        if '_' not in stem:
            continue
        asset_class, freq = stem.split('_', 1)
        df = pd.read_parquet(path)
        if 'ticker' not in df.columns and df.index.name != 'ticker':
            print(f'  skip {path.name}: no ticker')
            continue
        if 'ticker' not in df.columns:
            df = df.copy()
            df.index.name = 'ticker'
        indexed = df if df.index.name == 'ticker' else df.set_index('ticker')
        _save_fundamentals_table(asset_class, freq, indexed)
        print(f'  imported {path.name} -> {FUNDAMENTALS_DB_PATH.name}::{stem} ({len(df):,} rows)')

    if FUNDAMENTALS_DB_PATH.exists():
        con = duckdb.connect(str(FUNDAMENTALS_DB_PATH))
        try:
            _refresh_equities_wide(con)
        finally:
            con.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        '--pe-snapshot',
        type=Path,
        default=None,
        help='CSV/CSV.GZ with ticker + PE/growth columns to merge into equities_monthly',
    )
    ap.add_argument(
        '--keep-live-research-tables',
        action='store_true',
        help='Copy research tables but do not DROP them from ohlcv.duckdb',
    )
    args = ap.parse_args()

    print('=== migrate research OHLCV ===')
    migrate_research_ohlcv(drop_from_live=not args.keep_live_research_tables)

    print('=== migrate fundamentals parquet -> database/ ===')
    migrate_fundamentals_parquet()

    if args.pe_snapshot is not None:
        print(f'=== import PE snapshot {args.pe_snapshot} ===')
        n = import_pe_snapshot(args.pe_snapshot, merge=True)
        print(f'  wrote {n:,} equities_monthly rows (merged)')

    print('done')


if __name__ == '__main__':
    main()
