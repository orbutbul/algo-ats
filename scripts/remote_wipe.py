"""
scripts/remote_wipe.py — deletes all rows from every table in the given
data/*.duckdb file(s), leaving the schema and the incremental-fetch
bookkeeping files (*_last_run.txt, *_progress.json -- plain files, outside
these databases, untouched by this script regardless) intact.

Runs on the Oracle Cloud VM only, invoked by scripts/cloud_sync.py (local
machine) over SSH after it has verified every row from these files was
pulled and merged into the local data/*.duckdb copies. Never run this
directly unless you're sure the data has already been copied out -- it does
not check that itself.

Wipes every table it finds rather than a hardcoded list: these DB files only
ever hold ephemeral data tables on the cloud VM (never bookkeeping), so
"every table" and "everything that needs wiping" are the same set -- and a
hardcoded list would silently drift out of sync with the real schemas
defined in extraction/wsb.py::_TABLES, extraction/news.py::_TABLES, etc.

Usage (inside the airflow-scheduler container):
    python scripts/remote_wipe.py wsb news ohlcv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

DATA_DIR = Path('data')
DB_CHOICES = ['wsb', 'news', 'ohlcv']


def wipe_db(db_name: str) -> None:
    db_path = DATA_DIR / f'{db_name}.duckdb'
    if not db_path.exists():
        print(f'  {db_path}: does not exist, skipping')
        return
    con = duckdb.connect(str(db_path))
    try:
        tables = [r[0] for r in con.execute('SHOW TABLES').fetchall()]
        for table in tables:
            con.execute(f'DELETE FROM {table}')
            print(f'  {db_path}::{table}: wiped')
    finally:
        con.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('dbs', nargs='+', choices=DB_CHOICES, help='db shorthand(s) to wipe, e.g. wsb news ohlcv')
    args = p.parse_args()

    for db_name in args.dbs:
        wipe_db(db_name)


if __name__ == '__main__':
    main()
