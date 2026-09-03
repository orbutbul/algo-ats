"""
oracle_cloud/cloud_sync.py — pull the Oracle Cloud VM's accumulated pipeline data
into local data/*.duckdb, verify the merge, then wipe the cloud copy.

Run manually, whenever you want a sync (no daemon/listener on either end):
    python oracle_cloud/cloud_sync.py --host <public-ip> --key /path/to/private_key

The VM is treated as a lean, ephemeral buffer: local data/*.duckdb is the
permanent store and always grows; the VM's copies only ever hold one sync
interval's worth of rows and get wiped clean after each *verified* pull,
while its incremental-fetch bookkeeping files (ohlcv_robinhood_last_run.txt,
ohlcv_robinhood_progress.json, ohlcv_last_run.txt, news_last_run.txt) are
left untouched on the VM so the cloud pipeline resumes where it left off
instead of re-fetching full history next run. See DEPLOY.md.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import duckdb

# Running this as `python oracle_cloud/cloud_sync.py` only puts this file's
# own directory on sys.path, not the repo root -- needed for the
# extraction.news import below (and so DATA_DIR/TMP_DIR below resolve
# relative to wherever this is invoked from, run this from the repo root).
sys.path.insert(0, str(Path(__file__).parent.parent))
from extraction.news import _TABLES as NEWS_TABLES  # noqa: E402

DATA_DIR = Path('data')
TMP_DIR = DATA_DIR / '.cloud_sync_tmp'
REMOTE_REPO_DIR = 'ATS_trading'  # ~/ATS_trading on the VM

# dedup key per table, one entry per data/*.duckdb file synced. WSB isn't
# synced -- it never runs on the cloud (Reddit blocks the scraper from
# datacenter IPs; see wsb_run.py and DEPLOY.md), so data/wsb.duckdb never
# exists there. news keys are imported straight from the extraction module
# that owns them (extraction/news.py::_TABLES) so this script can't drift
# out of sync with the real upsert keys. The OHLCV table names/keys are
# hardcoded rather than imported from extraction/ohlcv*.py, since
# extraction/ohlcv_robinhood.py transitively imports robinhood_client.py,
# which raises at import time if Robinhood credentials aren't configured --
# not a dependency this sync script should have just to read a table name.
DB_SPECS: dict[str, dict[str, list[str]]] = {
    'news.duckdb': {name: spec['dedup_cols'] for name, spec in NEWS_TABLES.items()},
    'ohlcv.duckdb': {
        'ohlcv_1min_airflow': ['datetime', 'ticker'],  # extraction/ohlcv.py::OHLCV_AIRFLOW_TABLE
        'ohlcv_max_daily': ['datetime', 'ticker'],      # extraction/ohlcv.py::OHLCV_MAX_TABLE
        'robinhood_1min': ['datetime', 'ticker'],        # extraction/ohlcv_robinhood.py::STAGING_TABLE
        'massive_1min': ['datetime', 'ticker'],          # extraction/ohlcv_massive.py::STAGING_TABLE
    },
}


def _run(cmd: list[str]) -> None:
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def pull(host: str, key: str, user: str, port: int) -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    for db_name in DB_SPECS:
        remote = f'{user}@{host}:{REMOTE_REPO_DIR}/data/{db_name}'
        local = TMP_DIR / db_name
        _run(['scp', '-i', key, '-P', str(port), remote, str(local)])


def merge() -> dict[str, dict[str, int]]:
    """Upserts every pulled table into the matching local db (delete-then-
    insert on the table's dedup key, same pattern the extraction modules
    already use). Returns {db_name: {table: cloud_row_count}} for verify()."""
    pulled_counts: dict[str, dict[str, int]] = {}
    for db_name, tables in DB_SPECS.items():
        cloud_path = TMP_DIR / db_name
        if not cloud_path.exists():
            print(f'  {db_name}: no pulled file, skipping')
            continue
        pulled_counts[db_name] = {}
        con = duckdb.connect(str(DATA_DIR / db_name))
        try:
            con.execute(f"ATTACH '{cloud_path.as_posix()}' AS cloud (READ_ONLY)")
            cloud_tables = {
                r[0] for r in con.execute(
                    "SELECT table_name FROM duckdb_tables() WHERE database_name = 'cloud'"
                ).fetchall()
            }
            for table, dedup_cols in tables.items():
                if table not in cloud_tables:
                    continue
                cloud_count = con.execute(f'SELECT COUNT(*) FROM cloud.{table}').fetchone()[0]
                pulled_counts[db_name][table] = cloud_count
                if cloud_count == 0:
                    continue
                con.execute(f'CREATE TABLE IF NOT EXISTS {table} AS SELECT * FROM cloud.{table} WHERE 1=0')
                key_cols = ', '.join(dedup_cols)
                con.execute(f'DELETE FROM {table} WHERE ({key_cols}) IN (SELECT {key_cols} FROM cloud.{table})')
                con.execute(f'INSERT INTO {table} SELECT * FROM cloud.{table}')
                print(f'  {db_name}::{table}: merged {cloud_count} rows')
            con.execute('DETACH cloud')
        finally:
            con.close()
    return pulled_counts


def verify(pulled_counts: dict[str, dict[str, int]]) -> None:
    """Confirms every pulled row is now present locally under its dedup key.
    Raises (aborting before the wipe) on any shortfall."""
    for db_name, tables in pulled_counts.items():
        if not any(c > 0 for c in tables.values()):
            continue
        con = duckdb.connect(str(DATA_DIR / db_name), read_only=True)
        try:
            con.execute(f"ATTACH '{(TMP_DIR / db_name).as_posix()}' AS cloud (READ_ONLY)")
            for table, cloud_count in tables.items():
                if cloud_count == 0:
                    continue
                key_cols = ', '.join(DB_SPECS[db_name][table])
                matched = con.execute(
                    f'SELECT COUNT(*) FROM {table} '
                    f'WHERE ({key_cols}) IN (SELECT {key_cols} FROM cloud.{table})'
                ).fetchone()[0]
                if matched < cloud_count:
                    raise RuntimeError(
                        f'{db_name}::{table}: only {matched}/{cloud_count} pulled rows '
                        f'verified present locally -- aborting wipe'
                    )
                print(f'  {db_name}::{table}: verified {matched}/{cloud_count} rows present locally')
            con.execute('DETACH cloud')
        finally:
            con.close()


def wipe(host: str, key: str, user: str, port: int, docker: bool) -> None:
    db_names = ' '.join(name.removesuffix('.duckdb') for name in DB_SPECS)
    if docker:
        # A1 + Docker/Airflow deploy path (airflow/docker-compose.yaml).
        remote_cmd = (
            f'cd {REMOTE_REPO_DIR} && '
            f'docker compose -f airflow/docker-compose.yaml exec -T airflow-scheduler '
            f'python oracle_cloud/remote_wipe.py {db_names}'
        )
    else:
        # E2.1.Micro + native venv/cron deploy path (default) -- no
        # containers, just the venv's python directly. See DEPLOY.md.
        remote_cmd = f'cd {REMOTE_REPO_DIR} && venv/bin/python oracle_cloud/remote_wipe.py {db_names}'
    _run(['ssh', '-i', key, '-p', str(port), f'{user}@{host}', remote_cmd])


def cleanup_tmp() -> None:
    for f in TMP_DIR.glob('*.duckdb'):
        f.unlink()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--host', required=True, help='VM public IP or hostname')
    p.add_argument('--key', required=True, help='path to the SSH private key')
    p.add_argument('--user', default='ubuntu')
    p.add_argument('--port', type=int, default=22)
    p.add_argument('--docker', action='store_true', help='remote is the A1+Docker/Airflow deploy, not the default native venv/cron one')
    args = p.parse_args()

    print('Pulling data from the cloud VM...')
    pull(args.host, args.key, args.user, args.port)

    print('Merging into local data/*.duckdb...')
    pulled_counts = merge()

    print('Verifying merge...')
    verify(pulled_counts)

    print('Wiping cloud data tables (bookkeeping files left untouched)...')
    wipe(args.host, args.key, args.user, args.port, args.docker)

    cleanup_tmp()
    print('Sync complete.')


if __name__ == '__main__':
    main()
