"""
IO and CLI tests. DuckDB fixtures are synthetic and temporary.
They are not the repo's market history, and they are opened read-only.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from relative_strength.calculator import CALENDAR_LOOKBACK_CALENDAR_DAYS
from relative_strength.cli import main
from relative_strength.io import (
    OHLCV_DUCKDB_PATH,
    describe_daily_coverage,
    load_ohlcv_max_daily,
    load_repo_asset_classes,
    write_result,
)
from relative_strength.tests.test_calculator import AS_OF, _flat, _frame, full_calendar

REPO = Path(__file__).resolve().parents[2]


def _ranked_csv(tmp_path: Path) -> Path:
    days = full_calendar()
    series = {f'T{i:02d}': _flat(days, 100.0 + i) for i in range(1, 31)}
    series['SPY'] = _flat(days, 100.0)
    classes = {ticker: 'equities' for ticker in series}
    classes['SPY'] = 'etfs'
    frame = _frame(series, classes)
    path = tmp_path / 'synthetic_prices.csv'
    frame.to_csv(path, index=False)
    return path


def test_repo_checkout_has_no_daily_history_and_does_not_create_it():
    assert not OHLCV_DUCKDB_PATH.exists()
    report = describe_daily_coverage(OHLCV_DUCKDB_PATH)
    assert report.available is False
    assert report.table == 'ohlcv_max_daily'
    assert 'ohlcv_max_daily' in report.gap
    assert '1-minute' in report.gap
    assert 'does not create' in report.gap
    assert not OHLCV_DUCKDB_PATH.exists()
    assert not (REPO / 'data').exists()


def test_duckdb_read_is_readonly_and_ignores_minute_tables(tmp_path: Path):
    path = tmp_path / 'ohlcv.duckdb'
    con = duckdb.connect(str(path))
    con.execute('''
        CREATE TABLE ohlcv_max_daily (
            datetime TIMESTAMPTZ NOT NULL,
            ticker VARCHAR NOT NULL,
            close DOUBLE
        )
    ''')
    con.execute('''
        CREATE TABLE massive_1min (
            datetime TIMESTAMPTZ,
            ticker VARCHAR,
            close DOUBLE
        )
    ''')
    con.execute(
        "INSERT INTO ohlcv_max_daily VALUES (TIMESTAMPTZ '2024-06-28 00:00:00+00', 'AAA', 12.5)"
    )
    con.execute("INSERT INTO massive_1min VALUES (TIMESTAMPTZ '2024-06-28 20:00:00+00', 'AAA', 99)")
    before = con.execute(
        'SELECT CAST(datetime AS VARCHAR), ticker, close FROM ohlcv_max_daily ORDER BY 1, 2'
    ).fetchall()
    tables = sorted(row[0] for row in con.execute('SHOW TABLES').fetchall())
    con.close()

    mode = path.stat().st_mode
    path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    try:
        report = describe_daily_coverage(path)
        assert report.available is True
        assert report.rows == 1
        assert report.tickers == 1
        assert report.min_date == '2024-06-28'
        assert report.max_date == '2024-06-28'
        assert report.adjustment == 'split_and_dividend'
        assert 'not split-only' in report.adjustment_note
        assert 'massive_1min' in report.tables_present
        frame = load_ohlcv_max_daily(path)
        assert list(frame.columns) == ['date', 'ticker', 'close']
        assert frame.iloc[0]['ticker'] == 'AAA'
        assert frame.iloc[0]['close'] == pytest.approx(12.5)
    finally:
        path.chmod(mode)

    con = duckdb.connect(str(path), read_only=True)
    after = con.execute(
        'SELECT CAST(datetime AS VARCHAR), ticker, close FROM ohlcv_max_daily ORDER BY 1, 2'
    ).fetchall()
    tables_after = sorted(row[0] for row in con.execute('SHOW TABLES').fetchall())
    con.close()
    assert before == after
    assert tables_after == tables
    assert not path.with_name(path.name + '.wal').exists()


def test_missing_daily_table_names_the_gap(tmp_path: Path):
    path = tmp_path / 'ohlcv.duckdb'
    con = duckdb.connect(str(path))
    con.execute('CREATE TABLE robinhood_1min (datetime TIMESTAMPTZ, ticker VARCHAR, close DOUBLE)')
    con.close()
    report = describe_daily_coverage(path)
    assert report.available is False
    assert 'robinhood_1min' in report.gap
    assert 'not a substitute' in report.gap
    with pytest.raises(LookupError, match='ohlcv_max_daily'):
        load_ohlcv_max_daily(path)


def test_metadata_json_is_read_only(tmp_path: Path):
    payload = {'Aaa': {'asset_class': 'equities', 'trailingPE': 10}, 'spy': {'asset_class': 'etfs'}}
    path = tmp_path / 'metadata.json'
    path.write_text(json.dumps(payload), encoding='utf-8')
    before = path.read_bytes()
    classes = load_repo_asset_classes(parquet_path=tmp_path / 'missing.parquet', json_path=path)
    assert classes == {'AAA': 'equities', 'SPY': 'etfs'}
    assert path.read_bytes() == before
    assert load_repo_asset_classes(
        parquet_path=tmp_path / 'no.parquet', json_path=tmp_path / 'no.json',
    ) == {}


def test_cli_writes_new_artifacts_and_does_not_touch_the_csv(tmp_path: Path):
    source = _ranked_csv(tmp_path)
    original = source.read_bytes()
    out = tmp_path / 'out'
    code = main([
        'run',
        '--input', str(source),
        '--as-of', '2024-06-28',
        '--universe', 'stocks',
        '--adjustment', 'unknown',
        '--out-dir', str(out),
        '--no-rs-line',
    ])
    assert code == 0
    assert source.read_bytes() == original
    payload = json.loads((out / 'rs_proxy_2024-06-28.json').read_text(encoding='utf-8'))
    assert payload['summary']['method']['official_ibd_rs'] is False
    assert payload['summary']['eligible_count'] == 30
    assert payload['summary']['adjustment'] == 'unknown'
    assert payload['summary']['broad_market_rating'] is False
    by_ticker = {row['ticker']: row for row in payload['ratings']}
    assert by_ticker['T30']['rs_proxy'] == 99
    assert by_ticker['T01']['rs_proxy'] == 1
    assert by_ticker['T30']['r_63'] == pytest.approx(0.30)
    assert by_ticker['SPY']['reason'] == 'outside_universe'
    assert 'P/E' not in json.dumps(payload['summary'])
    csv_text = (out / 'rs_proxy_2024-06-28.csv').read_text(encoding='utf-8')
    assert csv_text.splitlines()[0].startswith('as_of_session,as_of_requested,ticker')
    assert not (REPO / 'data').exists()


def test_cli_module_and_duckdb_path(tmp_path: Path):
    source = tmp_path / 'tiny.csv'
    source.write_text('date,ticker,close,asset_class\n2024-06-28,AAA,10,equities\n', encoding='utf-8')
    original = source.read_bytes()
    out = tmp_path / 'cli-out'
    proc = subprocess.run(
        [
            sys.executable, '-m', 'relative_strength', 'run',
            '--input', str(source),
            '--as-of', '2024-06-28',
            '--universe', 'stocks',
            '--out-dir', str(out),
            '--no-rs-line',
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert source.read_bytes() == original
    assert 'short_history' in (out / 'rs_proxy_2024-06-28.json').read_text(encoding='utf-8')
    assert 'requests' not in proc.stderr
    assert 'order' not in proc.stderr.lower()

    db = tmp_path / 'ohlcv.duckdb'
    con = duckdb.connect(str(db))
    con.execute('CREATE TABLE ohlcv_max_daily (datetime TIMESTAMPTZ, ticker VARCHAR, close DOUBLE)')
    con.execute("INSERT INTO ohlcv_max_daily VALUES (TIMESTAMPTZ '2024-06-28 00:00:00+00', 'AAA', 10)")
    con.close()
    db_bytes = db.read_bytes()
    refused = main([
        'run', '--from-duckdb', '--duckdb-path', str(db),
        '--as-of', '2024-06-28', '--adjustment', 'split', '--out-dir', str(tmp_path / 'nope'),
    ])
    assert refused == 2
    ran = main([
        'run', '--from-duckdb', '--duckdb-path', str(db),
        '--as-of', '2024-06-28', '--universe', 'unseparated',
        '--out-dir', str(tmp_path / 'db-out'), '--no-rs-line',
    ])
    assert ran == 0
    assert db.read_bytes() == db_bytes
    body = json.loads((tmp_path / 'db-out' / 'rs_proxy_2024-06-28.json').read_text(encoding='utf-8'))
    assert body['summary']['adjustment'] == 'split_and_dividend'
    assert body['summary']['price_source'] == 'ohlcv.duckdb::ohlcv_max_daily'
    assert 'not split-only' in body['summary']['adjustment_note']
    assert body['ratings'][0]['as_of_session'] == '2024-06-28'


def test_coverage_command_on_missing_file(tmp_path: Path):
    missing = tmp_path / 'absent.duckdb'
    code = main(['coverage', '--duckdb-path', str(missing)])
    assert code == 2
    assert not missing.exists()


def test_import_and_compute_do_not_touch_network_or_brokers():
    script = '''
import socket
import sys
from datetime import date

def boom(*args, **kwargs):
    raise AssertionError("network connect")

socket.socket.connect = boom
socket.create_connection = boom

import relative_strength
from relative_strength.calculator import compute_relative_strength
import pandas as pd

assert "duckdb" not in sys.modules
assert "requests" not in sys.modules
assert "yfinance" not in sys.modules
assert "robinhood_client" not in sys.modules

frame = pd.DataFrame({
    "date": [date(2024, 6, 28)],
    "ticker": ["AAA"],
    "close": [10.0],
    "asset_class": ["equities"],
})
compute_relative_strength(frame, as_of="2024-06-28", universe="stocks", include_rs_line=False)
assert "duckdb" not in sys.modules
assert "requests" not in sys.modules
assert "yfinance" not in sys.modules
assert "robinhood_client" not in sys.modules
'''
    proc = subprocess.run([sys.executable, '-c', script], cwd=REPO, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr


def test_writer_does_not_require_the_source_path(tmp_path: Path):
    from relative_strength.calculator import compute_relative_strength

    frame = pd.DataFrame({
        'date': [date(2024, 6, 28)],
        'ticker': ['AAA'],
        'close': [10.0],
        'asset_class': ['equities'],
    })
    result = compute_relative_strength(frame, as_of=date(2024, 6, 28), universe='stocks', include_rs_line=False)
    paths = write_result(result, tmp_path / 'artifacts')
    assert paths['csv'].is_file()
    assert paths['json'].is_file()
    assert 'rs_line_csv' not in paths
    # A longer as-of with no new file at the source.
    assert not (tmp_path / 'ohlcv.duckdb').exists()
    # Silence unused import if the calendar constant is kept for readers of this module.
    assert CALENDAR_LOOKBACK_CALENDAR_DAYS > 252
    assert os.access(paths['json'], os.R_OK)
