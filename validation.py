"""
validation.py — data-quality checks + push-notification alerting for the
daily/hourly pipelines (both the legacy Task Scheduler scripts and the
Airflow DAGs).

Every validate_* function returns a list[str] of human-readable issues
(empty list = pass). None of them raise on a *data quality* problem —
only callers decide what to do with the issues list. Alerting goes out
via ntfy.sh (a plain HTTP POST, no account/credentials) rather than email.
"""

from __future__ import annotations

import os
import traceback
from datetime import date, datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd
import pandas_market_calendars as mcal
import requests
from dotenv import load_dotenv

load_dotenv()

from extraction.fundamentals import FUNDAMENTALS_DIR, FUNDAMENTALS_SPECS
from extraction.wsb import WSB_DB_PATH

OHLCV_DUCKDB_PATH = Path('data/ohlcv.duckdb')
OHLCV_AIRFLOW_TABLE = 'ohlcv_1min_airflow'

# Fraction of Close/Volume that may be null in today's rows before flagging.
OHLCV_NULL_FRACTION_THRESHOLD = 0.05


def is_market_day(on: date | None = None) -> bool:
    """NYSE market-day check (duplicated in daily_run.py / daily_run_dag.py —
    kept here too so validate_ohlcv can be called standalone)."""
    nyse = mcal.get_calendar('NYSE')
    d = (on or datetime.now(timezone.utc).date()).strftime('%Y-%m-%d')
    return not nyse.schedule(start_date=d, end_date=d).empty


def send_alert(subject: str, body: str) -> None:
    """POST a push notification to the ntfy.sh topic in NTFY_TOPIC. Never
    raises — a notification-service outage shouldn't crash a pipeline run."""
    topic = os.environ.get('NTFY_TOPIC')
    if not topic:
        print(f'[validation] NTFY_TOPIC not configured — skipping alert. Subject: {subject}')
        return
    try:
        requests.post(
            f'https://ntfy.sh/{topic}',
            data=body.encode('utf-8'),
            headers={'Title': subject, 'Priority': 'high', 'Tags': 'warning'},
            timeout=15,
        )
        print(f'[validation] Alert sent: {subject}')
    except Exception:
        print(f'[validation] Failed to send alert:\n{traceback.format_exc()}')


def alert_on_task_failure(context: dict) -> None:
    """Airflow task-level on_failure_callback — fires once per task, only
    after its retries are exhausted (not once per attempt)."""
    ti = context['task_instance']
    body = (
        f"DAG: {ti.dag_id}\nTask: {ti.task_id}\nRun: {context.get('run_id')}\n"
        f"Log: {ti.log_url}\n\nException:\n{context.get('exception')}"
    )
    send_alert(f'[ATS] Airflow task failed: {ti.dag_id}.{ti.task_id}', body)


def validate_screen(screen: dict) -> list[str]:
    if not (screen.get('equities') or screen.get('etfs') or screen.get('cryptos')):
        return ['screen_symbols() returned no equities, etfs, or cryptos']
    return []


def validate_ohlcv(screen: dict, run_date: date | None = None) -> list[str]:
    run_date = run_date or datetime.now(timezone.utc).date()
    if not is_market_day(run_date):
        return []

    label = f'{OHLCV_DUCKDB_PATH}::{OHLCV_AIRFLOW_TABLE}'
    if not OHLCV_DUCKDB_PATH.exists():
        return [f'{label} does not exist']

    con = duckdb.connect(str(OHLCV_DUCKDB_PATH), read_only=True)
    try:
        tables = {r[0] for r in con.execute('SHOW TABLES').fetchall()}
        if OHLCV_AIRFLOW_TABLE not in tables:
            return [f'{label} does not exist']
        todays = con.execute(
            f'SELECT close, volume FROM {OHLCV_AIRFLOW_TABLE} WHERE datetime::DATE = ?',
            [run_date],
        ).df()
    finally:
        con.close()

    if todays.empty:
        return [f'{label}: no rows for {run_date} (market day)']

    issues = []
    for col in ('close', 'volume'):
        frac = todays[col].isna().mean()
        if frac > OHLCV_NULL_FRACTION_THRESHOLD:
            issues.append(f'{label}: {col} is {frac:.0%} null for {run_date}')
    return issues


def validate_fundamentals(screen: dict, specs: dict = FUNDAMENTALS_SPECS) -> list[str]:
    issues = []
    for asset_class, freq_map in specs.items():
        if not screen.get(asset_class):
            continue
        for freq in freq_map:
            path = FUNDAMENTALS_DIR / f'{asset_class}_{freq}.parquet'
            if not path.exists():
                issues.append(f'{path} does not exist')
                continue
            if len(pd.read_parquet(path)) == 0:
                issues.append(f'{path} exists but has 0 rows')
    return issues


def validate_wsb(run_date: date | None = None, run_hour: int | None = None) -> list[str]:
    now = datetime.now(timezone.utc)
    run_date = run_date or now.date()
    run_hour = run_hour if run_hour is not None else now.hour

    if not WSB_DB_PATH.exists():
        return [f'{WSB_DB_PATH} does not exist']

    issues = []
    con = duckdb.connect(str(WSB_DB_PATH), read_only=True)
    try:
        for table in ('mentions_v2', 'positions_v2'):
            n = con.execute(
                f'SELECT count(*) FROM {table} WHERE date = ? AND hour = ?',
                [run_date, run_hour],
            ).fetchone()[0]
            if n == 0:
                issues.append(f'{table}: 0 rows for {run_date} hour={run_hour}')

        row = con.execute(
            'SELECT bullish_pct, bearish_pct FROM sentiment WHERE date = ? AND hour = ?',
            [run_date, run_hour],
        ).fetchone()
        if row is None:
            issues.append(f'sentiment: no row for {run_date} hour={run_hour}')
        elif row[0] is None and row[1] is None:
            issues.append(f'sentiment: bullish_pct and bearish_pct both null for {run_date} hour={run_hour}')
    finally:
        con.close()
    return issues
