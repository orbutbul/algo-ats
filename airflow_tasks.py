"""
airflow_tasks.py — thin CLI entrypoints for Airflow's DockerOperator tasks.

Each subcommand wraps one step from daily_run.py / hourly_run.py so it can run
as its own container, with results passed between tasks via XCom (the last
line printed to stdout).

Usage:
    python airflow_tasks.py screen
    python airflow_tasks.py ohlcv '<json screen dict>'
    python airflow_tasks.py fundamentals '<json screen dict>'
    python airflow_tasks.py wsb
"""

import json
import sys
from datetime import datetime, timezone

import pandas_market_calendars as mcal


def _is_market_day() -> bool:
    nyse = mcal.get_calendar('NYSE')
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    return not nyse.schedule(start_date=today, end_date=today).empty


def cmd_screen():
    from utils import screen_symbols
    screen = screen_symbols()
    # last stdout line == XCom value (DockerOperator xcom_all=False)
    print(json.dumps(screen))


def cmd_ohlcv(screen_json: str):
    if not _is_market_day():
        print(json.dumps({'skipped': 'not a market day'}))
        return
    from extraction.ohlcv import download_daily_1min
    screen = json.loads(screen_json)
    download_daily_1min(screen['equities'] + screen['etfs'], screen['cryptos'])
    print(json.dumps({'status': 'ok'}))


def cmd_fundamentals(screen_json: str):
    from extraction.fundamentals import fundamentals, FUNDAMENTALS_SPECS
    screen = json.loads(screen_json)
    fundamentals(screen, specs=FUNDAMENTALS_SPECS)
    print(json.dumps({'status': 'ok'}))


def cmd_wsb():
    from extraction.wsb import get_latest_wsb_data, save_wsb_data
    wsb_data = get_latest_wsb_data(post_type='moves')
    save_wsb_data(wsb_data)
    print(json.dumps({'status': 'ok'}))


if __name__ == '__main__':
    command = sys.argv[1]
    args = sys.argv[2:]
    globals()[f'cmd_{command}'](*args)
