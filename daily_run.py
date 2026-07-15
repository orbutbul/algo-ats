"""
daily_run.py — scheduled via Windows Task Scheduler at 5:00 PM ET.

Runs two pipelines in order:
  1. 1-minute OHLCV for today (equities, ETFs, crypto)
  2. Fundamentals (daily-freq fields only; weekly/monthly skipped if not stale)
"""

import logging
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas_market_calendars as mcal

from utils import screen_symbols
from data_ohlcv import download_daily_1min
from data_fundamentals import fundamentals, FUNDAMENTALS_SPECS

LOG_DIR = Path('logs')
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / 'daily_run.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_PATH, encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger('daily_run')


def is_market_day() -> bool:
    nyse = mcal.get_calendar('NYSE')
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    schedule = nyse.schedule(start_date=today, end_date=today)
    return not schedule.empty


def run():
    start = datetime.now()
    log.info('=' * 55)
    log.info('Daily run started: %s', start.strftime('%Y-%m-%d %H:%M:%S'))
    log.info('=' * 55)

    errors = []

    # ------------------------------------------------------------------
    # 1. Screen symbols (shared across all pipelines)
    # ------------------------------------------------------------------
    log.info('--- Screening symbols ---')
    try:
        screen = screen_symbols()
        log.info(
            'equities=%d, etfs=%d, cryptos=%d',
            len(screen['equities']), len(screen['etfs']), len(screen['cryptos']),
        )
    except Exception:
        log.error('FAILED to screen symbols — aborting run.')
        log.error(traceback.format_exc())
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. 1-minute OHLCV (only on market days)
    # ------------------------------------------------------------------
    log.info('--- 1-min OHLCV ---')
    if is_market_day():
        try:
            download_daily_1min(screen['equities'] + screen['etfs'], screen['cryptos'])
        except Exception:
            errors.append('1-min OHLCV')
            log.error(traceback.format_exc())
    else:
        log.info('Not a market day — skipping.')

    # ------------------------------------------------------------------
    # 3. Fundamentals (staleness-aware, only runs what needs updating)
    # ------------------------------------------------------------------
    log.info('--- Fundamentals ---')
    try:
        fundamentals(screen, specs=FUNDAMENTALS_SPECS)
    except Exception:
        errors.append('fundamentals')
        log.error(traceback.format_exc())

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = (datetime.now() - start).seconds
    log.info('=' * 55)
    if errors:
        log.warning('Completed with errors in %ds: %s', elapsed, ', '.join(errors))
    else:
        log.info('All pipelines completed successfully in %ds', elapsed)
    log.info('=' * 55)


if __name__ == '__main__':
    run()
