"""
daily_run.py — scheduled via Windows Task Scheduler at 5:00 PM ET.

Runs, independently of one another (a failure in one does not block the rest):
  1. Symbol screening (equities/ETFs/cryptos — feeds sections 2 and 3)
  2. 1-minute OHLCV for today (equities, ETFs, crypto)
  3. Fundamentals (daily-freq fields only; weekly/monthly skipped if not stale)
  4. WSB widget data (mentions, sentiment, leaderboard, holdings, trades)
"""

import logging
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas_market_calendars as mcal

from utils import screen_symbols
from extraction.ohlcv import download_daily_1min
from extraction.fundamentals import fundamentals, FUNDAMENTALS_SPECS
from extraction.wsb import get_latest_wsb_data, save_wsb_data

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
    # 1. Screen symbols (shared across sections 2 and 3 — section 4 doesn't
    #    depend on this, and still runs even if screening fails)
    # ------------------------------------------------------------------
    log.info('--- Screening symbols ---')
    screen = None
    try:
        screen = screen_symbols()
        log.info(
            'equities=%d, etfs=%d, cryptos=%d',
            len(screen['equities']), len(screen['etfs']), len(screen['cryptos']),
        )
    except Exception:
        errors.append('symbol screening')
        log.error(traceback.format_exc())

    # ------------------------------------------------------------------
    # 2. 1-minute OHLCV (only on market days)
    # ------------------------------------------------------------------
    log.info('--- 1-min OHLCV ---')
    if screen is None:
        errors.append('1-min OHLCV (skipped — symbol screening failed)')
        log.warning('Skipping — symbol screening failed.')
    elif is_market_day():
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
    if screen is None:
        errors.append('fundamentals (skipped — symbol screening failed)')
        log.warning('Skipping — symbol screening failed.')
    else:
        try:
            fundamentals(screen, specs=FUNDAMENTALS_SPECS)
        except Exception:
            errors.append('fundamentals')
            log.error(traceback.format_exc())

    # ------------------------------------------------------------------
    # 4. WSB widget data (mentions, sentiment, leaderboard, holdings, trades)
    # ------------------------------------------------------------------
    log.info('--- WSB data ---')
    try:
        wsb_data = get_latest_wsb_data(post_type='moves')
        save_wsb_data(wsb_data)
    except Exception:
        errors.append('WSB data')
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
