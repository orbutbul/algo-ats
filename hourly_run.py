"""
hourly_run.py — scheduled via Windows Task Scheduler, hourly.

Runs:
  1. WSB widget data (mentions, sentiment, leaderboard, holdings, trades) —
     appended as one snapshot per UTC hour, building an intraday history.
"""

import logging
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

# When launched via pythonw.exe (no console, so nothing pops up on screen),
# sys.stdout/stderr are None — bare print()s and the StreamHandler below
# would crash on the first write. Give them a harmless sink instead.
if sys.stdout is None:
    sys.stdout = open(os.devnull, 'w')
if sys.stderr is None:
    sys.stderr = open(os.devnull, 'w')

from extraction.wsb import get_latest_wsb_data, save_wsb_data
from validation import validate_wsb, send_alert

LOG_DIR = Path('logs')
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / 'hourly_run.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_PATH, encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger('hourly_run')


def run():
    start = datetime.now(timezone.utc)
    log.info('=' * 55)
    log.info('Hourly run started: %s', start.strftime('%Y-%m-%d %H:%M:%S'))
    log.info('=' * 55)

    # ------------------------------------------------------------------
    # 1. WSB widget data (mentions, sentiment, leaderboard, holdings, trades)
    # ------------------------------------------------------------------
    log.info('--- WSB data ---')
    issues = []
    try:
        wsb_data = get_latest_wsb_data(post_type='moves')
        save_wsb_data(wsb_data)
        log.info('WSB data saved successfully')
    except Exception:
        log.error(traceback.format_exc())
        issues.append('WSB scrape/save failed')
    else:
        issues += validate_wsb(run_date=start.date(), run_hour=start.hour)

    if issues:
        log.warning('Validation issues: %s', '; '.join(issues))
        send_alert(f'[ATS] hourly_run issues ({start.strftime("%Y-%m-%d %H:00")} UTC)', '\n'.join(issues))

    elapsed = (datetime.now(timezone.utc) - start).seconds
    log.info('=' * 55)
    log.info('Hourly run finished in %ds', elapsed)
    log.info('=' * 55)


if __name__ == '__main__':
    run()
