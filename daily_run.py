"""
daily_run.py — scheduled via Windows Task Scheduler at 5:00 PM ET.

Runs three pipelines in order:
  1. 1-minute OHLCV for today (equities, ETFs, crypto)
  2. Fundamentals (daily-freq fields only; weekly/monthly skipped if not stale)
  3. WSB app data from u/wsbapp's latest post
"""

import random
import sys
import traceback
from datetime import datetime, timezone

import pandas_market_calendars as mcal

from utils import screen_symbols
from data_ohlcv import download_daily_1min
from data_fundamentals import fundamentals, FUNDAMENTALS_SPECS
from wsb2 import get_latest_wsbapp_data, save_wsb_data


def is_market_day() -> bool:
    nyse = mcal.get_calendar('NYSE')
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    schedule = nyse.schedule(start_date=today, end_date=today)
    return not schedule.empty


def run():
    start = datetime.now()
    print(f"\n{'='*55}")
    print(f"  Daily run started: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*55}\n")

    errors = []

    # ------------------------------------------------------------------
    # 1. Screen symbols (shared across all pipelines)
    # ------------------------------------------------------------------
    print("--- Screening symbols ---")
    try:
        screen = screen_symbols()
        print(f"  equities={len(screen['equities'])}, etfs={len(screen['etfs'])}, cryptos={len(screen['cryptos'])}")
    except Exception:
        print("FAILED to screen symbols — aborting run.")
        traceback.print_exc()
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. 1-minute OHLCV (only on market days)
    # ------------------------------------------------------------------
    print("\n--- 1-min OHLCV ---")
    if is_market_day():
        try:
            download_daily_1min(screen['equities'] + screen['etfs'], screen['cryptos'])
        except Exception:
            errors.append('1-min OHLCV')
            traceback.print_exc()
    else:
        print("  Not a market day — skipping.")

    # ------------------------------------------------------------------
    # 3. Fundamentals (staleness-aware, only runs what needs updating)
    # ------------------------------------------------------------------
    print("\n--- Fundamentals ---")
    try:
        fundamentals(screen, specs=FUNDAMENTALS_SPECS)
    except Exception:
        errors.append('fundamentals')
        traceback.print_exc()

    # ------------------------------------------------------------------
    # 4. WSB app data
    # ------------------------------------------------------------------
    print("\n--- WSB app data ---")
    try:
        data = get_latest_wsbapp_data()
        print(f"  Post: {data['post_title']}")
        print(f"  Trending: {data['trendingTickersDaily']}")
        save_wsb_data(data)
    except Exception:
        errors.append('WSB')
        traceback.print_exc()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = (datetime.now() - start).seconds
    print(f"\n{'='*55}")
    if errors:
        print(f"  Completed with errors in {elapsed}s: {', '.join(errors)}")
    else:
        print(f"  All pipelines completed successfully in {elapsed}s")
    print(f"{'='*55}\n")
    print(f"FInished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, 'random id: ' + {str(random.randint(1000,9999))}")

if __name__ == '__main__':
    run()
