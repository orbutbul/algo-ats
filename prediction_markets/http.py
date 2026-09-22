"""
prediction_markets/http.py — shared HTTP helper for the venue
clients under prediction_markets/venues/. Same retry shape as
extraction/ohlcv_massive.py's `_get`/`_RateLimiter` (steady-interval throttle
+ backoff on 429/5xx), generalized so every venue client uses one retry
policy instead of four slightly different ones.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import requests

LOG_PATH = Path(__file__).parent.parent.parent / 'logs' / 'prediction_markets.log'
DEFAULT_TIMEOUT = 15          # seconds
DEFAULT_RETRIES = 3
DEFAULT_BACKOFF_BASE = 2      # seconds; backoff = base * 2**attempt, capped at DEFAULT_BACKOFF_CAP

log = logging.getLogger('prediction_markets')
if not log.handlers:
    log.setLevel(logging.INFO)
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        handler: logging.Handler = logging.FileHandler(LOG_PATH)
    except OSError:
        handler = logging.NullHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
    log.addHandler(handler)


class RateLimiter:
    """Steady-interval throttle: keeps spacing between calls at
    >= 1/calls_per_second seconds. Mirrors ohlcv_massive._RateLimiter,
    parameterized by calls/sec instead of calls/min since every prediction-
    market venue documents its limit that way."""

    def __init__(self, calls_per_second: float):
        self.interval = 1.0 / calls_per_second
        self._last_call = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self._last_call = time.monotonic()


class VenueHTTPError(Exception):
    """Non-retryable HTTP error from a venue (4xx other than 429)."""


def get_json(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    limiter: RateLimiter | None = None,
    retries: int = DEFAULT_RETRIES,
    timeout: int = DEFAULT_TIMEOUT,
) -> Any:
    """GET `url`, returning parsed JSON. Retries with exponential backoff on
    429/5xx/timeouts/connection errors; raises VenueHTTPError immediately on
    any other 4xx (a bug in the request, not an outage — retrying won't help)."""
    last_exc: Exception | None = None
    for attempt in range(retries):
        if limiter is not None:
            limiter.wait()
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
        except (requests.ConnectionError, requests.Timeout) as e:
            last_exc = e
            wait = DEFAULT_BACKOFF_BASE * (2 ** attempt)
            log.warning('network error on %s (attempt %d/%d): %s — retrying in %ds', url, attempt + 1, retries, e, wait)
            time.sleep(wait)
            continue

        if resp.status_code == 429 or resp.status_code >= 500:
            wait = DEFAULT_BACKOFF_BASE * (2 ** attempt)
            log.warning('%d on %s (attempt %d/%d) — retrying in %ds', resp.status_code, url, attempt + 1, retries, wait)
            time.sleep(wait)
            continue
        if 400 <= resp.status_code < 500:
            raise VenueHTTPError(f'{resp.status_code} on {url}: {resp.text[:500]}')

        resp.raise_for_status()
        return resp.json()

    raise VenueHTTPError(f'exceeded {retries} retries fetching {url}') from last_exc
