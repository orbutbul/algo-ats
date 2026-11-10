"""Idempotent client order ids.

Execution role: the same logical order (purpose, ticker, signal, tranche,
attempt) always maps to the same Alpaca `client_order_id`, so a retry cannot
double-send. Ids are a short hash, not a concatenation of the headline.
"""

from __future__ import annotations

import hashlib


def client_order_id(
    prefix: str,
    purpose: str,
    ticker: str,
    signal_id: str,
    tranche: int,
    attempt: int = 0,
) -> str:
    raw = f"{purpose}|{ticker.upper()}|{signal_id}|{tranche}|{attempt}"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:20]
    return f"{prefix}-{digest}"
