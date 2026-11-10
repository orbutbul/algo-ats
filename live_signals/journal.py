"""Append-only decision log.

Execution role: every signal the cycle considered, and every exit or halt,
is one CSV row. Taken and skipped rows both carry a reason. The file is the
review surface for the log-only week.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from live_signals.models import Decision

COLUMNS = [
    "ts",
    "ticker",
    "signal_id",
    "score",
    "sentiment",
    "action",
    "reason",
    "qty",
    "limit_price",
    "session",
    "mode",
    "details",
]


class DecisionLog:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, decisions: list[Decision]) -> None:
        if not decisions:
            return
        new_file = not self.path.exists() or self.path.stat().st_size == 0
        with self.path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            if new_file:
                writer.writeheader()
            for decision in decisions:
                writer.writerow(
                    {
                        "ts": decision.ts.isoformat(),
                        "ticker": decision.ticker,
                        "signal_id": decision.signal_id,
                        "score": "" if decision.score is None else decision.score,
                        "sentiment": decision.sentiment,
                        "action": decision.action,
                        "reason": decision.reason,
                        "qty": decision.qty,
                        "limit_price": "" if decision.limit_price is None else decision.limit_price,
                        "session": decision.session,
                        "mode": decision.mode,
                        "details": json.dumps(decision.details, sort_keys=True, default=str),
                    }
                )
