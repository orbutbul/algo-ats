"""Log-only polling loop.

Execution role: `python -m live_signals` reads a Wirewatch JSON file and a
market JSON file, plans a cycle, and appends the decision log. It forces
mode `log_only` even if the config file says paper, and it never constructs
an Alpaca client. Paper submission stays behind `Engine` plus an explicit
broker, which this entry point does not wire up.

Example:
  python -m live_signals --signals signals.json --market market.json --once
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

from live_signals.config import load_config, replace_config
from live_signals.engine import Engine
from live_signals.journal import DecisionLog
from live_signals.notify import MemoryNotifier
from live_signals.marketdata import enrich_market
from live_signals.sources import EmptyMarketSource, JsonFileMarketSource, JsonFileSignalSource
from live_signals.state import StateStore


def build_engine(args: argparse.Namespace) -> Engine:
    cfg = load_config(args.config) if args.config else load_config()
    if cfg.mode != "log_only":
        cfg = replace_config(cfg, mode="log_only")
    state_path = args.state or cfg.state_path
    log_path = args.log or cfg.decision_log_path
    state = StateStore(state_path, cfg.reference_equity)
    journal = DecisionLog(log_path)
    return Engine(
        cfg,
        state,
        journal,
        broker=None,
        notifier=MemoryNotifier(),
        kill_file=str(Path(args.kill_file or cfg.kill_file)),
        reset_file=str(Path(args.reset_file or cfg.reset_file)),
    )


def run_loop(engine: Engine, signals: JsonFileSignalSource, market: JsonFileMarketSource, *,
             sleep, clock, max_cycles: int | None) -> int:
    completed = 0
    while max_cycles is None or completed < max_cycles:
        now = clock()
        batch = signals.fetch(now)
        snapshot = enrich_market(market.fetch(now), [item.ticker for item in batch], now)
        engine.run_once(now, batch, snapshot)
        completed += 1
        if max_cycles is not None and completed >= max_cycles:
            break
        sleep(engine.config.poll_seconds)
    return completed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Wirewatch live-signal log-only loop")
    parser.add_argument("--signals", required=True, help="Wirewatch JSON file, re-read each poll")
    parser.add_argument(
        "--market", default=None,
        help="Optional quotes/ATR JSON. Tickers missing from it are loaded from Alpaca, then Yahoo.",
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--state", default=None)
    parser.add_argument("--log", default=None)
    parser.add_argument("--kill-file", default=None)
    parser.add_argument("--reset-file", default=None)
    parser.add_argument("--once", action="store_true", help="Run a single cycle and exit")
    parser.add_argument("--cycles", type=int, default=None, help="Stop after N cycles (tests)")
    args = parser.parse_args(argv)
    engine = build_engine(args)
    cycles = 1 if args.once else args.cycles
    run_loop(
        engine,
        JsonFileSignalSource(args.signals),
        JsonFileMarketSource(args.market) if args.market else EmptyMarketSource(),
        sleep=time.sleep,
        clock=lambda: datetime.now(timezone.utc),
        max_cycles=cycles,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
