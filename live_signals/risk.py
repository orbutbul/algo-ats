"""Daily loss, drawdown, and the kill switch.

Execution role: pure updates of the risk flags the planner reads. The engine
decides whether the kill file or env var is set and whether a reset was
requested; this module only applies that boolean to the persisted state.

A drawdown halt stays on until a reset is requested, and the reset rebases
the peak to current equity. A daily-loss halt stays on until the next
America/New_York date. The first cycle of a day snapshots equity, so a book
that is already down at process start does not trip until it loses another
`daily_loss_limit_pct` from that snapshot.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from live_signals.config import Config
from live_signals.models import RiskState
from live_signals.sessions import to_et


def kill_switch_active(cfg: Config, kill_file: str, env: dict[str, str] | None = None) -> bool:
    """True when LIVE_SIGNALS_KILL=1 or the kill file exists.

    Only the exact value `1` counts. The variable is not cleared here: the
    operator owns it for as long as they want the engine flat.
    """
    source = os.environ if env is None else env
    if source.get(cfg.kill_env) == "1":
        return True
    return Path(kill_file).exists()


def consume_drawdown_reset(cfg: Config, reset_file: str, env: dict[str, str] | None = None) -> bool:
    """Honor a one-shot drawdown reset and disarm it.

    Deletes the reset file and, when `env` is the real process environment,
    pops the env var. Leaving either armed would un-halt the next breaker in
    the same breath it tripped.
    """
    source = os.environ if env is None else env
    armed = source.get(cfg.reset_drawdown_env) == "1" or Path(reset_file).exists()
    path = Path(reset_file)
    if path.exists():
        path.unlink()
    if source.get(cfg.reset_drawdown_env) == "1":
        source.pop(cfg.reset_drawdown_env, None)
    return armed


def step_risk(
    state: RiskState,
    equity: float,
    now: datetime,
    cfg: Config,
    *,
    reset_requested: bool,
) -> RiskState:
    peak = state.peak_equity
    halted = state.drawdown_halted
    if reset_requested and halted:
        halted = False
        peak = equity
    if not halted:
        peak = max(peak, equity)
        if peak > 0 and equity <= peak * (1 - cfg.max_drawdown_pct):
            halted = True

    day = to_et(now).date()
    if state.day != day:
        day_start = equity
        day_halted = False
        trades = 0
        new_risk = 0.0
    else:
        day_start = state.day_start_equity
        day_halted = state.day_halted
        trades = state.trades_today
        new_risk = state.new_risk_today
    if day_start <= 0 or equity - day_start <= -cfg.daily_loss_limit_pct * day_start:
        day_halted = True
    return RiskState(
        peak_equity=peak,
        drawdown_halted=halted,
        day=day,
        day_start_equity=day_start,
        day_halted=day_halted,
        trades_today=trades,
        new_risk_today=new_risk,
    )
