"""Load and validate the live-signal config.

Execution role: every tunable the design doc names lives in
`default_config.json`. `load_config` rejects unknown keys so a typo cannot
silently fall back to a default. Mode `live` is rejected here as well as at
order time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from datetime import time
from pathlib import Path
from typing import Any, Mapping

from live_signals.errors import ConfigError

_PATH = Path(__file__).with_name("default_config.json")

_SESSIONS = ("rth", "pre", "after", "overnight", "closed")
_OPEN_SESSIONS = ("rth", "pre", "after", "overnight")


@dataclass(frozen=True)
class Config:
    mode: str
    poll_seconds: int
    reference_equity: float
    freshness_minutes: float
    clock_skew_minutes: float
    min_score: float
    single_social_min_score: float
    single_social_confirmed_min_score: float
    multi_social_min_score: float
    single_news_min_score: float
    confirm_min_return: float
    confirm_min_rel_volume: float
    cooldown_minutes: float
    score_upgrade_delta: float
    conflict_window_minutes: float
    conflict_margin: float
    flip_min_score: float
    neutral_exit_score: float
    opposite_exit_score: float
    hold_until_score: float
    low_score_trail_pct: float
    time_exits_enabled: bool
    conviction_k: float
    conviction_mid: float
    conviction_score_lo: float
    conviction_score_hi: float
    conviction_floor: float
    base_risk_pct: float
    max_kelly_risk_pct: float
    kelly_fraction: float
    kelly_p_lo: float
    kelly_p_span: float
    kelly_payoff_b: float
    atr_stop_mult: float
    min_stop_pct: float
    session_risk_mult: Mapping[str, float]
    session_position_cap: Mapping[str, float]
    gross_exposure_cap: float
    net_exposure_cap: float
    max_positions: int
    sector_cap: float
    corr_threshold: float
    corr_cluster_cap: float
    daily_new_risk_pct: float
    max_trades_per_day: int
    entry_offset: Mapping[str, float]
    max_chase: Mapping[str, float]
    max_spread_pct: Mapping[str, float]
    cancel_replace_seconds: Mapping[str, int]
    max_replaces: int
    scale_in_first_fraction: float
    scale_in_atr: float
    scale_in_window_minutes: float
    scale_in_sessions: tuple[str, ...]
    max_gap_atr: float
    tp1_r: float
    tp1_fraction: float
    tp2_r: float
    tp2_fraction: float
    scale_out_min_qty: int
    single_target_r: float
    trail_arm_r: float
    trail_atr_mult: float
    trail_min_fraction_of_r: float
    alpha_decay_hours: float
    alpha_decay_min_mfe_r: float
    max_hold_hours: float
    overnight_min_score: float
    flatten_before_close_minutes: float
    earnings_entry_blackout_before_hours: float
    earnings_entry_blackout_after_hours: float
    earnings_flatten_before_hours: float
    earnings_flatten_after_minutes: float
    macro_entry_blackout_before_minutes: float
    macro_entry_blackout_after_minutes: float
    macro_flatten_before_minutes: float
    macro_hold_min_score: float
    macro_tighten_r: float
    daily_loss_limit_pct: float
    max_drawdown_pct: float
    kill_env: str
    reset_drawdown_env: str
    manage_adopted_positions: bool
    order_id_prefix: str
    cred_base: float
    cred_per_extra_domain: float
    cred_institutional_bonus: float
    cred_confirmation_bonus: float
    cred_min: float
    cred_max: float
    institutional_source_types: tuple[str, ...]
    social_source_types: tuple[str, ...]
    state_path: str
    decision_log_path: str
    kill_file: str
    reset_file: str

    def __post_init__(self) -> None:
        _validate(self)


def _require_sessions(name: str, mapping: Mapping[str, Any], keys: tuple[str, ...]) -> None:
    missing = [key for key in keys if key not in mapping]
    extra = [key for key in mapping if key not in keys]
    if missing or extra:
        raise ConfigError(f"{name} must have keys {keys}, missing={missing} extra={extra}")


def _unit(name: str, value: float) -> None:
    if not 0 < float(value) <= 1:
        raise ConfigError(f"{name} must be in (0, 1], got {value}")


def _validate(cfg: Config) -> None:
    if cfg.mode not in ("log_only", "paper"):
        raise ConfigError("mode must be log_only or paper; live trading is hard-disabled")
    if not 300 <= int(cfg.poll_seconds) <= 900:
        raise ConfigError("poll_seconds must be between 300 and 900")
    if cfg.reference_equity <= 0:
        raise ConfigError("reference_equity must be positive")
    if cfg.freshness_minutes <= 0 or cfg.clock_skew_minutes < 0:
        raise ConfigError("freshness window must be positive")
    if cfg.conviction_score_lo != cfg.min_score:
        raise ConfigError("conviction_score_lo must equal min_score")
    if not cfg.conviction_score_lo < cfg.conviction_score_hi <= 100:
        raise ConfigError("conviction score bounds are out of order")
    if not 0 < cfg.conviction_floor < 1:
        raise ConfigError("conviction_floor must be in (0, 1)")
    if cfg.conviction_k <= 0:
        raise ConfigError("conviction_k must be positive")
    if cfg.base_risk_pct <= 0 or cfg.max_kelly_risk_pct <= 0:
        raise ConfigError("risk percents must be positive")
    if cfg.kelly_fraction <= 0 or cfg.kelly_payoff_b <= 0:
        raise ConfigError("kelly parameters must be positive")
    if not 0 < cfg.kelly_p_lo < 1 or cfg.kelly_p_span < 0:
        raise ConfigError("kelly probability parameters are out of range")
    if cfg.kelly_p_lo + cfg.kelly_p_span >= 1:
        raise ConfigError("kelly win probability must stay below 1")
    if cfg.atr_stop_mult <= 0 or cfg.min_stop_pct <= 0:
        raise ConfigError("stop parameters must be positive")
    _require_sessions("session_risk_mult", cfg.session_risk_mult, _SESSIONS)
    _require_sessions("session_position_cap", cfg.session_position_cap, _SESSIONS)
    for name in ("entry_offset", "max_chase", "max_spread_pct"):
        _require_sessions(name, getattr(cfg, name), _OPEN_SESSIONS)
    _require_sessions("cancel_replace_seconds", cfg.cancel_replace_seconds, _OPEN_SESSIONS)
    for session in _OPEN_SESSIONS:
        if cfg.session_risk_mult[session] < 0 or cfg.session_position_cap[session] < 0:
            raise ConfigError("session multipliers cannot be negative")
        if not cfg.entry_offset[session] < cfg.max_chase[session]:
            raise ConfigError(f"entry_offset must be inside max_chase for {session}")
        if int(cfg.cancel_replace_seconds[session]) <= 0:
            raise ConfigError("cancel/replace timeout must be positive")
    if cfg.session_risk_mult["closed"] != 0 or cfg.session_position_cap["closed"] != 0:
        raise ConfigError("closed session cannot take new risk")
    for name in (
        "gross_exposure_cap",
        "net_exposure_cap",
        "sector_cap",
        "corr_cluster_cap",
        "daily_new_risk_pct",
        "daily_loss_limit_pct",
        "max_drawdown_pct",
        "tp1_fraction",
        "tp2_fraction",
        "scale_in_first_fraction",
        "low_score_trail_pct",
    ):
        _unit(name, getattr(cfg, name))
    if not 0 <= cfg.hold_until_score < cfg.min_score:
        raise ConfigError("hold_until_score must be below the entry minimum")
    if not isinstance(cfg.time_exits_enabled, bool):
        raise ConfigError("time_exits_enabled must be a boolean")
    if cfg.tp1_fraction + cfg.tp2_fraction >= 1:
        raise ConfigError("take-profit fractions must leave a runner")
    if not cfg.tp1_r < cfg.tp2_r:
        raise ConfigError("tp1_r must be below tp2_r")
    if cfg.max_positions < 1 or cfg.max_trades_per_day < 1 or cfg.max_replaces < 0:
        raise ConfigError("position and trade caps must be positive")
    if not 0 < cfg.corr_threshold < 1:
        raise ConfigError("corr_threshold must be in (0, 1)")
    if cfg.scale_out_min_qty < 2:
        raise ConfigError("scale_out_min_qty must be at least 2")
    for session in cfg.scale_in_sessions:
        if session not in _OPEN_SESSIONS:
            raise ConfigError(f"unknown scale-in session {session}")
    if not cfg.order_id_prefix.isascii() or not cfg.order_id_prefix.replace("-", "").isalnum():
        raise ConfigError("order_id_prefix must be ascii alphanumeric")
    if cfg.cred_min <= 0 or cfg.cred_max > 1 or cfg.cred_min > cfg.cred_max:
        raise ConfigError("credibility bounds are out of range")
    if not cfg.institutional_source_types or not cfg.social_source_types:
        raise ConfigError("source type lists cannot be empty")
    overlap = set(cfg.institutional_source_types) & set(cfg.social_source_types)
    if overlap:
        raise ConfigError(f"source types cannot be both institutional and social: {sorted(overlap)}")


def _coerce(raw: dict[str, Any]) -> dict[str, Any]:
    data = dict(raw)
    data["scale_in_sessions"] = tuple(data["scale_in_sessions"])
    data["institutional_source_types"] = tuple(data["institutional_source_types"])
    data["social_source_types"] = tuple(data["social_source_types"])
    data["poll_seconds"] = int(data["poll_seconds"])
    data["max_positions"] = int(data["max_positions"])
    data["max_trades_per_day"] = int(data["max_trades_per_day"])
    data["max_replaces"] = int(data["max_replaces"])
    data["scale_out_min_qty"] = int(data["scale_out_min_qty"])
    data["cancel_replace_seconds"] = {
        key: int(value) for key, value in data["cancel_replace_seconds"].items()
    }
    return data


def config_fields() -> set[str]:
    return {item.name for item in fields(Config)}


def load_config(path: str | Path | None = None) -> Config:
    source = Path(path) if path is not None else _PATH
    try:
        raw = json.loads(source.read_text())
    except FileNotFoundError as exc:
        raise ConfigError(f"config not found: {source}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"config is not valid json: {source}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("config root must be an object")
    known = config_fields()
    unknown = sorted(set(raw) - known)
    missing = sorted(known - set(raw))
    if unknown:
        raise ConfigError(f"unknown config keys: {unknown}")
    if missing:
        raise ConfigError(f"missing config keys: {missing}")
    try:
        return Config(**_coerce(raw))
    except ConfigError:
        raise
    except (TypeError, ValueError) as exc:
        raise ConfigError(str(exc)) from exc


def default_config() -> Config:
    return load_config(_PATH)


def replace_config(cfg: Config, **changes: Any) -> Config:
    """Copy `cfg` with top-level keys replaced. Nested maps are replaced whole."""
    raw: dict[str, Any] = {}
    for item in fields(Config):
        value = getattr(cfg, item.name)
        if isinstance(value, time):
            value = value.isoformat()
        elif isinstance(value, tuple):
            value = list(value)
        raw[item.name] = value
    unknown = sorted(set(changes) - set(raw))
    if unknown:
        raise ConfigError(f"unknown config overrides: {unknown}")
    raw.update(changes)
    return load_config_from_mapping(raw)


def config_with(**changes: Any) -> Config:
    """Return the default config with top-level keys replaced.

    Nested maps are replaced whole. Pass the full map when changing one session.
    """
    raw = json.loads(_PATH.read_text())
    unknown = sorted(set(changes) - set(raw))
    if unknown:
        raise ConfigError(f"unknown config overrides: {unknown}")
    raw.update(changes)
    return load_config_from_mapping(raw)


def load_config_from_mapping(raw: Mapping[str, Any]) -> Config:
    known = config_fields()
    unknown = sorted(set(raw) - known)
    missing = sorted(known - set(raw))
    if unknown:
        raise ConfigError(f"unknown config keys: {unknown}")
    if missing:
        raise ConfigError(f"missing config keys: {missing}")
    return Config(**_coerce(dict(raw)))
