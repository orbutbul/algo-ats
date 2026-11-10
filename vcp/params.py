"""Ideal-path parameters for VCP detection — what should happen, not edge cases."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class VCPParams:
    """Knobs that map to Minervini VCP physics.

    Coils are built as peak → trough → recovery → next peak (not a filtered
    zigzag soup). Depths default to log-price space; ATR is used for swing
    significance and visual tightness checks.
    """

    lookback_bars: int = 60
    price_space: str = 'log'  # 'log' | 'pct'

    min_contractions: int = 2
    max_contractions: int = 6
    # Each contraction depth must be <= prior_depth * contraction_ratio.
    # ~0.5 is the textbook half-rule; 0.7 leaves slack for real charts.
    contraction_ratio: float = 0.80
    max_final_depth: float = 0.10  # last coil <= ~10% (log/pct space)
    min_first_depth: float = 0.06  # first pullback should be meaningful

    # --- ATR (visual volatility unit) ---
    atr_window: int = 14
    # Swing confirmation: reverse by this many ATRs (preferred) or fixed pct.
    pivot_threshold_mode: str = 'atr'  # 'atr' | 'pct'
    pivot_atr_mult: float = 1.25
    pivot_up_th: float = 0.035  # used when pivot_threshold_mode='pct'
    pivot_down_th: float = 0.035
    # Also require ATR-space shrink / tightness (complements log % rules).
    use_atr_depth_checks: bool = True
    min_first_atr_mult: float = 1.5
    max_final_atr_mult: float = 1.75
    atr_contraction_ratio: float = 0.85
    # Non-final coils should reclaim at least this fraction of the pullback
    # before the next peak (constructive recovery).
    min_recovery_frac: float = 0.35

    volume_ma_window: int = 50
    # Final-coil average volume must be <= this fraction of the volume MA.
    vdu_max_ratio: float = 0.85
    # Soft: each successive contraction's mean volume should not rise a lot.
    volume_rise_tol: float = 1.25
    breakout_rel_vol: float = 1.3

    require_trend_template: bool = True
    ma_fast: int = 50
    ma_mid: int = 150
    ma_slow: int = 200
    # 200-day MA higher than this many bars ago (Minervini: rising ~1 month+).
    ma_slow_rising_bars: int = 22

    require_higher_lows: bool = True
    allow_single_bar_final: bool = True
    # Minimum peak→final-valley span. Classic VCPs are multi-week bases.
    min_base_bars: int = 12
    # Breakout must arrive within this many bars after the final valley.
    max_bars_to_breakout: int = 20
    # After a breakout, ignore new patterns that share the same final peak.
    dedupe_pivot_tol: float = 0.005

    def to_dict(self) -> dict:
        return asdict(self)
