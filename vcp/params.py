"""Ideal-path parameters for VCP detection — what should happen, not edge cases."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class VCPParams:
    """Knobs that map to Minervini VCP physics.

    Coils are peak → trough → recovery → next peak. The entry is a *low-volume*
    coil into the pivot, then a volume expansion on the breakout — not an ATR
    setup. Depths are linear percent pullbacks: (peak - trough) / peak.
    """

    lookback_bars: int = 60
    price_space: str = 'pct'  # 'pct' (linear %) | 'log'

    min_contractions: int = 2
    max_contractions: int = 6
    # Each contraction depth must be <= prior_depth * contraction_ratio.
    contraction_ratio: float = 0.80
    max_final_depth: float = 0.10  # last coil <= ~10%
    min_first_depth: float = 0.06  # first pullback should be meaningful

    # Fixed-% zigzag (visual swings without ATR).
    pivot_up_th: float = 0.035
    pivot_down_th: float = 0.035

    # Non-final coils should reclaim at least this fraction of the pullback.
    min_recovery_frac: float = 0.35

    # --- Volume dry-up (the actual "low vol" entry) ---
    volume_ma_window: int = 50
    # Final-coil average volume vs 50d MA — hard dry-up.
    vdu_max_ratio: float = 0.65
    # Soft: pullback volume should not rise much coil-to-coil.
    volume_rise_tol: float = 1.15
    # Right before breakout: require at least one very quiet bar in the final coil.
    vdu_quiet_bar_ratio: float = 0.45  # min bar in final coil <= 45% of vol MA
    # Breakout must expand vs the final-coil dry-up mean (not only vs 50d MA).
    breakout_vs_dryup_mult: float = 1.8
    breakout_rel_vol: float = 1.4  # also vs 50d MA

    require_trend_template: bool = True
    ma_fast: int = 50
    ma_mid: int = 150
    ma_slow: int = 200
    ma_slow_rising_bars: int = 22

    require_higher_lows: bool = True
    allow_single_bar_final: bool = True
    min_base_bars: int = 12
    max_bars_to_breakout: int = 20
    dedupe_pivot_tol: float = 0.005

    def to_dict(self) -> dict:
        return asdict(self)
