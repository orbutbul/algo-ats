"""Ideal-path parameters for VCP detection — what should happen, not edge cases."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class VCPParams:
    """Knobs that map to Minervini VCP physics.

    Depths are measured in log-price space when ``price_space='log'`` so a 20%
    pullback is comparable across price levels. Volume dry-up and breakout
    expansion are required; the final contraction may be a single tight bar.
    """

    lookback_bars: int = 60
    price_space: str = 'log'  # 'log' | 'pct'

    min_contractions: int = 2
    max_contractions: int = 6
    # Each contraction depth must be <= prior_depth * contraction_ratio.
    # ~0.5 is the textbook half-rule; 0.7 leaves slack for real charts.
    contraction_ratio: float = 0.70
    max_final_depth: float = 0.08  # last coil <= ~8%
    min_first_depth: float = 0.08  # first pullback should be meaningful
    # Ignore sub-threshold zigzag legs when building sequences (noise).
    min_contraction_depth: float = 0.03

    # Zigzag reversal thresholds (fraction). Sized to major base swings so a
    # multi-week VCP is not shredded into noise legs; the final tight/sideways
    # coil is attached separately when needed.
    pivot_up_th: float = 0.035
    pivot_down_th: float = 0.035

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
    # Minimum peak→final-valley span. Classic VCPs are multi-week bases;
    # very short coils are usually noise relative to the ~60d window.
    min_base_bars: int = 12
    # Breakout must arrive within this many bars after the final valley.
    max_bars_to_breakout: int = 20
    # After a breakout, ignore new patterns that share the same final peak.
    dedupe_pivot_tol: float = 0.005

    def to_dict(self) -> dict:
        return asdict(self)
