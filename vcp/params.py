"""Ideal-path parameters for shelf / excursion VCP detection."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class VCPParams:
    """Knobs that map to Minervini VCP physics on daily High/Low bars.

    A contraction is an *excursion under a resistance shelf*: leave the shelf →
    make a low → return to (or near) the same high. Nested wiggles that never
    reclaim the shelf are noise inside one excursion. Smaller shelf-to-shelf
    dips are noise-scale; only base-scale excursions count as T's, with an
    optional tight final micro-coil.
    """

    lookback_bars: int = 150  # max base width (first peak → final trough)
    price_space: str = 'log'  # 'log' | 'pct' — depth comparison space

    min_contractions: int = 3
    max_contractions: int = 6
    # Soft depth shrink: each base T depth should be <= prior * ratio (scored hard-gate).
    contraction_ratio: float = 0.80
    max_final_depth: float = 0.10  # last coil <= ~10%
    min_first_depth: float = 0.06  # first pullback should be meaningful

    # Zigzag only seeds candidate shelf anchors (swing highs on High/Low).
    pivot_up_th: float = 0.035
    pivot_down_th: float = 0.035

    # --- Shelf / excursion structure ---
    # Return high is "around" the shelf if within this fraction of shelf price.
    shelf_tol: float = 0.025
    # Every high in a contraction, including the return, may print at most this
    # far above the shelf. A push beyond it breaks the level.
    shelf_overshoot: float = 0.01
    # Excursions shallower than this fraction of the *first* depth are noise-scale.
    noise_depth_frac: float = 0.33
    # Minimum bars from leave high → trough (filters 1-bar flicker).
    min_excursion_bars: int = 2
    # Higher-low rule: next trough >= prior * (1 - tol).
    higher_low_tol: float = 0.01
    # Minimum confidence to emit a breakout (0–1 composite score).
    min_confidence: float = 0.45

    # --- Volume dry-up (low-volume entry into the pivot) ---
    volume_ma_window: int = 50
    vdu_max_ratio: float = 0.70
    volume_rise_tol: float = 2.5
    vdu_quiet_bar_ratio: float = 0.55
    breakout_vs_dryup_mult: float = 1.6
    breakout_rel_vol: float = 1.3

    require_trend_template: bool = True
    ma_fast: int = 50
    ma_mid: int = 150
    ma_slow: int = 200
    ma_slow_rising_bars: int = 22

    require_higher_lows: bool = True
    allow_single_bar_final: bool = True
    min_base_bars: int = 30
    max_bars_to_breakout: int = 20
    dedupe_pivot_tol: float = 0.005

    # Legacy name kept so older CLI flags still map; unused by shelf model.
    min_recovery_frac: float = 0.80
    price_source: str = 'hlc3'

    def to_dict(self) -> dict:
        return asdict(self)
