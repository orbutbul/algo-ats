"""VCP detection via resistance-shelf excursions on daily High/Low.

A contraction is: leave a shared high (shelf) → lowest low of that trip →
return to (or near) the same shelf. Wiggles that never reclaim the shelf are
noise inside one excursion. Tiny shelf-to-shelf dips are noise-scale T's and
are dropped, except an optional tight final micro-coil.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from vcp.params import VCPParams
from vcp.pivots import Pivot, detect_pivots


@dataclass(frozen=True)
class Coil:
    """One VCP contraction: shelf leave → excursion trough → shelf return."""

    peak: Pivot
    trough: Pivot
    recovery_peak: Pivot | None
    depth_log: float
    depth_pct: float  # (shelf - trough) / shelf
    mean_volume: float
    min_volume: float
    bars: int
    recovery_frac: float  # how far the return climbed back toward the shelf
    shelf_price: float = 0.0
    is_noise: bool = False
    is_micro: bool = False

    def depth_for(self, space: str) -> float:
        return self.depth_log if space == 'log' else self.depth_pct

    @property
    def depth(self) -> float:
        return self.depth_log


Contraction = Coil


@dataclass(frozen=True)
class VCPBreakout:
    ticker: str
    breakout_date: pd.Timestamp
    pivot_price: float
    stop_price: float
    n_contractions: int
    depths: tuple[float, ...]
    volume_dryup_ratios: tuple[float, ...]
    quiet_bar_ratios: tuple[float, ...]
    breakout_rel_vol: float
    breakout_vs_dryup: float
    footprint: str
    contractions: tuple[Coil, ...] = field(default=(), repr=False)
    confidence: float = 0.0
    shelf_price: float = 0.0

    def to_dict(self) -> dict:
        return {
            'ticker': self.ticker,
            'breakout_date': str(pd.Timestamp(self.breakout_date).date()),
            'pivot_price': self.pivot_price,
            'stop_price': self.stop_price,
            'n_contractions': self.n_contractions,
            'depths': list(self.depths),
            'volume_dryup_ratios': list(self.volume_dryup_ratios),
            'quiet_bar_ratios': list(self.quiet_bar_ratios),
            'breakout_rel_vol': self.breakout_rel_vol,
            'breakout_vs_dryup': self.breakout_vs_dryup,
            'footprint': self.footprint,
            'confidence': self.confidence,
            'shelf_price': self.shelf_price,
        }


@dataclass(frozen=True)
class VCPResult:
    ticker: str
    params: VCPParams
    breakouts: tuple[VCPBreakout, ...]
    pivots: tuple[Pivot, ...] = field(repr=False)

    def to_frame(self) -> pd.DataFrame:
        cols = [
            'ticker',
            'breakout_date',
            'pivot_price',
            'stop_price',
            'n_contractions',
            'depths',
            'volume_dryup_ratios',
            'quiet_bar_ratios',
            'breakout_rel_vol',
            'breakout_vs_dryup',
            'footprint',
            'confidence',
            'shelf_price',
        ]
        if not self.breakouts:
            return pd.DataFrame(columns=cols)
        return pd.DataFrame([b.to_dict() for b in self.breakouts])


def _depth_log(peak: float, valley: float) -> float:
    if peak <= 0 or valley <= 0 or valley > peak:
        return float('nan')
    return float(np.log(peak) - np.log(valley))


def _depth_pct(peak: float, valley: float) -> float:
    if peak <= 0 or valley <= 0 or valley > peak:
        return float('nan')
    return float((peak - valley) / peak)


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    need = ['open', 'high', 'low', 'close', 'volume']
    missing = [c for c in need if c not in cols]
    if missing:
        raise ValueError(f'OHLCV missing columns {missing}; got {list(df.columns)}')
    out = pd.DataFrame(
        {
            'Open': df[cols['open']].astype(float),
            'High': df[cols['high']].astype(float),
            'Low': df[cols['low']].astype(float),
            'Close': df[cols['close']].astype(float),
            'Volume': df[cols['volume']].astype(float),
        },
        index=pd.to_datetime(df.index),
    )
    out = out.sort_index()
    out = out[~out.index.duplicated(keep='last')]
    out = out.dropna(subset=['High', 'Low', 'Close', 'Volume'])
    out['HLC3'] = (out['High'] + out['Low'] + out['Close']) / 3.0
    return out


def _rolling_mean(values: np.ndarray, window: int, min_periods: int | None = None) -> np.ndarray:
    if min_periods is None:
        min_periods = window
    n = int(values.size)
    out = np.full(n, np.nan, dtype=np.float64)
    if window <= 0 or min_periods <= 0 or n < min_periods:
        return out
    cumulative = np.empty(n + 1, dtype=np.float64)
    cumulative[0] = 0.0
    np.cumsum(values, out=cumulative[1:])
    partial_end = min(n, window - 1)
    if min_periods - 1 < partial_end:
        idx = np.arange(min_periods - 1, partial_end)
        out[idx] = cumulative[idx + 1] / (idx + 1)
    if n >= window:
        out[window - 1 :] = (cumulative[window:] - cumulative[: n - window + 1]) / window
    return out


def _rolling_minmax(values: np.ndarray, window: int, min_periods: int, *, is_min: bool) -> np.ndarray:
    n = int(values.size)
    out = np.full(n, np.nan, dtype=np.float64)
    if window <= 0 or min_periods <= 0 or n < min_periods:
        return out
    reduce = np.min if is_min else np.max
    partial_end = min(n, window - 1)
    for i in range(min_periods - 1, partial_end):
        out[i] = reduce(values[: i + 1])
    if n >= window:
        windows = np.lib.stride_tricks.sliding_window_view(values, window)
        out[window - 1 :] = reduce(windows, axis=1)
    return out


def _trend_template_mask(close: np.ndarray, high: np.ndarray, low: np.ndarray, params: VCPParams) -> np.ndarray:
    ma_fast = _rolling_mean(close, params.ma_fast)
    ma_mid = _rolling_mean(close, params.ma_mid)
    ma_slow = _rolling_mean(close, params.ma_slow)
    rising = np.zeros(close.size, dtype=bool)
    shift = params.ma_slow_rising_bars
    if shift > 0 and close.size > shift:
        rising[shift:] = ma_slow[shift:] > ma_slow[:-shift]
    year_low = _rolling_minmax(low, 252, 100, is_min=True)
    year_high = _rolling_minmax(high, 252, 100, is_min=False)
    return (
        (close > ma_fast)
        & (close > ma_mid)
        & (close > ma_slow)
        & (ma_fast > ma_mid)
        & (ma_mid > ma_slow)
        & rising
        & (close >= year_low * 1.30)
        & (close >= year_high * 0.75)
    )


def _shelf_band(shelf: float, params: VCPParams) -> tuple[float, float]:
    lo = shelf * (1.0 - params.shelf_tol)
    hi = shelf * (1.0 + params.shelf_overshoot)
    return lo, hi


def _touch_regions(high: np.ndarray, start: int, end: int, band_lo: float, band_hi: float) -> list[tuple[int, int]]:
    """Inclusive [lo, hi] index spans where High is inside the shelf band."""
    regions: list[tuple[int, int]] = []
    i = start
    while i < end:
        if band_lo <= high[i] <= band_hi:
            j = i
            while j + 1 < end and band_lo <= high[j + 1] <= band_hi:
                j += 1
            regions.append((i, j))
            i = j + 1
        else:
            i += 1
    return regions


def _coil_from_excursion(
    *,
    leave_i: int,
    trough_i: int,
    return_i: int,
    shelf: float,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    index: pd.Index,
    is_noise: bool = False,
    is_micro: bool = False,
) -> Coil | None:
    trough_px = float(low[trough_i])
    leave_px = float(high[leave_i])
    return_px = float(high[return_i])
    depth_pct = _depth_pct(shelf, trough_px)
    depth_log = _depth_log(shelf, trough_px)
    if not math.isfinite(depth_pct) or depth_pct <= 0.0:
        return None
    seg = volume[leave_i : trough_i + 1]
    if seg.size == 0:
        mean_vol = float('nan')
        min_vol = float('nan')
    else:
        mean_vol = float(np.mean(seg))
        min_vol = float(np.min(seg))
    pull = shelf - trough_px
    recovery_frac = float((return_px - trough_px) / pull) if pull > 0 else float('nan')
    return Coil(
        peak=Pivot(index=index[leave_i], price=leave_px, kind='high', bar=leave_i),
        trough=Pivot(index=index[trough_i], price=trough_px, kind='low', bar=trough_i),
        recovery_peak=Pivot(index=index[return_i], price=return_px, kind='high', bar=return_i),
        depth_log=depth_log,
        depth_pct=depth_pct,
        mean_volume=mean_vol,
        min_volume=min_vol,
        bars=max(1, trough_i - leave_i),
        recovery_frac=recovery_frac,
        shelf_price=float(shelf),
        is_noise=is_noise,
        is_micro=is_micro,
    )


def build_excursions_under_shelf(
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    index: pd.Index,
    start: int,
    shelf: float,
    params: VCPParams,
) -> list[Coil]:
    """Shelf contacts → excursion coils (trough = lowest low between contacts)."""
    n = high.size
    end = min(n, start + params.lookback_bars + params.max_bars_to_breakout + 5)
    band_lo, band_hi = _shelf_band(shelf, params)
    # Left peak must itself be a shelf contact.
    if not (band_lo <= high[start] <= band_hi or high[start] >= band_lo):
        # Allow the anchor high slightly above band_hi by using it as shelf.
        if high[start] < band_lo:
            return []
    regions = _touch_regions(high, start, end, band_lo, band_hi)
    if not regions:
        # Force the anchor bar as the first touch if High is the shelf itself.
        if high[start] >= band_lo:
            regions = [(start, start)]
            # Continue scanning after the anchor for later touches.
            regions.extend(_touch_regions(high, start + 1, end, band_lo, band_hi))
        else:
            return []

    coils: list[Coil] = []
    for r0, r1 in zip(regions, regions[1:]):
        leave_seg = high[r0[0] : r0[1] + 1]
        leave_off = int(np.argmax(leave_seg))
        leave_i = r0[0] + leave_off
        ret_seg = high[r1[0] : r1[1] + 1]
        ret_off = int(np.argmax(ret_seg))
        return_i = r1[0] + ret_off
        gap_lo = leave_i + 1
        gap_hi = return_i
        if gap_hi <= gap_lo:
            continue
        trough_off = int(np.argmin(low[gap_lo:gap_hi]))
        trough_i = gap_lo + trough_off
        if trough_i - leave_i < params.min_excursion_bars and (return_i - leave_i) < params.min_excursion_bars:
            continue
        coil = _coil_from_excursion(
            leave_i=leave_i,
            trough_i=trough_i,
            return_i=return_i,
            shelf=shelf,
            high=high,
            low=low,
            volume=volume,
            index=index,
        )
        if coil is not None:
            coils.append(coil)
    return coils


def classify_base_excursions(coils: list[Coil], params: VCPParams) -> list[Coil]:
    """Drop noise-scale shelf dips; optionally keep a final micro as last T."""
    if not coils:
        return []
    first = coils[0].depth_pct
    if not math.isfinite(first) or first <= 0:
        return []
    noise_cut = first * params.noise_depth_frac
    base: list[Coil] = []
    pending_micro: Coil | None = None
    for coil in coils:
        if coil.depth_pct >= noise_cut:
            if pending_micro is not None:
                # A real base T after a tiny dip — tiny dip was mid-base noise; drop it.
                pending_micro = None
            base.append(coil)
        else:
            pending_micro = coil
    if (
        params.allow_single_bar_final
        and pending_micro is not None
        and pending_micro.depth_pct <= params.max_final_depth
        and pending_micro.depth_pct > 0
    ):
        base.append(
            Coil(
                peak=pending_micro.peak,
                trough=pending_micro.trough,
                recovery_peak=pending_micro.recovery_peak,
                depth_log=pending_micro.depth_log,
                depth_pct=pending_micro.depth_pct,
                mean_volume=pending_micro.mean_volume,
                min_volume=pending_micro.min_volume,
                bars=pending_micro.bars,
                recovery_frac=pending_micro.recovery_frac,
                shelf_price=pending_micro.shelf_price,
                is_noise=False,
                is_micro=True,
            )
        )
    return base


def build_coils(pivots: list[Pivot], volume: pd.Series) -> list[Coil]:
    """Legacy helper: H-L-H pairs as coils (tests / debugging). Prefer shelf path."""
    vol = np.ascontiguousarray(volume.to_numpy(dtype=np.float64, copy=False))
    mean_of = np.nanmean if np.isnan(vol).any() else np.mean
    min_of = np.nanmin if np.isnan(vol).any() else np.min
    coils: list[Coil] = []
    i = 0
    n_pivots = len(pivots)
    while i + 1 < n_pivots:
        a, b = pivots[i], pivots[i + 1]
        if not (a.kind == 'high' and b.kind == 'low'):
            i += 1
            continue
        recovery: Pivot | None = None
        if i + 2 < n_pivots and pivots[i + 2].kind == 'high':
            recovery = pivots[i + 2]
        seg = vol[a.bar : b.bar + 1]
        mean_vol = float(mean_of(seg)) if seg.size else float('nan')
        min_vol = float(min_of(seg)) if seg.size else float('nan')
        pull = a.price - b.price
        recovery_frac = (
            float((recovery.price - b.price) / pull) if recovery is not None and pull > 0 else float('nan')
        )
        coils.append(
            Coil(
                peak=a,
                trough=b,
                recovery_peak=recovery,
                depth_log=_depth_log(a.price, b.price),
                depth_pct=_depth_pct(a.price, b.price),
                mean_volume=mean_vol,
                min_volume=min_vol,
                bars=max(1, b.bar - a.bar),
                recovery_frac=recovery_frac,
                shelf_price=float(a.price),
            )
        )
        i += 2
    return coils


def _volume_ok(seq: list[Coil], vol_ma: np.ndarray, params: VCPParams) -> bool:
    final = seq[-1]
    tb = final.trough.bar
    base = vol_ma[tb] if 0 <= tb < vol_ma.size else math.nan
    if not math.isfinite(base) or base <= 0.0:
        return False
    if not math.isfinite(final.mean_volume) or final.mean_volume / base > params.vdu_max_ratio:
        return False
    if not math.isfinite(final.min_volume) or final.min_volume / base > params.vdu_quiet_bar_ratio:
        return False
    tol = params.volume_rise_tol
    for prev, cur in zip(seq, seq[1:]):
        if (
            math.isfinite(prev.mean_volume)
            and math.isfinite(cur.mean_volume)
            and prev.mean_volume > 0.0
            and cur.mean_volume > prev.mean_volume * tol
        ):
            return False
    return True


def _structure_ok(seq: list[Coil], params: VCPParams, min_first: float, max_final: float) -> bool:
    if not (params.min_contractions <= len(seq) <= params.max_contractions):
        return False
    d0 = seq[0].depth_for(params.price_space)
    if not math.isfinite(d0) or d0 < min_first:
        return False
    d_last = seq[-1].depth_for(params.price_space)
    if not math.isfinite(d_last) or d_last > max_final:
        return False
    if not params.allow_single_bar_final and seq[-1].bars < 2:
        return False
    ratio = params.contraction_ratio
    for prev, cur in zip(seq, seq[1:]):
        pd_ = prev.depth_for(params.price_space)
        cd = cur.depth_for(params.price_space)
        if not math.isfinite(pd_) or not math.isfinite(cd) or cd > pd_ * ratio:
            return False
    if params.require_higher_lows:
        tol = params.higher_low_tol
        for prev, cur in zip(seq, seq[1:]):
            if cur.trough.price < prev.trough.price * (1.0 - tol):
                return False
    return True


def _depth_bounds(params: VCPParams) -> tuple[float, float]:
    if params.price_space == 'log':
        min_first = float(np.log(1.0 / (1.0 - params.min_first_depth)))
        max_final = float(np.log(1.0 / (1.0 - params.max_final_depth)))
    else:
        min_first = params.min_first_depth
        max_final = params.max_final_depth
    return min_first, max_final


def _score_sequence(
    seq: list[Coil],
    shelf: float,
    breakout_rel: float,
    breakout_vs_dryup: float,
    vol_ma: np.ndarray,
    params: VCPParams,
) -> float:
    """0–1 cleanliness: shelf returns, rising lows, depth decay, dry-up, expansion."""
    if shelf <= 0 or not seq:
        return 0.0

    # Shelf return quality: how close each recovery high is to the shelf.
    shelf_bits: list[float] = []
    for c in seq:
        if c.recovery_peak is None:
            continue
        err = abs(c.recovery_peak.price - shelf) / shelf
        shelf_bits.append(float(np.clip(1.0 - err / max(params.shelf_tol, 1e-6), 0.0, 1.0)))
    shelf_score = float(np.mean(shelf_bits)) if shelf_bits else 0.0

    # Rising lows (magnitude-aware).
    low_bits: list[float] = []
    for prev, cur in zip(seq, seq[1:]):
        if prev.trough.price <= 0:
            continue
        lift = (cur.trough.price - prev.trough.price) / prev.trough.price
        low_bits.append(float(np.clip(0.5 + lift / 0.05, 0.0, 1.0)))
    low_score = float(np.mean(low_bits)) if low_bits else 0.5

    # Depth decay toward classic ~half-ish without requiring it.
    depths = [c.depth_pct for c in seq if math.isfinite(c.depth_pct) and c.depth_pct > 0]
    if len(depths) >= 2:
        ratios = [depths[i + 1] / depths[i] for i in range(len(depths) - 1)]
        # Ideal around 0.5; acceptable to ~0.8.
        decay_score = float(np.mean([np.clip(1.0 - abs(r - 0.5) / 0.5, 0.0, 1.0) for r in ratios]))
    else:
        decay_score = 0.0

    tb = seq[-1].trough.bar
    base = vol_ma[tb] if 0 <= tb < vol_ma.size else math.nan
    if math.isfinite(base) and base > 0 and math.isfinite(seq[-1].mean_volume):
        vdu = seq[-1].mean_volume / base
        vdu_score = float(np.clip((params.vdu_max_ratio - vdu) / params.vdu_max_ratio + 0.5, 0.0, 1.0))
    else:
        vdu_score = 0.0

    exp_score = float(
        np.clip(
            0.5 * (breakout_rel / max(params.breakout_rel_vol, 1e-6))
            + 0.5 * (breakout_vs_dryup / max(params.breakout_vs_dryup_mult, 1e-6)),
            0.0,
            1.0,
        )
    )
    # Prefer 3–4 contractions slightly.
    n = len(seq)
    n_score = 1.0 if n in (3, 4) else (0.75 if n == 2 else 0.85)

    conf = (
        0.22 * shelf_score
        + 0.18 * low_score
        + 0.20 * decay_score
        + 0.20 * vdu_score
        + 0.15 * exp_score
        + 0.05 * n_score
    )
    return float(np.clip(conf, 0.0, 1.0))


def _depth_label_pct(coil: Coil, space: str) -> float:
    # log-depth d = ln(peak/trough) → percent pullback = 1 - e^{-d}
    if space == 'log':
        return float((1.0 - np.exp(-coil.depth_log)) * 100.0)
    return float(coil.depth_pct * 100.0)


def _footprint(seq: list[Coil], vol_ma: np.ndarray, space: str, confidence: float) -> str:
    first_pct = _depth_label_pct(seq[0], space)
    last_pct = _depth_label_pct(seq[-1], space)
    weeks = max(1, round((seq[-1].trough.bar - seq[0].peak.bar) / 5))
    tb = seq[-1].trough.bar
    base = vol_ma[tb] if 0 <= tb < vol_ma.size else math.nan
    vdu = seq[-1].mean_volume / base if math.isfinite(base) and base > 0 else float('nan')
    vdu_bit = f' V{vdu:.0%}' if math.isfinite(vdu) else ''
    return f'{weeks}W {first_pct:.0f}/{last_pct:.1f} {len(seq)}T{vdu_bit} C{confidence:.2f}'


def _find_breakout(
    close: np.ndarray,
    high: np.ndarray,
    volume: np.ndarray,
    vol_ma: np.ndarray,
    trend: np.ndarray | None,
    index: pd.Index,
    seq: list[Coil],
    shelf: float,
    params: VCPParams,
) -> tuple[pd.Timestamp, float, float] | None:
    """Breakout = close (and preferably high) through the shelf with volume expansion."""
    pivot_price = shelf
    dryup_mean = seq[-1].mean_volume
    start = seq[-1].trough.bar + 1
    # Prefer scanning after the last shelf return when present.
    if seq[-1].recovery_peak is not None:
        start = max(start, seq[-1].recovery_peak.bar)
    end = min(close.size, start + params.max_bars_to_breakout)
    need_trend = params.require_trend_template
    for i in range(start, end):
        if close[i] <= pivot_price:
            continue
        base = vol_ma[i]
        if not math.isfinite(base) or base <= 0.0 or not math.isfinite(dryup_mean) or dryup_mean <= 0.0:
            continue
        vol = volume[i]
        rel = vol / base
        vs_dryup = vol / dryup_mean
        if rel < params.breakout_rel_vol:
            continue
        if vs_dryup < params.breakout_vs_dryup_mult:
            continue
        if need_trend and not trend[i]:
            continue
        # Prefer true range expansion through the shelf.
        if high[i] < pivot_price:
            continue
        return index[i], rel, vs_dryup
    return None


def detect_vcp_breakouts(
    ohlcv: pd.DataFrame,
    ticker: str = '',
    params: VCPParams | None = None,
) -> VCPResult:
    """Scan daily OHLCV for shelf-excursion VCPs with volume dry-up breakouts."""
    params = params or VCPParams()
    df = _normalize_ohlcv(ohlcv)
    if len(df) < max(params.lookback_bars, params.ma_slow + params.ma_slow_rising_bars):
        return VCPResult(ticker=ticker, params=params, breakouts=(), pivots=())

    high = np.ascontiguousarray(df['High'].to_numpy(dtype=np.float64, copy=False))
    low = np.ascontiguousarray(df['Low'].to_numpy(dtype=np.float64, copy=False))
    close = np.ascontiguousarray(df['Close'].to_numpy(dtype=np.float64, copy=False))
    volume = np.ascontiguousarray(df['Volume'].to_numpy(dtype=np.float64, copy=False))
    index = df.index

    # Swing highs on High/Low seed candidate shelves (not HLC3).
    pivots = detect_pivots(df['High'], df['Low'], params.pivot_up_th, params.pivot_down_th)
    swing_highs = [p for p in pivots if p.kind == 'high']

    vol_ma = _rolling_mean(
        volume,
        params.volume_ma_window,
        min_periods=max(5, params.volume_ma_window // 5),
    )
    trend = _trend_template_mask(close, high, low, params) if params.require_trend_template else None
    min_first, max_final = _depth_bounds(params)

    breakouts: list[VCPBreakout] = []
    used_shelf_bars: set[int] = set()

    for anchor in swing_highs:
        shelf = float(anchor.price)
        if shelf <= 0:
            continue
        raw = build_excursions_under_shelf(high, low, volume, index, anchor.bar, shelf, params)
        base = classify_base_excursions(raw, params)
        if len(base) < params.min_contractions:
            continue

        # Try trailing windows of base-scale excursions ending at each position.
        for end in range(params.min_contractions, len(base) + 1):
            for n in range(params.min_contractions, min(params.max_contractions, end) + 1):
                seq = base[end - n : end]
                span = seq[-1].trough.bar - seq[0].peak.bar
                if span > params.lookback_bars or span < params.min_base_bars:
                    continue
                if not _structure_ok(seq, params, min_first, max_final):
                    continue
                if not _volume_ok(seq, vol_ma, params):
                    continue
                # Dedupe: one setup per left-peak / shelf family.
                left_bar = seq[0].peak.bar
                if left_bar in used_shelf_bars:
                    continue
                if any(
                    abs(shelf - b.shelf_price) / b.shelf_price <= params.dedupe_pivot_tol
                    and abs((seq[-1].peak.bar) - b.contractions[-1].peak.bar) <= 3
                    for b in breakouts
                ):
                    continue

                found = _find_breakout(close, high, volume, vol_ma, trend, index, seq, shelf, params)
                if found is None:
                    continue
                bdate, rel, vs_dryup = found
                conf = _score_sequence(seq, shelf, rel, vs_dryup, vol_ma, params)
                if conf < params.min_confidence:
                    continue

                vdu: list[float] = []
                quiet: list[float] = []
                for coil in seq:
                    vb = vol_ma[coil.trough.bar] if 0 <= coil.trough.bar < vol_ma.size else math.nan
                    if math.isfinite(vb) and vb > 0.0:
                        vdu.append(float(coil.mean_volume / vb))
                        quiet.append(float(coil.min_volume / vb))
                    else:
                        vdu.append(float('nan'))
                        quiet.append(float('nan'))

                breakouts.append(
                    VCPBreakout(
                        ticker=ticker,
                        breakout_date=pd.Timestamp(bdate),
                        pivot_price=float(shelf),
                        stop_price=float(seq[-1].trough.price),
                        n_contractions=len(seq),
                        depths=tuple(float(c.depth_for(params.price_space)) for c in seq),
                        volume_dryup_ratios=tuple(vdu),
                        quiet_bar_ratios=tuple(quiet),
                        breakout_rel_vol=float(rel),
                        breakout_vs_dryup=float(vs_dryup),
                        footprint=_footprint(seq, vol_ma, params.price_space, conf),
                        confidence=float(conf),
                        shelf_price=float(shelf),
                        contractions=tuple(seq),
                    )
                )
                used_shelf_bars.add(left_bar)

    breakouts.sort(key=lambda b: (b.breakout_date, -b.confidence))
    return VCPResult(ticker=ticker, params=params, breakouts=tuple(breakouts), pivots=tuple(pivots))
