"""VCP structure + breakout detection on daily OHLCV.

Coils are peak → trough → recovery → next peak. The entry signal is a
*volume* dry-up into the final coil, then expansion on the pivot break.
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
    """One VCP contraction: peak → trough, then recovery into the next peak."""

    peak: Pivot
    trough: Pivot
    recovery_peak: Pivot | None
    depth_log: float
    depth_pct: float  # linear: (peak - trough) / peak
    mean_volume: float
    min_volume: float  # quietest bar in the pullback (dry-up tell)
    bars: int
    recovery_frac: float

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
    contractions: tuple[Coil, ...] = field(repr=False)

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


def _price_series(df: pd.DataFrame, source: str) -> pd.Series:
    if source == 'close':
        return df['Close']
    return df['HLC3']


def _rolling_mean(values: np.ndarray, window: int, min_periods: int | None = None) -> np.ndarray:
    """Positional rolling mean. Matches pandas on NaN-free input."""
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
    """Positional rolling min or max. Matches pandas on NaN-free input."""
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
    """Minervini trend template as a boolean array aligned to ``close``."""
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


def build_coils(pivots: list[Pivot], volume: pd.Series) -> list[Coil]:
    """Walk H-L-H-L… as contiguous coils: peak → trough → recovery peak."""
    vol = np.ascontiguousarray(volume.to_numpy(dtype=np.float64, copy=False))
    # Skipna, same as Series.mean / Series.min, only when the series actually has gaps.
    if np.isnan(vol).any():
        mean_of = np.nanmean
        min_of = np.nanmin
    else:
        mean_of = np.mean
        min_of = np.min
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
        if seg.size == 0:
            mean_vol = float('nan')
            min_vol = float('nan')
        else:
            mean_vol = float(mean_of(seg))
            min_vol = float(min_of(seg))
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
            )
        )
        i += 2
    return coils


def _depth_bounds(params: VCPParams) -> tuple[float, float]:
    if params.price_space == 'log':
        min_first = float(np.log(1.0 / (1.0 - params.min_first_depth)))
        max_final = float(np.log(1.0 / (1.0 - params.max_final_depth)))
    else:
        min_first = params.min_first_depth
        max_final = params.max_final_depth
    return min_first, max_final


def _micro_coil(
    last: Coil,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    vol_ma: np.ndarray,
    index: pd.Index,
    params: VCPParams,
    max_final: float,
) -> Coil | None:
    """Quiet sideways final coil after recovery, or None when the original would not append one.

    Depends only on the last coil, so each coil is resolved once and reused for every window
    that ends on it.
    """
    if not params.allow_single_bar_final or last.recovery_peak is None:
        return None
    last_depth = last.depth_for(params.price_space)
    if last_depth <= max_final:
        return None

    n = high.size
    start = last.recovery_peak.bar
    end = min(n, start + 9)
    if end - start < 2:
        return None
    window_high = high[start:end]
    peak_off = int(window_high.argmax())
    peak_price = float(window_high[peak_off])
    peak_iloc = start + peak_off
    trail_end = min(n, peak_iloc + 6)
    if trail_end <= peak_iloc:
        return None
    trail_low = low[peak_iloc:trail_end]
    valley_off = int(trail_low.argmin())
    valley_price = float(trail_low[valley_off])
    valley_iloc = peak_iloc + valley_off
    depth_pct = _depth_pct(peak_price, valley_price)
    depth_log = _depth_log(peak_price, valley_price)
    depth_use = depth_log if params.price_space == 'log' else depth_pct
    if not math.isfinite(depth_use) or depth_use <= 0.0 or depth_use > max_final:
        return None
    if depth_use >= last_depth:
        return None

    seg = volume[peak_iloc : valley_iloc + 1]
    if seg.size == 0:
        return None
    mean_vol = float(seg.mean())
    min_vol = float(seg.min())
    base = vol_ma[valley_iloc] if valley_iloc < vol_ma.size else math.nan
    if not math.isfinite(base) or base <= 0.0 or mean_vol / base > params.vdu_max_ratio:
        return None
    return Coil(
        peak=Pivot(index=index[peak_iloc], price=peak_price, kind='high', bar=peak_iloc),
        trough=Pivot(index=index[valley_iloc], price=valley_price, kind='low', bar=valley_iloc),
        recovery_peak=None,
        depth_log=depth_log,
        depth_pct=depth_pct,
        mean_volume=mean_vol,
        min_volume=min_vol,
        bars=max(1, valley_iloc - peak_iloc),
        recovery_frac=float('nan'),
    )


def _link_ok(
    has_rec: bool,
    rec_bar: int,
    rec_px: float,
    rec_frac: float,
    next_bar: int,
    next_px: float,
    is_last: bool,
    min_recovery: float,
) -> bool:
    """Same accept rule as the old peak-sharing / micro-final link check."""
    if not has_rec:
        return False
    # np.isclose(..., rtol=0, atol=1e-9) on finite prices.
    shared = next_bar == rec_bar and abs(next_px - rec_px) <= 1e-9
    if not shared and not (is_last and next_bar >= rec_bar):
        return False
    if shared and math.isfinite(rec_frac) and rec_frac < min_recovery:
        return False
    return True


def _volume_ok(
    means: list[float],
    mins: list[float],
    trough_bars: list[int],
    vol_ma: np.ndarray,
    params: VCPParams,
) -> bool:
    final = len(means) - 1
    tb = trough_bars[final]
    base = vol_ma[tb] if 0 <= tb < vol_ma.size else math.nan
    if not math.isfinite(base) or base <= 0.0:
        return False
    mean_final = means[final]
    min_final = mins[final]
    if not math.isfinite(mean_final) or mean_final / base > params.vdu_max_ratio:
        return False
    if not math.isfinite(min_final) or min_final / base > params.vdu_quiet_bar_ratio:
        return False
    tol = params.volume_rise_tol
    for i in range(final):
        prev = means[i]
        cur = means[i + 1]
        if math.isfinite(prev) and math.isfinite(cur) and prev > 0.0 and cur > prev * tol:
            return False
    return True


def _window_ok(
    depth: np.ndarray,
    mean_v: np.ndarray,
    min_v: np.ndarray,
    bars: np.ndarray,
    peak_bar: np.ndarray,
    trough_bar: np.ndarray,
    peak_px: np.ndarray,
    trough_px: np.ndarray,
    has_rec: np.ndarray,
    rec_bar: np.ndarray,
    rec_px: np.ndarray,
    rec_frac: np.ndarray,
    s: int,
    e: int,
    tail: Coil | None,
    vol_ma: np.ndarray,
    params: VCPParams,
    min_first: float,
    max_final: float,
) -> bool:
    """Coils ``[s, e)`` plus an optional micro-coil must be a VCP sequence."""
    n_raw = e - s
    n = n_raw + (1 if tail is not None else 0)
    if n < params.min_contractions or n > params.max_contractions or n_raw < 1:
        return False

    for i in range(s, e):
        d = depth[i]
        if not math.isfinite(d) or d <= 0.0:
            return False
    tail_depth = 0.0
    if tail is not None:
        tail_depth = tail.depth_log if params.price_space == 'log' else tail.depth_pct
        if not math.isfinite(tail_depth) or tail_depth <= 0.0:
            return False

    min_recovery = params.min_recovery_frac
    if tail is None:
        last_pair = e - 2
        for i in range(s, e - 1):
            if not _link_ok(
                bool(has_rec[i]),
                int(rec_bar[i]),
                float(rec_px[i]),
                float(rec_frac[i]),
                int(peak_bar[i + 1]),
                float(peak_px[i + 1]),
                i == last_pair,
                min_recovery,
            ):
                return False
    else:
        for i in range(s, e - 1):
            if not _link_ok(
                bool(has_rec[i]),
                int(rec_bar[i]),
                float(rec_px[i]),
                float(rec_frac[i]),
                int(peak_bar[i + 1]),
                float(peak_px[i + 1]),
                False,
                min_recovery,
            ):
                return False
        if not _link_ok(
            bool(has_rec[e - 1]),
            int(rec_bar[e - 1]),
            float(rec_px[e - 1]),
            float(rec_frac[e - 1]),
            tail.peak.bar,
            tail.peak.price,
            True,
            min_recovery,
        ):
            return False

    if depth[s] < min_first:
        return False
    ratio = params.contraction_ratio
    for i in range(s, e - 1):
        if depth[i + 1] > depth[i] * ratio:
            return False
    if tail is not None and tail_depth > depth[e - 1] * ratio:
        return False
    if params.require_higher_lows:
        for i in range(s, e - 1):
            if trough_px[i + 1] < trough_px[i]:
                return False
        if tail is not None and tail.trough.price < trough_px[e - 1]:
            return False

    final_depth = tail_depth if tail is not None else depth[e - 1]
    if final_depth > max_final:
        return False
    final_bars = tail.bars if tail is not None else int(bars[e - 1])
    if not params.allow_single_bar_final and final_bars < 2:
        return False

    means = [float(mean_v[i]) for i in range(s, e)]
    mins = [float(min_v[i]) for i in range(s, e)]
    troughs = [int(trough_bar[i]) for i in range(s, e)]
    if tail is not None:
        means.append(tail.mean_volume)
        mins.append(tail.min_volume)
        troughs.append(tail.trough.bar)
    return _volume_ok(means, mins, troughs, vol_ma, params)


def _depth_label_pct(coil: Coil, space: str) -> float:
    """Footprint label always in familiar percent terms."""
    if space == 'log':
        return float((np.exp(coil.depth_log) - 1.0) * 100.0)
    return float(coil.depth_pct * 100.0)


def _footprint(seq: list[Coil], vol_ma: np.ndarray, space: str = 'log') -> str:
    first_pct = _depth_label_pct(seq[0], space)
    last_pct = _depth_label_pct(seq[-1], space)
    weeks = max(1, round((seq[-1].trough.bar - seq[0].peak.bar) / 5))
    tb = seq[-1].trough.bar
    base = vol_ma[tb] if 0 <= tb < vol_ma.size else math.nan
    vdu = seq[-1].mean_volume / base if math.isfinite(base) and base > 0 else float('nan')
    vdu_bit = f' V{vdu:.0%}' if math.isfinite(vdu) else ''
    return f'{weeks}W {first_pct:.0f}/{last_pct:.1f} {len(seq)}T{vdu_bit}'


def _find_breakout(
    close: np.ndarray,
    volume: np.ndarray,
    vol_ma: np.ndarray,
    trend: np.ndarray | None,
    index: pd.Index,
    seq: list[Coil],
    params: VCPParams,
) -> tuple[pd.Timestamp, float, float] | None:
    """Breakout = close above final-coil peak with volume expansion vs dry-up."""
    pivot_price = seq[-1].peak.price
    dryup_mean = seq[-1].mean_volume
    start = seq[-1].trough.bar + 1
    end = min(close.size, start + params.max_bars_to_breakout)
    need_trend = params.require_trend_template
    for i in range(start, end):
        px = close[i]
        if px <= pivot_price:
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
        return index[i], rel, vs_dryup
    return None


def detect_vcp_breakouts(
    ohlcv: pd.DataFrame,
    ticker: str = '',
    params: VCPParams | None = None,
) -> VCPResult:
    """Scan daily OHLCV for contiguous VCP coils with volume dry-up breakouts."""
    params = params or VCPParams()
    df = _normalize_ohlcv(ohlcv)
    if len(df) < max(params.lookback_bars, params.ma_slow + params.ma_slow_rising_bars):
        return VCPResult(ticker=ticker, params=params, breakouts=(), pivots=())

    # Swings + depths on HLC3 (or Close); breakout still requires Close through pivot.
    px = _price_series(df, params.price_source)
    pivots = detect_pivots(px, px, params.pivot_up_th, params.pivot_down_th)
    high = np.ascontiguousarray(df['High'].to_numpy(dtype=np.float64, copy=False))
    low = np.ascontiguousarray(df['Low'].to_numpy(dtype=np.float64, copy=False))
    close = np.ascontiguousarray(df['Close'].to_numpy(dtype=np.float64, copy=False))
    volume = np.ascontiguousarray(df['Volume'].to_numpy(dtype=np.float64, copy=False))
    vol_ma = _rolling_mean(
        volume,
        params.volume_ma_window,
        min_periods=max(5, params.volume_ma_window // 5),
    )
    trend = _trend_template_mask(close, high, low, params) if params.require_trend_template else None

    coils = build_coils(pivots, df['Volume'])
    n_coils = len(coils)
    space = params.price_space
    depth = np.empty(n_coils, dtype=np.float64)
    mean_v = np.empty(n_coils, dtype=np.float64)
    min_v = np.empty(n_coils, dtype=np.float64)
    bars = np.empty(n_coils, dtype=np.int64)
    peak_bar = np.empty(n_coils, dtype=np.int64)
    trough_bar = np.empty(n_coils, dtype=np.int64)
    peak_px = np.empty(n_coils, dtype=np.float64)
    trough_px = np.empty(n_coils, dtype=np.float64)
    has_rec = np.zeros(n_coils, dtype=np.bool_)
    rec_bar = np.full(n_coils, -1, dtype=np.int64)
    rec_px = np.full(n_coils, np.nan, dtype=np.float64)
    rec_frac = np.full(n_coils, np.nan, dtype=np.float64)
    for i, coil in enumerate(coils):
        depth[i] = coil.depth_log if space == 'log' else coil.depth_pct
        mean_v[i] = coil.mean_volume
        min_v[i] = coil.min_volume
        bars[i] = coil.bars
        peak_bar[i] = coil.peak.bar
        trough_bar[i] = coil.trough.bar
        peak_px[i] = coil.peak.price
        trough_px[i] = coil.trough.price
        recovery = coil.recovery_peak
        if recovery is not None:
            has_rec[i] = True
            rec_bar[i] = recovery.bar
            rec_px[i] = recovery.price
            rec_frac[i] = coil.recovery_frac

    min_first, max_final = _depth_bounds(params)
    index = df.index
    micros: list[Coil | None] = [
        _micro_coil(coil, high, low, volume, vol_ma, index, params, max_final) for coil in coils
    ]
    breakouts: list[VCPBreakout] = []
    used_pivot_bars: set[int] = set()
    lookback = params.lookback_bars
    min_base = params.min_base_bars
    min_n = params.min_contractions
    max_n = params.max_contractions

    for end in range(min_n, n_coils + 1):
        last = end - 1
        micro = micros[last]
        for n in range(min_n, min(max_n, end) + 1):
            s = end - n
            span = int(trough_bar[last] - peak_bar[s])
            if span > lookback or span < min_base:
                continue
            tail = micro
            if tail is not None:
                if n + 1 > max_n:
                    continue
                span = tail.trough.bar - int(peak_bar[s])
                if span > lookback or span < min_base:
                    continue
            elif depth[last] > max_final:
                # The old tightener leaves this window unchanged, and the final coil is still too deep.
                continue
            if not _window_ok(
                depth, mean_v, min_v, bars, peak_bar, trough_bar, peak_px, trough_px,
                has_rec, rec_bar, rec_px, rec_frac,
                s, end, tail, vol_ma, params, min_first, max_final,
            ):
                continue
            seq = coils[s:end] if tail is None else coils[s:end] + [tail]
            pivot_i = seq[-1].peak.bar
            if pivot_i in used_pivot_bars:
                continue
            pivot_price = seq[-1].peak.price
            if any(
                abs(pivot_price - b.pivot_price) / b.pivot_price <= params.dedupe_pivot_tol
                for b in breakouts
            ):
                continue
            found = _find_breakout(close, volume, vol_ma, trend, index, seq, params)
            if found is None:
                continue
            bdate, rel, vs_dryup = found
            vdu: list[float] = []
            quiet: list[float] = []
            for coil in seq:
                base = vol_ma[coil.trough.bar] if 0 <= coil.trough.bar < vol_ma.size else math.nan
                if math.isfinite(base) and base > 0.0:
                    vdu.append(float(coil.mean_volume / base))
                    quiet.append(float(coil.min_volume / base))
                else:
                    vdu.append(float('nan'))
                    quiet.append(float('nan'))
            breakouts.append(
                VCPBreakout(
                    ticker=ticker,
                    breakout_date=pd.Timestamp(bdate),
                    pivot_price=float(pivot_price),
                    stop_price=float(seq[-1].trough.price),
                    n_contractions=len(seq),
                    depths=tuple(float(c.depth_for(space)) for c in seq),
                    volume_dryup_ratios=tuple(vdu),
                    quiet_bar_ratios=tuple(quiet),
                    breakout_rel_vol=float(rel),
                    breakout_vs_dryup=float(vs_dryup),
                    footprint=_footprint(seq, vol_ma, space),
                    contractions=tuple(seq),
                )
            )
            used_pivot_bars.add(pivot_i)

    breakouts.sort(key=lambda b: b.breakout_date)
    return VCPResult(ticker=ticker, params=params, breakouts=tuple(breakouts), pivots=tuple(pivots))
