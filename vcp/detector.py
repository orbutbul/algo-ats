"""VCP structure + breakout detection on daily OHLCV.

Coils are built as peak → trough → recovery → next peak (contiguous structure),
not by deleting noise legs from a zigzag list and gluing strangers together.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from vcp.params import VCPParams
from vcp.pivots import Pivot, detect_pivots
from vcp.vbt_compat import vbt


@dataclass(frozen=True)
class Coil:
    """One VCP contraction: peak → trough, then recovery into the next peak."""

    peak: Pivot
    trough: Pivot
    recovery_peak: Pivot | None
    depth_log: float
    depth_pct: float
    depth_atr: float
    mean_volume: float
    bars: int
    recovery_frac: float  # 0..1+ of pullback reclaimed into recovery_peak

    @property
    def depth(self) -> float:
        return self.depth_log


# Back-compat alias used by older plot/tests naming.
Contraction = Coil


@dataclass(frozen=True)
class VCPBreakout:
    ticker: str
    breakout_date: pd.Timestamp
    pivot_price: float
    stop_price: float
    n_contractions: int
    depths: tuple[float, ...]
    depth_atrs: tuple[float, ...]
    volume_dryup_ratios: tuple[float, ...]
    breakout_rel_vol: float
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
            'depth_atrs': list(self.depth_atrs),
            'volume_dryup_ratios': list(self.volume_dryup_ratios),
            'breakout_rel_vol': self.breakout_rel_vol,
            'footprint': self.footprint,
        }


@dataclass(frozen=True)
class VCPResult:
    ticker: str
    params: VCPParams
    breakouts: tuple[VCPBreakout, ...]
    pivots: tuple[Pivot, ...] = field(repr=False)
    atr: pd.Series | None = field(default=None, repr=False)

    def to_frame(self) -> pd.DataFrame:
        if not self.breakouts:
            return pd.DataFrame(
                columns=[
                    'ticker',
                    'breakout_date',
                    'pivot_price',
                    'stop_price',
                    'n_contractions',
                    'depths',
                    'depth_atrs',
                    'volume_dryup_ratios',
                    'breakout_rel_vol',
                    'footprint',
                ]
            )
        return pd.DataFrame([b.to_dict() for b in self.breakouts])


def _series_1d(x: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(x, pd.DataFrame):
        return x.iloc[:, 0]
    return x


def _depth_log(peak: float, valley: float) -> float:
    if peak <= 0 or valley <= 0 or valley > peak:
        return float('nan')
    return float(np.log(peak) - np.log(valley))


def _depth_pct(peak: float, valley: float) -> float:
    if peak <= 0 or valley <= 0 or valley > peak:
        return float('nan')
    return float((peak - valley) / peak)


def _depth_to_pct_label(depth_log: float) -> float:
    if not np.isfinite(depth_log):
        return float('nan')
    return float((np.exp(depth_log) - 1.0) * 100.0)


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
    return out.dropna(subset=['High', 'Low', 'Close', 'Volume'])


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    atr = vbt.ATR.run(high, low, close, window=window).atr
    return _series_1d(atr)


def _trend_template_mask(close: pd.Series, high: pd.Series, low: pd.Series, params: VCPParams) -> pd.Series:
    """Minervini Stage-2 style filter via vbt moving averages."""
    ma_fast = _series_1d(vbt.MA.run(close, params.ma_fast).ma)
    ma_mid = _series_1d(vbt.MA.run(close, params.ma_mid).ma)
    ma_slow = _series_1d(vbt.MA.run(close, params.ma_slow).ma)
    rising = ma_slow > ma_slow.shift(params.ma_slow_rising_bars)
    year_low = low.rolling(252, min_periods=100).min()
    year_high = high.rolling(252, min_periods=100).max()
    above_low = close >= year_low * 1.30
    near_high = close >= year_high * 0.75
    return (
        (close > ma_fast)
        & (close > ma_mid)
        & (close > ma_slow)
        & (ma_fast > ma_mid)
        & (ma_mid > ma_slow)
        & rising
        & above_low
        & near_high
    )


def build_coils(
    pivots: list[Pivot],
    volume: pd.Series,
    atr: pd.Series,
) -> list[Coil]:
    """Walk H-L-H-L… as contiguous coils: peak → trough → recovery peak.

    Next coil starts at the recovery peak (shared vertex), so sequences are
    structurally adjacent on the chart — not re-glued after noise filtering.
    """
    coils: list[Coil] = []
    i = 0
    while i + 1 < len(pivots):
        a, b = pivots[i], pivots[i + 1]
        if not (a.kind == 'high' and b.kind == 'low'):
            i += 1
            continue

        recovery: Pivot | None = None
        if i + 2 < len(pivots) and pivots[i + 2].kind == 'high':
            recovery = pivots[i + 2]

        depth_log = _depth_log(a.price, b.price)
        depth_pct = _depth_pct(a.price, b.price)
        atr_v = float(atr.iloc[b.bar]) if b.bar < len(atr) and np.isfinite(atr.iloc[b.bar]) else float('nan')
        depth_atr = float((a.price - b.price) / atr_v) if atr_v and atr_v > 0 else float('nan')

        seg = volume.iloc[a.bar : b.bar + 1]
        mean_vol = float(seg.mean()) if len(seg) else float('nan')

        pull = a.price - b.price
        if recovery is not None and pull > 0:
            recovery_frac = float((recovery.price - b.price) / pull)
        else:
            recovery_frac = float('nan')

        coils.append(
            Coil(
                peak=a,
                trough=b,
                recovery_peak=recovery,
                depth_log=depth_log,
                depth_pct=depth_pct,
                depth_atr=depth_atr,
                mean_volume=mean_vol,
                bars=max(1, b.bar - a.bar),
                recovery_frac=recovery_frac,
            )
        )
        # Advance to the recovery peak so the next coil shares that vertex.
        i = i + 2 if recovery is not None else i + 2
    return coils


def _maybe_tighten_final_with_sideways(
    coils: list[Coil],
    ohlcv: pd.DataFrame,
    atr: pd.Series,
    vol_ma: pd.Series,
    params: VCPParams,
) -> list[Coil]:
    """If the last coil is still loose, allow a 1-bar/sideways micro coil after recovery."""
    if not params.allow_single_bar_final or not coils:
        return coils
    last = coils[-1]
    if last.recovery_peak is None:
        return coils
    if np.isfinite(last.depth_atr) and last.depth_atr <= params.max_final_atr_mult:
        return coils
    if last.depth_pct <= params.max_final_depth:
        return coils

    start = last.recovery_peak.bar
    window = ohlcv.iloc[start : start + 9]
    if len(window) < 2:
        return coils
    peak_price = float(window['High'].max())
    peak_pos = int(window['High'].values.argmax())
    peak_iloc = start + peak_pos
    trail = ohlcv.iloc[peak_iloc : peak_iloc + 6]
    if trail.empty:
        return coils
    valley_price = float(trail['Low'].min())
    valley_pos = int(trail['Low'].values.argmin())
    valley_iloc = peak_iloc + valley_pos

    depth_log = _depth_log(peak_price, valley_price)
    depth_pct = _depth_pct(peak_price, valley_price)
    atr_v = float(atr.iloc[valley_iloc]) if valley_iloc < len(atr) else float('nan')
    depth_atr = float((peak_price - valley_price) / atr_v) if atr_v and atr_v > 0 else float('nan')
    if not np.isfinite(depth_pct) or depth_pct <= 0 or depth_pct > params.max_final_depth:
        return coils
    if params.use_atr_depth_checks and np.isfinite(depth_atr) and depth_atr > params.max_final_atr_mult:
        return coils
    if depth_pct >= last.depth_pct:
        return coils

    mean_vol = float(ohlcv['Volume'].iloc[peak_iloc : valley_iloc + 1].mean())
    base = vol_ma.iloc[valley_iloc] if valley_iloc < len(vol_ma) else np.nan
    if not np.isfinite(base) or base <= 0 or mean_vol / base > params.vdu_max_ratio:
        return coils

    micro_peak = Pivot(index=ohlcv.index[peak_iloc], price=peak_price, kind='high', bar=peak_iloc)
    micro_trough = Pivot(index=ohlcv.index[valley_iloc], price=valley_price, kind='low', bar=valley_iloc)
    micro = Coil(
        peak=micro_peak,
        trough=micro_trough,
        recovery_peak=None,
        depth_log=depth_log,
        depth_pct=depth_pct,
        depth_atr=depth_atr,
        mean_volume=mean_vol,
        bars=max(1, valley_iloc - peak_iloc),
        recovery_frac=float('nan'),
    )
    return coils + [micro]


def _sequence_ok(seq: list[Coil], vol_ma: pd.Series, params: VCPParams) -> bool:
    if not (params.min_contractions <= len(seq) <= params.max_contractions):
        return False
    if any(not np.isfinite(c.depth_log) or c.depth_log <= 0 for c in seq):
        return False

    # Contiguity: each coil's peak must be the previous coil's recovery peak.
    # Exception: an optional sideways micro-coil may start at/after that recovery.
    for i, (prev, cur) in enumerate(zip(seq, seq[1:])):
        if prev.recovery_peak is None:
            return False
        shared = cur.peak.bar == prev.recovery_peak.bar and np.isclose(
            cur.peak.price, prev.recovery_peak.price, rtol=0, atol=1e-9
        )
        micro_final = i == len(seq) - 2 and cur.peak.bar >= prev.recovery_peak.bar
        if not (shared or micro_final):
            return False
        # Constructive recovery into the next peak (visual "base" behavior).
        if shared and np.isfinite(prev.recovery_frac) and prev.recovery_frac < params.min_recovery_frac:
            return False

    min_first_log = float(np.log(1.0 / (1.0 - params.min_first_depth)))
    if seq[0].depth_log < min_first_log:
        return False
    if params.use_atr_depth_checks and np.isfinite(seq[0].depth_atr):
        if seq[0].depth_atr < params.min_first_atr_mult:
            return False

    for prev, cur in zip(seq, seq[1:]):
        if cur.depth_log > prev.depth_log * params.contraction_ratio:
            return False
        if (
            params.use_atr_depth_checks
            and np.isfinite(prev.depth_atr)
            and np.isfinite(cur.depth_atr)
            and cur.depth_atr > prev.depth_atr * params.atr_contraction_ratio
        ):
            return False
        if params.require_higher_lows and cur.trough.price < prev.trough.price:
            return False
        if (
            np.isfinite(prev.mean_volume)
            and np.isfinite(cur.mean_volume)
            and prev.mean_volume > 0
            and cur.mean_volume > prev.mean_volume * params.volume_rise_tol
        ):
            return False

    final = seq[-1]
    if final.depth_pct > params.max_final_depth:
        return False
    if params.use_atr_depth_checks and np.isfinite(final.depth_atr):
        if final.depth_atr > params.max_final_atr_mult:
            return False
    if not params.allow_single_bar_final and final.bars < 2:
        return False

    base = vol_ma.iloc[final.trough.bar] if final.trough.bar < len(vol_ma) else np.nan
    if not np.isfinite(base) or base <= 0 or not np.isfinite(final.mean_volume):
        return False
    if final.mean_volume / base > params.vdu_max_ratio:
        return False
    return True


def _footprint(seq: list[Coil]) -> str:
    first_pct = _depth_to_pct_label(seq[0].depth_log)
    last_pct = _depth_to_pct_label(seq[-1].depth_log)
    weeks = max(1, round((seq[-1].trough.bar - seq[0].peak.bar) / 5))
    last_atr = seq[-1].depth_atr
    atr_bit = f' {last_atr:.1f}A' if np.isfinite(last_atr) else ''
    return f'{weeks}W {first_pct:.0f}/{last_pct:.1f} {len(seq)}T{atr_bit}'


def _find_breakout(
    ohlcv: pd.DataFrame,
    seq: list[Coil],
    vol_ma: pd.Series,
    trend_ok: pd.Series,
    params: VCPParams,
) -> tuple[pd.Timestamp, float] | None:
    # Pivot = high of the final coil (right-side resistance).
    pivot_price = seq[-1].peak.price
    start = seq[-1].trough.bar + 1
    end = min(len(ohlcv), start + params.max_bars_to_breakout)
    for i in range(start, end):
        close = float(ohlcv['Close'].iloc[i])
        vol = float(ohlcv['Volume'].iloc[i])
        base = float(vol_ma.iloc[i]) if np.isfinite(vol_ma.iloc[i]) else np.nan
        if close <= pivot_price:
            continue
        if not np.isfinite(base) or base <= 0:
            continue
        rel = vol / base
        if rel < params.breakout_rel_vol:
            continue
        if params.require_trend_template and not bool(trend_ok.iloc[i]):
            continue
        return ohlcv.index[i], rel
    return None


def detect_vcp_breakouts(
    ohlcv: pd.DataFrame,
    ticker: str = '',
    params: VCPParams | None = None,
) -> VCPResult:
    """Scan daily OHLCV for contiguous VCP coil sequences and breakouts."""
    params = params or VCPParams()
    df = _normalize_ohlcv(ohlcv)
    if len(df) < max(params.lookback_bars, params.ma_slow + params.ma_slow_rising_bars):
        return VCPResult(ticker=ticker, params=params, breakouts=(), pivots=(), atr=None)

    atr = compute_atr(df['High'], df['Low'], df['Close'], params.atr_window)
    pivots = detect_pivots(
        df['High'],
        df['Low'],
        up_th=params.pivot_up_th,
        down_th=params.pivot_down_th,
        atr=atr,
        atr_mult=params.pivot_atr_mult,
        threshold_mode=params.pivot_threshold_mode,
    )
    vol_ma = df['Volume'].rolling(params.volume_ma_window, min_periods=max(5, params.volume_ma_window // 5)).mean()
    trend_ok = (
        _trend_template_mask(df['Close'], df['High'], df['Low'], params)
        if params.require_trend_template
        else pd.Series(True, index=df.index)
    )

    coils = build_coils(pivots, df['Volume'], atr)
    breakouts: list[VCPBreakout] = []
    used_pivot_bars: set[int] = set()

    for end in range(params.min_contractions, len(coils) + 1):
        for n in range(params.min_contractions, min(params.max_contractions, end) + 1):
            seq = list(coils[end - n : end])
            # Only mark a VCP when every coil but the last has a recovery vertex
            # (the chain is structurally complete into the final trough).
            if any(c.recovery_peak is None for c in seq[:-1]):
                continue
            span = seq[-1].trough.bar - seq[0].peak.bar
            if span > params.lookback_bars or span < params.min_base_bars:
                continue
            seq = _maybe_tighten_final_with_sideways(seq, df, atr, vol_ma, params)
            if len(seq) > params.max_contractions:
                continue
            span = seq[-1].trough.bar - seq[0].peak.bar
            if span > params.lookback_bars or span < params.min_base_bars:
                continue
            if not _sequence_ok(seq, vol_ma, params):
                continue
            pivot_bar = seq[-1].peak.bar
            if pivot_bar in used_pivot_bars:
                continue
            if any(
                abs(seq[-1].peak.price - b.pivot_price) / b.pivot_price <= params.dedupe_pivot_tol
                for b in breakouts
            ):
                continue
            found = _find_breakout(df, seq, vol_ma, trend_ok, params)
            if found is None:
                continue
            bdate, rel = found
            vdu = []
            for c in seq:
                base = vol_ma.iloc[c.trough.bar]
                vdu.append(float(c.mean_volume / base) if np.isfinite(base) and base > 0 else float('nan'))
            breakouts.append(
                VCPBreakout(
                    ticker=ticker,
                    breakout_date=pd.Timestamp(bdate),
                    pivot_price=float(seq[-1].peak.price),
                    stop_price=float(seq[-1].trough.price),
                    n_contractions=len(seq),
                    depths=tuple(float(c.depth_log) for c in seq),
                    depth_atrs=tuple(float(c.depth_atr) for c in seq),
                    volume_dryup_ratios=tuple(vdu),
                    breakout_rel_vol=float(rel),
                    footprint=_footprint(seq),
                    contractions=tuple(seq),
                )
            )
            used_pivot_bars.add(pivot_bar)

    breakouts.sort(key=lambda b: b.breakout_date)
    return VCPResult(
        ticker=ticker,
        params=params,
        breakouts=tuple(breakouts),
        pivots=tuple(pivots),
        atr=atr,
    )
