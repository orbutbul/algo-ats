"""VCP structure + breakout detection on daily OHLCV."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from vcp.params import VCPParams
from vcp.pivots import Pivot, detect_pivots
from vcp.vbt_compat import vbt


@dataclass(frozen=True)
class Contraction:
    peak: Pivot
    valley: Pivot
    depth: float  # log or pct depending on params
    mean_volume: float
    bars: int


@dataclass(frozen=True)
class VCPBreakout:
    ticker: str
    breakout_date: pd.Timestamp
    pivot_price: float
    stop_price: float
    n_contractions: int
    depths: tuple[float, ...]
    volume_dryup_ratios: tuple[float, ...]
    breakout_rel_vol: float
    footprint: str
    contractions: tuple[Contraction, ...] = field(repr=False)

    def to_dict(self) -> dict:
        return {
            'ticker': self.ticker,
            'breakout_date': str(pd.Timestamp(self.breakout_date).date()),
            'pivot_price': self.pivot_price,
            'stop_price': self.stop_price,
            'n_contractions': self.n_contractions,
            'depths': list(self.depths),
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
                    'volume_dryup_ratios',
                    'breakout_rel_vol',
                    'footprint',
                ]
            )
        return pd.DataFrame([b.to_dict() for b in self.breakouts])


def _depth(peak: float, valley: float, space: str) -> float:
    if peak <= 0 or valley <= 0 or valley > peak:
        return float('nan')
    if space == 'log':
        return float(np.log(peak) - np.log(valley))
    return float((peak - valley) / peak)


def _depth_to_pct(depth: float, space: str) -> float:
    """Convert internal depth to approximate percent for footprints."""
    if space == 'log':
        return float((np.exp(depth) - 1.0) * 100.0)
    return float(depth * 100.0)


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


def _trend_template_mask(close: pd.Series, high: pd.Series, low: pd.Series, params: VCPParams) -> pd.Series:
    """Minervini Stage-2 style filter via vbt moving averages."""
    ma_fast = vbt.MA.run(close, params.ma_fast).ma
    ma_mid = vbt.MA.run(close, params.ma_mid).ma
    ma_slow = vbt.MA.run(close, params.ma_slow).ma
    # Align to 1-D Series (OSS vectorbt may return MultiIndex columns).
    def _s(x: pd.Series | pd.DataFrame) -> pd.Series:
        if isinstance(x, pd.DataFrame):
            return x.iloc[:, 0]
        return x

    ma_fast, ma_mid, ma_slow = _s(ma_fast), _s(ma_mid), _s(ma_slow)
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


def _contractions_from_pivots(
    pivots: list[Pivot],
    volume: pd.Series,
    params: VCPParams,
) -> list[Contraction]:
    out: list[Contraction] = []
    i = 0
    while i + 1 < len(pivots):
        a, b = pivots[i], pivots[i + 1]
        if a.kind == 'high' and b.kind == 'low':
            depth = _depth(a.price, b.price, params.price_space)
            # Volume over the pullback window (peak bar → valley bar inclusive).
            seg = volume.iloc[a.bar : b.bar + 1]
            mean_vol = float(seg.mean()) if len(seg) else float('nan')
            out.append(
                Contraction(
                    peak=a,
                    valley=b,
                    depth=depth,
                    mean_volume=mean_vol,
                    bars=max(1, b.bar - a.bar),
                )
            )
            i += 2
        else:
            i += 1
    return out


def _maybe_add_sideways_final(
    contractions: list[Contraction],
    ohlcv: pd.DataFrame,
    vol_ma: pd.Series,
    params: VCPParams,
) -> list[Contraction]:
    """Optionally append a tiny final coil when the zigzag stopped one early.

    Only attaches when the zigzag final is still loose, the micro range is
    tighter, and volume on that micro coil is dry vs the volume MA.
    """
    if not params.allow_single_bar_final or not contractions:
        return contractions
    last = contractions[-1]
    max_final = (
        float(np.log(1.0 / (1.0 - params.max_final_depth)))
        if params.price_space == 'log'
        else params.max_final_depth
    )
    # Zigzag already gave a tight final coil — keep it.
    if last.depth <= max_final and last.bars <= 5:
        return contractions

    after = ohlcv.iloc[last.valley.bar + 1 :]
    if after.empty:
        return contractions
    window = after.tail(8)
    peak_price = float(window['High'].max())
    peak_bar_pos = int(window['High'].values.argmax())
    peak_iloc = ohlcv.index.get_loc(window.index[peak_bar_pos])
    if isinstance(peak_iloc, slice):
        peak_iloc = int(peak_iloc.start)
    else:
        peak_iloc = int(peak_iloc)
    if peak_iloc <= last.valley.bar:
        return contractions

    trail = ohlcv.iloc[peak_iloc : peak_iloc + 6]
    if trail.empty:
        return contractions
    valley_price = float(trail['Low'].min())
    valley_bar_pos = int(trail['Low'].values.argmin())
    valley_iloc = peak_iloc + valley_bar_pos
    depth = _depth(peak_price, valley_price, params.price_space)
    if not np.isfinite(depth) or depth <= 0 or depth > max_final or depth >= last.depth:
        return contractions

    mean_vol = float(ohlcv['Volume'].iloc[peak_iloc : valley_iloc + 1].mean())
    base = vol_ma.iloc[valley_iloc] if valley_iloc < len(vol_ma) else np.nan
    if not np.isfinite(base) or base <= 0 or mean_vol / base > params.vdu_max_ratio:
        return contractions

    micro = Contraction(
        peak=Pivot(index=ohlcv.index[peak_iloc], price=peak_price, kind='high', bar=peak_iloc),
        valley=Pivot(index=ohlcv.index[valley_iloc], price=valley_price, kind='low', bar=valley_iloc),
        depth=depth,
        mean_volume=mean_vol,
        bars=max(1, valley_iloc - peak_iloc),
    )
    return contractions + [micro]


def _sequence_ok(seq: list[Contraction], vol_ma: pd.Series, params: VCPParams) -> bool:
    if not (params.min_contractions <= len(seq) <= params.max_contractions):
        return False
    if any(not np.isfinite(c.depth) or c.depth <= 0 for c in seq):
        return False
    if params.price_space == 'log':
        min_first = float(np.log(1.0 / (1.0 - params.min_first_depth)))
    else:
        min_first = params.min_first_depth
    if seq[0].depth < min_first:
        return False

    for prev, cur in zip(seq, seq[1:]):
        if cur.depth > prev.depth * params.contraction_ratio:
            return False
        if params.require_higher_lows and cur.valley.price < prev.valley.price:
            return False
        # Soft volume path: don't allow big volume increases on pullbacks.
        if (
            np.isfinite(prev.mean_volume)
            and np.isfinite(cur.mean_volume)
            and prev.mean_volume > 0
            and cur.mean_volume > prev.mean_volume * params.volume_rise_tol
        ):
            return False

    final = seq[-1]
    if final.depth > (
        float(np.log(1.0 / (1.0 - params.max_final_depth)))
        if params.price_space == 'log'
        else params.max_final_depth
    ):
        return False
    if not params.allow_single_bar_final and final.bars < 2:
        return False

    # Hard volume dry-up on final coil vs rolling volume MA at the valley.
    base = vol_ma.iloc[final.valley.bar] if final.valley.bar < len(vol_ma) else np.nan
    if not np.isfinite(base) or base <= 0 or not np.isfinite(final.mean_volume):
        return False
    if final.mean_volume / base > params.vdu_max_ratio:
        return False
    return True


def _footprint(seq: list[Contraction], params: VCPParams) -> str:
    first_pct = _depth_to_pct(seq[0].depth, params.price_space)
    last_pct = _depth_to_pct(seq[-1].depth, params.price_space)
    weeks = max(1, round((seq[-1].valley.bar - seq[0].peak.bar) / 5))
    return f'{weeks}W {first_pct:.0f}/{last_pct:.1f} {len(seq)}T'


def _find_breakout(
    ohlcv: pd.DataFrame,
    seq: list[Contraction],
    vol_ma: pd.Series,
    trend_ok: pd.Series,
    params: VCPParams,
) -> tuple[pd.Timestamp, float] | None:
    pivot_price = seq[-1].peak.price
    start = seq[-1].valley.bar + 1
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
    """Scan daily OHLCV for VCP structures and their volume-confirmed breakouts.

    Works off the ideal path encoded in ``VCPParams``: progressive log-depth
    contractions, volume dry-up into the final coil, Stage-2 trend template,
    then a pivot break on elevated volume.
    """
    params = params or VCPParams()
    df = _normalize_ohlcv(ohlcv)
    if len(df) < max(params.lookback_bars, params.ma_slow + params.ma_slow_rising_bars):
        return VCPResult(ticker=ticker, params=params, breakouts=(), pivots=())

    pivots = detect_pivots(df['High'], df['Low'], params.pivot_up_th, params.pivot_down_th)
    vol_ma = df['Volume'].rolling(params.volume_ma_window, min_periods=max(5, params.volume_ma_window // 5)).mean()
    trend_ok = (
        _trend_template_mask(df['Close'], df['High'], df['Low'], params)
        if params.require_trend_template
        else pd.Series(True, index=df.index)
    )

    contractions = _contractions_from_pivots(pivots, df['Volume'], params)
    min_leg = (
        float(np.log(1.0 / (1.0 - params.min_contraction_depth)))
        if params.price_space == 'log'
        else params.min_contraction_depth
    )
    # Drop noise legs before sequencing so major coils can sit next to each other.
    contractions = [c for c in contractions if np.isfinite(c.depth) and c.depth >= min_leg]
    breakouts: list[VCPBreakout] = []
    used_pivot_bars: set[int] = set()

    # Evaluate every trailing sequence of 2..max contractions that fits in lookback.
    for end in range(params.min_contractions, len(contractions) + 1):
        for n in range(params.min_contractions, min(params.max_contractions, end) + 1):
            seq = list(contractions[end - n : end])
            span = seq[-1].valley.bar - seq[0].peak.bar
            # Base must fit the ~60d alpha window and last long enough to matter.
            if span > params.lookback_bars or span < params.min_base_bars:
                continue
            seq = _maybe_add_sideways_final(seq, df, vol_ma, params)
            if len(seq) > params.max_contractions:
                seq = seq[-params.max_contractions :]
            span = seq[-1].valley.bar - seq[0].peak.bar
            if span > params.lookback_bars or span < params.min_base_bars:
                continue
            if not _sequence_ok(seq, vol_ma, params):
                continue
            pivot_bar = seq[-1].peak.bar
            if pivot_bar in used_pivot_bars:
                continue
            # Dedupe near-identical pivots already taken.
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
                base = vol_ma.iloc[c.valley.bar]
                vdu.append(float(c.mean_volume / base) if np.isfinite(base) and base > 0 else float('nan'))
            breakouts.append(
                VCPBreakout(
                    ticker=ticker,
                    breakout_date=pd.Timestamp(bdate),
                    pivot_price=float(seq[-1].peak.price),
                    stop_price=float(seq[-1].valley.price),
                    n_contractions=len(seq),
                    depths=tuple(float(c.depth) for c in seq),
                    volume_dryup_ratios=tuple(vdu),
                    breakout_rel_vol=float(rel),
                    footprint=_footprint(seq, params),
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
    )
