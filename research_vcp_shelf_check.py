"""Show how far AQN and POET trade above the drawn shelf inside a contraction."""

import numpy as np

from vcp.detector import detect_vcp_breakouts
from vcp.io import load_daily_ohlcv
from vcp.viewer import VIEW_PARAMS

DUCK = r'C:\Users\Or\Downloads\VSCode\ATS_trading\data\ohlcv.duckdb'


def main() -> None:
    for ticker in ('AQN', 'POET', 'GRDN'):
        frame = load_daily_ohlcv(ticker, path=DUCK, start='2015-10-06')
        high = frame['High'].to_numpy()
        result = detect_vcp_breakouts(frame, ticker=ticker, params=VIEW_PARAMS)
        print(f'\n{ticker} breakouts {len(result.breakouts)}')
        for bo in result.breakouts:
            print(
                f'  {str(bo.breakout_date)[:10]} shelf {bo.shelf_price:.2f} {bo.footprint}'
            )
            for i, coil in enumerate(bo.contractions, start=1):
                lo = coil.peak.bar
                hi = coil.recovery_peak.bar if coil.recovery_peak is not None else coil.trough.bar
                window = high[lo : hi + 1]
                top = float(np.nanmax(window)) if window.size else float('nan')
                over = top / bo.shelf_price - 1.0
                ret = coil.recovery_peak.price / bo.shelf_price - 1.0 if coil.recovery_peak else float('nan')
                print(
                    f'    T{i} {str(coil.peak.index)[:10]} -> {str(coil.trough.index)[:10]}'
                    f' -> {str(coil.recovery_peak.index)[:10] if coil.recovery_peak else "-"}'
                    f'  return {ret:+.1%}  max high in leg {over:+.1%} ({top:.2f})'
                )


if __name__ == '__main__':
    main()
