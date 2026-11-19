"""Compare current breakouts to the pre-change snapshot."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd

from research_vcp_golden import TICKERS, _load, _pack
from vcp.detector import detect_vcp_breakouts


def main() -> None:
    golden = json.loads(Path('artifacts/vcp_golden_breakouts.json').read_text())
    con = duckdb.connect(r'C:\Users\Or\Downloads\VSCode\ATS_trading\data\ohlcv.duckdb', read_only=True)
    bad = 0
    for ticker in TICKERS:
        frame = _load(con, ticker)
        got = _pack(detect_vcp_breakouts(frame, ticker=ticker))
        if got != golden[ticker]:
            bad += 1
            print('MISMATCH', ticker, 'golden', len(golden[ticker]), 'got', len(got))
        else:
            print('ok', ticker, len(got))
    con.close()
    if bad:
        raise SystemExit(f'{bad} mismatches')
    print('all match')


if __name__ == '__main__':
    main()
