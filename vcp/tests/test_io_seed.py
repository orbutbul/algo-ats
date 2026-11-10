"""yfinance column shapes must flatten into open/high/low/close/volume."""

from __future__ import annotations

import pandas as pd

from vcp.io import _frames_from_yfinance


def test_frames_from_yfinance_ticker_price_multiindex():
    idx = pd.date_range('2024-01-02', periods=3, freq='B')
    cols = pd.MultiIndex.from_product([['MSFT'], ['Open', 'High', 'Low', 'Close', 'Volume']], names=['Ticker', 'Price'])
    raw = pd.DataFrame(
        [
            [10, 11, 9, 10.5, 1000],
            [10.5, 12, 10, 11.5, 1100],
            [11.5, 13, 11, 12.5, 1200],
        ],
        index=idx,
        columns=cols,
    )
    frames = _frames_from_yfinance(raw, ['MSFT'])
    assert len(frames) == 1
    frame = frames[0]
    assert {'datetime', 'ticker', 'open', 'high', 'low', 'close', 'volume'} <= set(frame.columns)
    assert frame['ticker'].iloc[0] == 'MSFT'
    assert float(frame['close'].iloc[-1]) == 12.5


def test_frames_from_yfinance_price_ticker_multiindex():
    idx = pd.date_range('2024-01-02', periods=2, freq='B')
    cols = pd.MultiIndex.from_product(
        [['Open', 'High', 'Low', 'Close', 'Volume'], ['TEVA']],
        names=['Price', 'Ticker'],
    )
    raw = pd.DataFrame(
        [[1, 1.1, 0.9, 1.05, 500], [1.05, 1.2, 1.0, 1.1, 600]],
        index=idx,
        columns=cols,
    )
    frames = _frames_from_yfinance(raw, ['TEVA'])
    assert len(frames) == 1
    assert list(frames[0]['close']) == [1.05, 1.1]
