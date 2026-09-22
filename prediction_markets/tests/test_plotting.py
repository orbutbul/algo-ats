"""
Unit tests for prediction_markets/plotting.py -- checks the chart is built
correctly (right number of series, cumulative depth math, no crashes on
edge cases), not what it looks like (that was eyeballed manually per the
dataviz skill's render-and-look step).
"""

import matplotlib
matplotlib.use('Agg')  # headless, no display needed for tests

from datetime import datetime, timezone

import pytest

from prediction_markets.models import OrderBook, OrderBookLevel
from prediction_markets.plotting import plot_order_book


def _book(bids, asks):
    return OrderBook(
        venue='test', market_id='TEST-MKT', ts=datetime.now(timezone.utc),
        bids=[OrderBookLevel(price=p, size=s, side='bid') for p, s in bids],
        asks=[OrderBookLevel(price=p, size=s, side='ask') for p, s in asks],
    )


def test_cumulative_depth_increases_away_from_spread():
    # best-first (as OrderBook.bids/.asks always are): highest bid first, lowest ask first
    book = _book(bids=[(0.50, 10), (0.48, 5), (0.45, 20)], asks=[(0.52, 8), (0.55, 12), (0.60, 30)])
    ax = plot_order_book(book)

    lines = {line.get_label(): line for line in ax.get_lines() if line.get_label() in ('Bid', 'Ask')}
    assert set(lines) == {'Bid', 'Ask'}

    bid_x, bid_y = lines['Bid'].get_data()
    # x ascending (price low -> high, toward the spread)
    assert list(bid_x) == [0.45, 0.48, 0.50]
    # cumulative size DECREASES toward the spread (best bid = smallest cumulative)
    assert list(bid_y) == [35, 15, 10]

    ask_x, ask_y = lines['Ask'].get_data()
    assert list(ask_x) == [0.52, 0.55, 0.60]
    # cumulative size INCREASES away from the spread
    assert list(ask_y) == [8, 20, 50]


def test_best_bid_ask_annotated():
    book = _book(bids=[(0.50, 10)], asks=[(0.52, 8)])
    ax = plot_order_book(book)
    texts = [t.get_text() for t in ax.texts]
    assert any('0.5000' in t for t in texts)   # best bid
    assert any('0.5200' in t for t in texts)   # best ask


def test_handles_one_sided_book_without_crashing():
    book = _book(bids=[], asks=[(0.5, 10), (0.6, 20)])
    ax = plot_order_book(book)
    labels = [line.get_label() for line in ax.get_lines() if line.get_label() in ('Bid', 'Ask')]
    assert labels == ['Ask']


def test_handles_empty_book_without_crashing():
    book = _book(bids=[], asks=[])
    ax = plot_order_book(book)   # must not raise
    assert ax.get_lines() == [] or all(l.get_label() not in ('Bid', 'Ask') for l in ax.get_lines())


def test_plots_into_given_axes(monkeypatch):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    book = _book(bids=[(0.5, 10)], asks=[(0.52, 8)])
    returned = plot_order_book(book, ax=ax)
    assert returned is ax
