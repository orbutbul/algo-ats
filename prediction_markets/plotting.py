"""
prediction_markets/plotting.py — order-book depth chart, standalone from any
dashboard infrastructure (same "ad hoc visualization" convention as
regime/plotting.py) so it can be called directly on a `models.OrderBook`
from a notebook or a REPL.

plot_order_book() draws the classic L2 depth chart (Robinhood/most retail
brokers' "Level II" view): cumulative size on the y-axis, price on the
x-axis, a stepped green area for bid depth building up as price moves away
from the spread on the left, and a stepped red area for ask depth building
up moving right, meeting at the spread in the middle.

Color note: green/bid vs red/ask is the universal convention for this exact
chart (matches every retail broker's L2 view, including the reference this
was built from) but red-green is a known colorblind confusion pair — this
function does NOT rely on hue alone to separate the two regions: they're
also spatially separated (opposite sides of the spread, with a visible gap
between best bid and best ask), hatch-textured differently, and directly
labeled ("Bid"/"Ask" plus best-price annotations), matching the dataviz
skill's secondary-encoding requirement for a red/green pair that fails the
CVD separation check on its own.
"""

from __future__ import annotations

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
except ImportError as e:  # pragma: no cover - matplotlib is a repo-wide dependency already
    raise ImportError('plot_order_book requires matplotlib (already a repo dependency)') from e

from prediction_markets.models import OrderBook

BID_COLOR = '#0ca30c'   # dataviz skill's status "good" step
ASK_COLOR = '#d03b3b'   # dataviz skill's status "critical" step
FILL_ALPHA = 0.25
LINE_WIDTH = 2


def _cumulative_depth(prices: list[float], sizes: list[float], *, ascending: bool) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative size at each price level, walking outward from the best
    price. `prices`/`sizes` must already be best-first (as OrderBook.bids/
    .asks are). Returns (x, y) sorted by price ascending either way, since
    that's the order the x-axis needs regardless of which side it's for."""
    cum = np.cumsum(sizes)
    prices = np.asarray(prices, dtype=float)
    if not ascending:
        # bids are best-first = price DESCENDING; reverse both so x is
        # ascending left-to-right while keeping each price paired with its
        # own (correct) cumulative value.
        prices = prices[::-1]
        cum = cum[::-1]
    return prices, cum


def plot_order_book(
    book: OrderBook,
    *,
    ax: Axes | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (8, 4.5),
) -> Axes:
    """Depth chart for one OrderBook snapshot. Pass `ax` to draw into an
    existing axes (e.g. a subplot grid); otherwise a new figure is created.
    Returns the axes either way, so callers can keep annotating it."""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    bid_prices = [lvl.price for lvl in book.bids]
    bid_sizes = [lvl.size for lvl in book.bids]
    ask_prices = [lvl.price for lvl in book.asks]
    ask_sizes = [lvl.size for lvl in book.asks]

    if bid_prices:
        x, y = _cumulative_depth(bid_prices, bid_sizes, ascending=False)
        ax.step(x, y, where='post', color=BID_COLOR, linewidth=LINE_WIDTH, label='Bid')
        ax.fill_between(x, y, step='post', color=BID_COLOR, alpha=FILL_ALPHA, hatch='//')
    if ask_prices:
        x, y = _cumulative_depth(ask_prices, ask_sizes, ascending=True)
        ax.step(x, y, where='post', color=ASK_COLOR, linewidth=LINE_WIDTH, label='Ask')
        ax.fill_between(x, y, step='post', color=ASK_COLOR, alpha=FILL_ALPHA, hatch='\\\\')

    # Spread marker + best bid/ask annotations, echoing the "BEST BID $x /
    # BEST ASK $y" readout above the reference chart -- these double as each
    # series' direct label (color-matched text), which is why plot_order_book
    # skips a separate legend in the normal two-sided case below.
    has_annotation = book.best_bid is not None and book.best_ask is not None
    if has_annotation:
        mid = (book.best_bid + book.best_ask) / 2
        ax.axvline(mid, color='0.75', linestyle='--', linewidth=1, zorder=0)
        # Anchored at the shared spread MIDPOINT rather than each side's own
        # (possibly nearly-identical) x -- a one-cent-wide spread would
        # otherwise force the two labels to collide, since their pixel
        # separation would be governed by the real (tiny) price gap instead
        # of a fixed padding. This mirrors the reference chart, which shows
        # "BEST BID x / BEST ASK y" as a header decoupled from the x-axis.
        # A light backing box, since a steep step (typical right at the best
        # price) can otherwise rise straight up through the label text.
        label_bg = dict(facecolor='white', edgecolor='none', alpha=0.75, pad=1.5)
        trans = ax.get_xaxis_transform()  # x in data coords, y in axes-fraction
        ax.annotate(f'Best bid: ${book.best_bid:,.4f}', xy=(mid, 0.97), xycoords=trans,
                    xytext=(-6, 0), textcoords='offset points', bbox=label_bg,
                    ha='right', va='top', color=BID_COLOR, fontsize=9, fontweight='bold')
        ax.annotate(f'Best ask: ${book.best_ask:,.4f}', xy=(mid, 0.97), xycoords=trans,
                    xytext=(6, 0), textcoords='offset points', bbox=label_bg,
                    ha='left', va='top', color=ASK_COLOR, fontsize=9, fontweight='bold')

    ax.set_xlabel('Price')
    ax.set_ylabel('Cumulative size')
    ax.set_title(title or f'{book.venue} — {book.market_id} order book depth')
    ax.grid(True, color='0.9', linewidth=0.8, zorder=0)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    # Only a one-sided (or empty) book falls back to a legend -- the normal
    # case is already directly labeled by the best-bid/ask annotations above.
    if (bid_prices or ask_prices) and not has_annotation:
        ax.legend(loc='upper right', frameon=False)
    return ax
