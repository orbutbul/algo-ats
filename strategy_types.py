"""
strategy_types.py — data contracts shared by strategy_base.py and its
implementations.

Input side describes current broker state (positions, open orders), supplied
by the caller — a Strategy never queries a broker itself. Output side is
decisions only (open/update an order, close a position); no broker calls
happen here. Consumed later by alpaca_live_trading_executor.py /
alpaca_paper_trading_executor.py.
"""

from dataclasses import dataclass
from enum import Enum


class OrderSide(Enum):
    BUY = 'buy'
    SELL = 'sell'


class OrderType(Enum):
    MARKET = 'market'
    LIMIT = 'limit'


# ---------------------------------------------------------------------------
# Input context: current broker-side state, supplied by the caller
# ---------------------------------------------------------------------------

@dataclass
class Position:
    symbol: str
    qty: float                    # signed: negative = short
    avg_entry_price: float


@dataclass
class PendingOrder:
    order_id: str                 # broker-assigned id, opaque to the strategy
    symbol: str
    side: OrderSide
    qty: float
    order_type: OrderType
    limit_price: float | None = None


@dataclass
class PortfolioState:
    positions: dict[str, Position]      # symbol -> Position, nonzero only
    open_orders: list[PendingOrder]


# ---------------------------------------------------------------------------
# Output: decisions only, no broker calls
# ---------------------------------------------------------------------------

@dataclass
class OpenOrder:
    symbol: str
    side: OrderSide
    qty: float | None = None      # exactly one of qty/weight is set
    weight: float | None = None   # fraction of allocatable equity
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None


@dataclass
class UpdateOrder:
    order_id: str
    symbol: str
    qty: float | None = None
    limit_price: float | None = None


@dataclass
class ClosePosition:
    symbol: str
    fraction: float = 1.0         # 1.0 = full close, else partial


Decision = OpenOrder | UpdateOrder | ClosePosition
