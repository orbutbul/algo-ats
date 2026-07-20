"""
execution/alpaca_base.py — shared Alpaca order-execution logic.

Consumes Decision objects from strategy/types.py and submits them via
alpaca-py's TradingClient. Also reads current account/position/order state
back into PortfolioState so strategy_runner.py can feed a Strategy real
context instead of an empty placeholder.

Not used directly — execution/alpaca_paper.py and execution/alpaca_live.py
subclass this with their own credentials and paper/live flag.
"""

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide as AlpacaSide
from alpaca.trading.enums import OrderType as AlpacaOrderType
from alpaca.trading.enums import QueryOrderStatus, TimeInForce
from alpaca.trading.requests import (
    ClosePositionRequest,
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    ReplaceOrderRequest,
)

from strategy.types import (
    ClosePosition,
    Decision,
    OpenOrder,
    OrderSide,
    OrderType,
    PendingOrder,
    Position,
    PortfolioState,
    UpdateOrder,
)


class AlpacaExecutorBase:
    def __init__(self, api_key: str, secret_key: str, paper: bool):
        self._trading = TradingClient(api_key, secret_key, paper=paper)
        self._data = StockHistoricalDataClient(api_key, secret_key)

    def get_portfolio_state(self) -> PortfolioState:
        positions = {
            p.symbol: Position(symbol=p.symbol, qty=float(p.qty), avg_entry_price=float(p.avg_entry_price))
            for p in self._trading.get_all_positions()
        }
        orders = self._trading.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN))
        open_orders = [
            PendingOrder(
                order_id=str(o.id),
                symbol=o.symbol,
                side=OrderSide.BUY if o.side == AlpacaSide.BUY else OrderSide.SELL,
                qty=float(o.qty) if o.qty is not None else 0.0,
                # Alpaca orders can be stop/stop_limit/trailing_stop too, which our
                # Decision contract doesn't model — fold anything non-limit into MARKET
                # since it's context only (the strategy isn't meant to update those).
                order_type=OrderType.LIMIT if o.type == AlpacaOrderType.LIMIT else OrderType.MARKET,
                limit_price=float(o.limit_price) if o.limit_price is not None else None,
            )
            for o in orders
        ]
        return PortfolioState(positions=positions, open_orders=open_orders)

    def execute(self, decision: Decision):
        if isinstance(decision, OpenOrder):
            return self._open_order(decision)
        if isinstance(decision, UpdateOrder):
            return self._update_order(decision)
        if isinstance(decision, ClosePosition):
            return self._close_position(decision)
        raise TypeError(f'Unknown decision type: {type(decision)!r}')

    def _latest_price(self, symbol: str) -> float:
        trade = self._data.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=symbol))[symbol]
        return float(trade.price)

    def _resolve_qty(self, decision: OpenOrder) -> float:
        if decision.qty is not None:
            return decision.qty
        equity = float(self._trading.get_account().equity)
        price = self._latest_price(decision.symbol)
        return (decision.weight * equity) / price

    def _open_order(self, decision: OpenOrder):
        qty = self._resolve_qty(decision)
        side = AlpacaSide.BUY if decision.side == OrderSide.BUY else AlpacaSide.SELL
        if decision.order_type == OrderType.LIMIT:
            order_data = LimitOrderRequest(
                symbol=decision.symbol, qty=qty, side=side,
                time_in_force=TimeInForce.DAY, limit_price=decision.limit_price,
            )
        else:
            order_data = MarketOrderRequest(
                symbol=decision.symbol, qty=qty, side=side, time_in_force=TimeInForce.DAY,
            )
        return self._trading.submit_order(order_data=order_data)

    def _update_order(self, decision: UpdateOrder):
        order_data = ReplaceOrderRequest(qty=decision.qty, limit_price=decision.limit_price)
        return self._trading.replace_order_by_id(order_id=decision.order_id, order_data=order_data)

    def _close_position(self, decision: ClosePosition):
        close_options = None if decision.fraction >= 1.0 else ClosePositionRequest(percentage=str(decision.fraction * 100))
        return self._trading.close_position(symbol_or_asset_id=decision.symbol, close_options=close_options)
