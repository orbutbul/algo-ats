"""
strategy/utils.py — helpers shared by Strategy implementations.

Strategy.on_data() must always return list[Decision] regardless of whether
a strategy reasons about one asset at a time (strategy/ma_cross.py) or a
full-universe target-weight vector (cross-asset/cross-sectional strategies).
weights_to_decisions() is the translation for the latter case, so
strategy_runner.py never needs to know which kind of strategy it's driving.
"""

from strategy.types import ClosePosition, Decision, OpenOrder, OrderSide, PortfolioState


def weights_to_decisions(target_weights: dict[str, float], portfolio: PortfolioState) -> list[Decision]:
    """
    Diffs a full-universe target-weight vector against currently held
    positions and returns the Decisions needed to move toward it.

    v1 scope: opens new positions for symbols newly present with nonzero
    target weight, and closes positions for symbols no longer in the target
    set (or targeted at zero). Does NOT resize positions that are already
    held and whose target weight simply changed -- computing that delta
    needs current market value, which lives with the executor (via account
    equity + live price), not here.
    """
    held = set(portfolio.positions)
    targeted = {symbol for symbol, weight in target_weights.items() if weight != 0}

    decisions: list[Decision] = [
        OpenOrder(
            symbol=symbol,
            side=OrderSide.BUY if target_weights[symbol] > 0 else OrderSide.SELL,
            weight=abs(target_weights[symbol]),
        )
        for symbol in sorted(targeted - held)
    ]
    decisions += [ClosePosition(symbol=symbol) for symbol in sorted(held - targeted)]
    return decisions
