"""
strategy_runner.py — polls LiveDataClient, feeds bars + current portfolio
state to a Strategy, and executes whatever Decisions it returns via an
AlpacaExecutorBase (paper trading by default — swap in
AlpacaLiveTradingExecutor deliberately, never by default).
"""

import logging
import time

from alpaca_paper_trading_executor import AlpacaPaperTradingExecutor
from live_client import LiveDataClient
from strategy_ma_cross import MovingAverageCrossStrategy

logging.basicConfig(level=logging.INFO)
log = logging.getLogger('strategy_runner')


def main(poll_seconds: int = 60):
    client = LiveDataClient()
    strategy = MovingAverageCrossStrategy()
    executor = AlpacaPaperTradingExecutor()
    try:
        while True:
            portfolio = executor.get_portfolio_state()
            decisions = strategy.on_data(client.snapshot(), portfolio)
            for d in decisions:
                log.info('executing decision: %r', d)
                executor.execute(d)
            time.sleep(poll_seconds)
    finally:
        client.stop()


if __name__ == '__main__':
    main()
