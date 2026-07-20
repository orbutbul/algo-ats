"""
execution/alpaca_paper.py — AlpacaExecutorBase wired to paper trading,
using the paper-account keys already in .env (ALPACA_API_KEY/ALPACA_SECRET_KEY).
"""

import os

from dotenv import load_dotenv

from execution.alpaca_base import AlpacaExecutorBase

load_dotenv()


class AlpacaPaperTradingExecutor(AlpacaExecutorBase):
    def __init__(self):
        super().__init__(os.environ['ALPACA_API_KEY'], os.environ['ALPACA_SECRET_KEY'], paper=True)
