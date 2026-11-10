"""
relative_strength — opt-in IBD/Minervini-inspired RS proxy.

Research calculator. Importing this package does not open DuckDB, the
network, or a broker, and the trading pipeline does not import it.
See relative_strength/README.md.
"""

from relative_strength.calculator import (
    BROAD_MARKET_MIN,
    LOOKBACKS,
    MAX_MISSING_SESSIONS,
    MIN_UNIVERSE_FOR_PERCENTILE,
    WEIGHTS,
    PriceSchemaError,
    RelativeStrengthResult,
    compute_relative_strength,
    percentile_from_average_rank,
)

__all__ = [
    'BROAD_MARKET_MIN',
    'LOOKBACKS',
    'MAX_MISSING_SESSIONS',
    'MIN_UNIVERSE_FOR_PERCENTILE',
    'WEIGHTS',
    'PriceSchemaError',
    'RelativeStrengthResult',
    'compute_relative_strength',
    'percentile_from_average_rank',
]
