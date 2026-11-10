"""Static sector map for the 30% sector cap.

Execution role: a checked-in ticker → sector table so the cap works without a
fundamentals feed. These are liquid US names a news scanner actually hits,
labeled with the GICS sector an ETF such as XLK or XLF would use. A ticker
that is not listed is uncapped, not dumped into an "unknown" bucket. Add a
line here when a name outside the list starts to matter.
"""

from __future__ import annotations

# GICS-style sector names. The cap groups by this string.
SECTORS: dict[str, str] = {
    # Technology
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology", "AVGO": "Technology",
    "ORCL": "Technology", "CRM": "Technology", "AMD": "Technology", "ADBE": "Technology",
    "CSCO": "Technology", "IBM": "Technology", "INTC": "Technology", "AMAT": "Technology",
    "MU": "Technology", "QCOM": "Technology", "NOW": "Technology", "INTU": "Technology",
    "TXN": "Technology", "PANW": "Technology", "CRWD": "Technology", "SNPS": "Technology",
    "CDNS": "Technology", "AMZN": "Consumer Discretionary",
    # Communication Services
    "GOOGL": "Communication Services", "GOOG": "Communication Services", "META": "Communication Services",
    "NFLX": "Communication Services", "DIS": "Communication Services", "CMCSA": "Communication Services",
    "TMUS": "Communication Services", "VZ": "Communication Services", "T": "Communication Services",
    # Consumer Discretionary
    "TSLA": "Consumer Discretionary", "HD": "Consumer Discretionary", "MCD": "Consumer Discretionary",
    "NKE": "Consumer Discretionary", "LOW": "Consumer Discretionary", "SBUX": "Consumer Discretionary",
    "BKNG": "Consumer Discretionary", "TJX": "Consumer Discretionary", "CMG": "Consumer Discretionary",
    # Consumer Staples
    "WMT": "Consumer Staples", "PG": "Consumer Staples", "KO": "Consumer Staples",
    "PEP": "Consumer Staples", "COST": "Consumer Staples", "PM": "Consumer Staples",
    "MO": "Consumer Staples", "CL": "Consumer Staples", "MDLZ": "Consumer Staples",
    # Financials
    "BRK.B": "Financials", "BRK-B": "Financials", "JPM": "Financials", "V": "Financials",
    "MA": "Financials", "BAC": "Financials", "GS": "Financials", "MS": "Financials",
    "WFC": "Financials", "C": "Financials", "BLK": "Financials", "AXP": "Financials",
    "SCHW": "Financials", "SPGI": "Financials", "CME": "Financials",
    # Health Care
    "LLY": "Health Care", "UNH": "Health Care", "JNJ": "Health Care", "MRK": "Health Care",
    "ABBV": "Health Care", "TMO": "Health Care", "ABT": "Health Care", "DHR": "Health Care",
    "PFE": "Health Care", "AMGN": "Health Care", "BMY": "Health Care", "ISRG": "Health Care",
    # Industrials
    "GE": "Industrials", "CAT": "Industrials", "RTX": "Industrials", "HON": "Industrials",
    "UNP": "Industrials", "DE": "Industrials", "UPS": "Industrials", "BA": "Industrials",
    "LMT": "Industrials",
    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy", "EOG": "Energy",
    # Materials
    "LIN": "Materials", "SHW": "Materials", "FCX": "Materials", "NEM": "Materials",
    # Real Estate
    "PLD": "Real Estate", "AMT": "Real Estate", "EQIX": "Real Estate", "SPG": "Real Estate",
    # Utilities
    "NEE": "Utilities", "SO": "Utilities", "DUK": "Utilities", "AEP": "Utilities",
}


def sector_of(ticker: str) -> str | None:
    """Return the static sector, or None when this name is not in the map."""
    return SECTORS.get(ticker.upper())
