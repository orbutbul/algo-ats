"""
vcp — Mark Minervini Volatility Contraction Pattern detector (research).

Pure detection + charting. Importing this package does not open a broker or
place orders. Price history is read only when a caller asks (see vcp.io).
"""

from vcp.detector import VCPBreakout, VCPResult, detect_vcp_breakouts
from vcp.params import VCPParams

__all__ = [
    'VCPBreakout',
    'VCPParams',
    'VCPResult',
    'detect_vcp_breakouts',
]
