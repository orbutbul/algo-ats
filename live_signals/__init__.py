"""Live, event-driven paper trader for Wirewatch news signals.

Self-contained package. It does not import the existing ATS pipeline and it
does not place orders on import. The design, formulas, and rollout are in
DESIGN.md. The CLI (`python -m live_signals`) is log-only.
"""

from live_signals.broker import PAPER_ACCOUNT_ID

__all__ = ["PAPER_ACCOUNT_ID"]
