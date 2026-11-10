"""Prefer vectorbtpro when installed; fall back to open-source vectorbt."""

from __future__ import annotations

try:
    from vectorbtpro import vbt  # type: ignore
    VBT_FLAVOR = 'vectorbtpro'
except ImportError:  # pragma: no cover - exercised in OSS-only envs
    import vectorbt as vbt  # type: ignore
    VBT_FLAVOR = 'vectorbt'

__all__ = ['VBT_FLAVOR', 'vbt']
