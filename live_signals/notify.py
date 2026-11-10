"""Notification hooks.

Execution role: the engine reports takes, exits, and halts. The default
implementation records them in memory. Nothing here opens a socket; a later
mail or Slack sender can implement `Notifier.emit` without changing the engine.
Payloads are scrubbed so a credential key cannot land in a message by accident.
"""

from __future__ import annotations

from typing import Protocol

_SECRET_HINTS = ("secret", "api_key", "apca", "authorization", "token", "password")


def scrub(payload: dict) -> dict:
    clean: dict = {}
    for key, value in payload.items():
        lowered = key.lower()
        if any(hint in lowered for hint in _SECRET_HINTS):
            clean[key] = "***"
        elif isinstance(value, dict):
            clean[key] = scrub(value)
        else:
            clean[key] = value
    return clean


class Notifier(Protocol):
    def emit(self, event: str, payload: dict) -> None: ...


class NullNotifier:
    def emit(self, event: str, payload: dict) -> None:
        return None


class MemoryNotifier:
    """Test double and the default stand-in until a real sender is wired."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def emit(self, event: str, payload: dict) -> None:
        self.events.append((event, scrub(dict(payload))))
