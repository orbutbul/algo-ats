"""Shared constants for the live feed publisher (live_feed.py) and its
subscribers (live_client.py) — kept separate from both so importing one
doesn't pull in the other's side effects (e.g. live_feed's logging setup)."""

PUB_ADDRESS = 'tcp://127.0.0.1:5556'
HISTORY_ADDRESS = 'tcp://127.0.0.1:5557'
