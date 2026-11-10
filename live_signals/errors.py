"""Error types for the live-signal engine.

Execution role: callers distinguish a hard disabled live path, a paper-account
mismatch, a broker failure, and a bad config without parsing message text.
"""


class ConfigError(ValueError):
    """A config file is missing keys, has unknown keys, or fails a range check."""


class LiveTradingDisabled(RuntimeError):
    """Submission was requested in a mode other than paper.

    There is no live-money mode. `log_only` records decisions and does not
    submit. Any other mode, including `live`, raises this.
    """


class AccountGuardError(RuntimeError):
    """Paper submission was pointed at an account other than PA32ZMSEI."""


class BrokerError(RuntimeError):
    """A broker call failed. The message must not contain credentials."""
