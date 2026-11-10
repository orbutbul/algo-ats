"""Run one live-signal cycle.

Execution role: load the book, plan, and either simulate fills (log_only) or
send them through a broker (paper). Log-only never calls broker methods, even
if a broker object was passed in. Paper mode calls `assert_submission_allowed`
after reading the account and before any cancel, replace, or submit. Live
trading is not a mode.

The decision log is written after the book is saved, so a crash cannot record
a take that the next cycle will repeat.
"""

from __future__ import annotations

from datetime import datetime

from live_signals.apply import apply_plan, mark_equity, resolve_resting, simulate_results
from live_signals.broker import Broker, assert_submission_allowed
from live_signals.config import Config
from live_signals.errors import AccountGuardError, LiveTradingDisabled
from live_signals.journal import DecisionLog
from live_signals.models import Decision, MarketSnapshot, NewsSignal, make_decision
from live_signals.notify import Notifier, NullNotifier
from live_signals.planner import plan_cycle
from live_signals.reconcile import sync_broker
from live_signals.risk import consume_drawdown_reset, kill_switch_active
from live_signals.sessions import classify_session
from live_signals.state import StateStore


class Engine:
    def __init__(
        self,
        config: Config,
        state: StateStore,
        journal: DecisionLog,
        broker: Broker | None = None,
        notifier: Notifier | None = None,
        *,
        kill_file: str | None = None,
        reset_file: str | None = None,
    ) -> None:
        if config.mode not in ("log_only", "paper"):
            raise LiveTradingDisabled("live trading is hard-disabled")
        self.config = config
        self.state = state
        self.journal = journal
        self.broker = broker
        self.notifier = notifier or NullNotifier()
        self.kill_file = kill_file if kill_file is not None else config.kill_file
        self.reset_file = reset_file if reset_file is not None else config.reset_file

    def run_once(
        self,
        now: datetime,
        signals: list[NewsSignal],
        market: MarketSnapshot,
    ) -> list[Decision]:
        if self.config.mode == "log_only":
            return self._run_log_only(now, signals, market)
        if self.config.mode == "paper":
            return self._run_paper(now, signals, market)
        raise LiveTradingDisabled("live trading is hard-disabled")

    def _flags(self, already_halted: bool) -> tuple[bool, bool]:
        kill = kill_switch_active(self.config, self.kill_file)
        # A reset file only clears a breaker that is already on. Consuming it
        # in the same cycle that trips the breaker would silently un-halt.
        reset = already_halted and consume_drawdown_reset(self.config, self.reset_file)
        return kill, reset

    def _emit(self, decisions: list[Decision], kill: bool) -> None:
        if kill or any(item.reason in ("daily_loss_halt", "drawdown_halt", "kill_switch") for item in decisions):
            reason = "kill_switch" if kill else next(
                (item.reason for item in decisions if item.reason in ("daily_loss_halt", "drawdown_halt")),
                "halt",
            )
            self.notifier.emit("halt", {"reason": reason})
        for item in decisions:
            if item.action in ("take", "exit"):
                self.notifier.emit(
                    item.action,
                    {"ticker": item.ticker, "reason": item.reason, "qty": item.qty},
                )

    def _run_log_only(self, now, signals, market) -> list[Decision]:
        book = resolve_resting(self.state.load(), market, now, self.config)
        equity = mark_equity(book, market.quotes)
        kill, reset = self._flags(book.risk.drawdown_halted)
        plan = plan_cycle(
            book, now, signals, market, self.config,
            kill=kill, reset_requested=reset, equity=equity, mode="log_only",
        )
        results = simulate_results(plan.intents, market)
        updated = apply_plan(
            book,
            position_edits=plan.position_edits,
            drop_working=plan.drop_working,
            replace_working=plan.replace_working,
            seen_add=plan.seen_add,
            intents=plan.intents,
            results=results,
            now=now,
            cfg=self.config,
            risk_state=plan.risk,
        )
        self.state.save(updated)
        self.journal.append(plan.decisions)
        self._emit(plan.decisions, kill)
        return plan.decisions

    def _run_paper(self, now, signals, market) -> list[Decision]:
        if self.broker is None:
            raise LiveTradingDisabled("paper mode requires a broker")
        session = classify_session(now, set(market.holidays), market.early_closes)
        marks = {ticker: quote.last for ticker, quote in market.quotes.items()}
        self.broker.mark_prices(marks)
        self.broker.set_quotes(market.quotes)
        account = self.broker.get_account()
        try:
            assert_submission_allowed("paper", account.account_id, account.account_type)
        except (AccountGuardError, LiveTradingDisabled):
            blocked = make_decision(
                now, ticker="", action="halt", reason="account_guard",
                session=session, mode="paper",
                details={"account_id": account.account_id},
            )
            self.journal.append([blocked])
            raise
        book = self.state.load()
        kill, reset = self._flags(book.risk.drawdown_halted)
        synced = sync_broker(
            book, account, now, market, self.config, session=session, mode="paper",
        )
        for client_id in synced.cancel_ids:
            self.broker.cancel_order(client_id)
        equity = account.equity
        plan = plan_cycle(
            synced.book, now, signals, market, self.config,
            kill=kill, reset_requested=reset, equity=equity, mode="paper",
        )
        results = {}
        for intent in plan.intents:
            if intent.action == "cancel":
                self.broker.cancel_order(intent.client_order_id)
            elif intent.action == "replace" and intent.new_limit is not None:
                self.broker.replace_order(intent.client_order_id, intent.new_limit)
            elif intent.action == "submit" and intent.order is not None:
                results[intent.order.client_order_id] = self.broker.submit_order(intent.order)
        updated = apply_plan(
            synced.book,
            position_edits=plan.position_edits,
            drop_working=plan.drop_working,
            replace_working=plan.replace_working,
            seen_add=plan.seen_add,
            intents=plan.intents,
            results=results,
            now=now,
            cfg=self.config,
            risk_state=plan.risk,
        )
        self.state.save(updated)
        decisions = list(synced.decisions) + list(plan.decisions)
        self.journal.append(decisions)
        self._emit(decisions, kill)
        return decisions
