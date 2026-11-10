"""Alpaca paper broker adapter.

Execution role: build order payloads and call a transport. The only base URL
this module will use is the paper endpoint. There is no live URL. API keys
are read from the environment inside `from_env` and are scrubbed out of any
exception text. Tests inject a transport and never open a socket.

Not used by the CLI. Nothing in this package constructs it unless a caller
asks for paper mode with this class explicitly.
"""

from __future__ import annotations

import os
from typing import Callable

from live_signals.broker import PAPER_ACCOUNT_ID, assert_submission_allowed
from live_signals.errors import AccountGuardError, BrokerError
from live_signals.models import AccountView, BrokerOrder, BrokerPosition, OrderRequest, Quote, SubmitResult

PAPER_BASE_URL = "https://paper-api.alpaca.markets"

Transport = Callable[[str, str, dict | None], dict | list]


def scrub_text(message: str, secrets: list[str]) -> str:
    clean = message
    for secret in secrets:
        if secret:
            clean = clean.replace(secret, "***")
    return clean


def order_payload(order: OrderRequest) -> dict:
    """JSON body for POST /v2/orders. Limit, whole shares, no notional, no bracket, no trail."""
    if order.order_type != "limit":
        raise BrokerError("only limit orders are supported")
    if isinstance(order.qty, bool) or not isinstance(order.qty, int) or order.qty < 1:
        raise BrokerError("qty must be a positive whole number of shares")
    if order.extended_hours and order.time_in_force not in ("day", "gtc"):
        raise BrokerError("extended hours requires time_in_force day or gtc")
    return {
        "symbol": order.ticker,
        "qty": str(order.qty),
        "side": order.side,
        "type": "limit",
        "time_in_force": order.time_in_force,
        "limit_price": f"{order.limit_price:.2f}",
        "extended_hours": bool(order.extended_hours),
        "client_order_id": order.client_order_id,
    }


def _whole(value: object) -> int:
    number = float(value)  # type: ignore[arg-type]
    if abs(number - round(number)) > 1e-6:
        raise BrokerError("fractional share quantities are not supported")
    return int(round(number))


class AlpacaPaperBroker:
    def __init__(self, *, api_key: str, api_secret: str, transport: Transport, base_url: str | None = None) -> None:
        if not api_key or not api_secret:
            raise BrokerError("ALPACA_API_KEY and ALPACA_SECRET_KEY are required")
        url = (base_url or PAPER_BASE_URL).rstrip("/")
        if url != PAPER_BASE_URL:
            raise BrokerError("only the Alpaca paper endpoint is permitted")
        self._api_key = api_key
        self._api_secret = api_secret
        self._transport = transport
        self._base = url

    def __repr__(self) -> str:
        return "AlpacaPaperBroker(paper)"

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None, transport: Transport | None = None) -> AlpacaPaperBroker:
        source = os.environ if env is None else env
        api_key = source.get("ALPACA_API_KEY", "")
        api_secret = source.get("ALPACA_SECRET_KEY") or source.get("ALPACA_API_SECRET") or ""
        if not api_key or not api_secret:
            raise BrokerError("ALPACA_API_KEY and ALPACA_SECRET_KEY are required")
        if transport is None:
            transport = _requests_transport(api_key, api_secret)
        return cls(api_key=api_key, api_secret=api_secret, transport=transport)

    def mark_prices(self, marks: dict[str, float]) -> None:
        return None

    def set_quotes(self, quotes: dict[str, Quote]) -> None:
        return None

    def get_account(self) -> AccountView:
        account = self._call("GET", "/v2/account", None)
        positions = self._call("GET", "/v2/positions", None)
        orders = self._call("GET", "/v2/orders?status=open", None)
        return _account_view(account, positions, orders)

    def submit_order(self, order: OrderRequest) -> SubmitResult:
        account = self._call("GET", "/v2/account", None)
        _guard_account_payload(account)
        body = order_payload(order)
        forbidden = {"notional", "trail_price", "trail_percent", "order_class", "stop_price"}
        if forbidden & set(body):
            raise BrokerError("order payload contains a forbidden field")
        response = self._call("POST", "/v2/orders", body)
        return _submit_result(response)

    def cancel_order(self, client_order_id: str) -> None:
        account = self._call("GET", "/v2/account", None)
        _guard_account_payload(account)
        self._call("DELETE", f"/v2/orders:by_client_order_id?client_order_id={client_order_id}", None)

    def replace_order(self, client_order_id: str, limit_price: float) -> SubmitResult:
        account = self._call("GET", "/v2/account", None)
        _guard_account_payload(account)
        current = self._call("GET", f"/v2/orders:by_client_order_id?client_order_id={client_order_id}", None)
        broker_id = str(current.get("id", ""))
        if not broker_id:
            raise BrokerError("alpaca order id missing")
        response = self._call("PATCH", f"/v2/orders/{broker_id}", {"limit_price": f"{limit_price:.2f}"})
        return _submit_result(response)

    def _call(self, method: str, path: str, body: dict | None) -> dict | list:
        try:
            return self._transport(method, self._base + path, body)
        except BrokerError as exc:
            raise BrokerError(scrub_text(str(exc), [self._api_key, self._api_secret])) from None
        except Exception as exc:
            message = scrub_text(str(exc), [self._api_key, self._api_secret])
            raise BrokerError(message) from None


def _guard_account_payload(account: dict | list) -> None:
    if not isinstance(account, dict):
        raise BrokerError("alpaca account response was not an object")
    number = str(account.get("account_number", ""))
    # Paper accounts are the only ones this endpoint should return. The
    # account number is the guard the engine also applies; both have to pass.
    try:
        assert_submission_allowed("paper", number, "paper")
    except AccountGuardError as exc:
        raise AccountGuardError(str(exc)) from None
    if number != PAPER_ACCOUNT_ID:
        raise AccountGuardError(f"refusing orders: account {number} is not {PAPER_ACCOUNT_ID}")


def _account_view(account: dict | list, positions: dict | list, orders: dict | list) -> AccountView:
    if not isinstance(account, dict):
        raise BrokerError("alpaca account response was not an object")
    parsed_positions: dict[str, BrokerPosition] = {}
    for item in positions or []:
        if not isinstance(item, dict):
            continue
        qty = _whole(item.get("qty", 0))
        parsed_positions[item["symbol"]] = BrokerPosition(
            ticker=item["symbol"],
            qty=qty,
            avg_price=float(item.get("avg_entry_price") or 0),
        )
    parsed_orders: dict[str, BrokerOrder] = {}
    for item in orders or []:
        if not isinstance(item, dict):
            continue
        cid = str(item.get("client_order_id") or "")
        if not cid:
            continue
        limit = item.get("limit_price")
        parsed_orders[cid] = BrokerOrder(
            client_order_id=cid,
            ticker=str(item.get("symbol") or ""),
            side=str(item.get("side") or ""),
            qty=abs(_whole(item.get("qty") or 0)),
            limit_price=float(limit) if limit is not None else None,
            status=str(item.get("status") or "open"),
        )
    return AccountView(
        account_id=str(account.get("account_number") or ""),
        account_type="paper",
        equity=float(account.get("equity") or 0),
        cash=float(account.get("cash") or 0),
        positions=parsed_positions,
        open_orders=parsed_orders,
    )


def _submit_result(response: dict | list) -> SubmitResult:
    if not isinstance(response, dict):
        raise BrokerError("alpaca order response was not an object")
    status = str(response.get("status") or "open")
    if status == "filled":
        qty = abs(_whole(response.get("filled_qty") or response.get("qty") or 0))
        price = response.get("filled_avg_price")
        return SubmitResult("filled", float(price) if price is not None else None, qty)
    return SubmitResult("open")


def _requests_transport(api_key: str, api_secret: str) -> Transport:
    def transport(method: str, url: str, body: dict | None) -> dict | list:
        import requests

        try:
            response = requests.request(
                method,
                url,
                json=body,
                timeout=15,
                headers={
                    "APCA-API-KEY-ID": api_key,
                    "APCA-API-SECRET-KEY": api_secret,
                },
            )
            response.raise_for_status()
            if not response.content:
                return {}
            payload = response.json()
            if not isinstance(payload, (dict, list)):
                raise BrokerError("alpaca response was not json")
            return payload
        except BrokerError:
            raise
        except Exception as exc:
            raise BrokerError(scrub_text(str(exc), [api_key, api_secret])) from None

    return transport
