"""
Direct client for the Robinhood MCP trading server, bypassing Claude Code.

Reuses the OAuth access token that Claude Code already obtained (stored in
%USERPROFILE%\\.claude\\.credentials.json under mcpOAuth). Speaks MCP's
JSON-RPC 2.0 over the streamable-HTTP transport directly to
https://agent.robinhood.com/mcp/trading.

This is a REAL trading connection. place_order() defaults to dry_run=True,
which calls review_equity_order (no order placed) instead of
place_equity_order. You must pass dry_run=False explicitly to send a real
order.
"""

import json
import os
import time
import uuid
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

_raw_credentials_path = os.environ.get("CLAUDE_CREDENTIALS_PATH")
if not _raw_credentials_path:
    raise RuntimeError("CLAUDE_CREDENTIALS_PATH is not set (check your .env file)")
CREDENTIALS_PATH = os.path.expanduser(_raw_credentials_path)
MCP_KEY_PREFIX = "robinhod-trading|"
TOKEN_ENDPOINT = "https://api.robinhood.com/oauth2/token/"

# The access token Claude Code obtains interactively is short-lived (observed
# ~5-10 minutes) and is normally kept fresh by Claude Code silently
# refreshing it during an active session. An unattended run (e.g. this
# module invoked from the Airflow daily_run DAG) has no such session
# refreshing it in the background, so by the time a scheduled job fires the
# cached token is routinely already dead -- refresh proactively here instead
# of just failing, using the long-lived refresh_token stored alongside it.
REFRESH_BUFFER_MS = 5 * 60 * 1000


def _refresh_access_token(key: str, match: dict) -> dict:
    """
    Exchanges match['refreshToken'] for a new access token via the
    standard OAuth refresh_token grant (public client, no secret --
    token_endpoint_auth_methods_supported is 'none' per the server's
    discovery metadata) and persists the result back into
    CREDENTIALS_PATH so Claude Code and any other consumer pick up the
    same refreshed token instead of it silently going stale.
    """
    resp = requests.post(
        TOKEN_ENDPOINT,
        data={
            "grant_type": "refresh_token",
            "refresh_token": match["refreshToken"],
            "client_id": match["clientId"],
        },
        timeout=15,
    )
    resp.raise_for_status()
    payload = resp.json()

    match = {
        **match,
        "accessToken": payload["access_token"],
        "refreshToken": payload.get("refresh_token", match["refreshToken"]),
        "expiresAt": time.time() * 1000 + payload["expires_in"] * 1000,
    }

    with open(CREDENTIALS_PATH, "r") as f:
        creds = json.load(f)
    creds.setdefault("mcpOAuth", {})[key] = match
    with open(CREDENTIALS_PATH, "w") as f:
        json.dump(creds, f, indent=2)

    return match


def _load_mcp_credentials() -> dict:
    with open(CREDENTIALS_PATH, "r") as f:
        creds = json.load(f)

    mcp_oauth = creds.get("mcpOAuth", {})
    key = next((k for k in mcp_oauth if k.startswith(MCP_KEY_PREFIX)), None)
    if key is None:
        raise RuntimeError(
            "No robinhod-trading MCP credentials found. "
            "Connect it in Claude Code first (the /mcp command)."
        )
    match = mcp_oauth[key]

    expires_at_ms = match.get("expiresAt")
    if not expires_at_ms or time.time() * 1000 > expires_at_ms - REFRESH_BUFFER_MS:
        if not match.get("refreshToken"):
            raise RuntimeError(
                "Robinhood MCP access token has expired and no refresh_token "
                "is on file. Reconnect via `/mcp` in Claude Code, then re-run "
                "this script."
            )
        try:
            match = _refresh_access_token(key, match)
        except Exception as e:
            raise RuntimeError(
                "Robinhood MCP access token has expired and refreshing it "
                f"failed ({e}). Reconnect via `/mcp` in Claude Code, then "
                "re-run this script."
            ) from e

    return match


class RobinhoodMCPClient:
    def __init__(self):
        creds = _load_mcp_credentials()
        self.server_url = creds["serverUrl"]
        self.access_token = creds["accessToken"]
        self.session_id: Optional[str] = None
        self._next_id = 1
        self._initialize()

    def _headers(self) -> dict:
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        return headers

    def _next_request_id(self) -> int:
        rid = self._next_id
        self._next_id += 1
        return rid

    @staticmethod
    def _parse_response(resp: requests.Response) -> dict:
        content_type = resp.headers.get("Content-Type", "")
        if "text/event-stream" in content_type:
            for line in resp.text.splitlines():
                if line.startswith("data:"):
                    return json.loads(line[len("data:"):].strip())
            raise RuntimeError(f"No data event in SSE response: {resp.text!r}")
        return resp.json()

    def _post(self, payload: dict, expect_response: bool = True) -> Optional[dict]:
        resp = requests.post(self.server_url, headers=self._headers(), json=payload, timeout=30)
        if "Mcp-Session-Id" in resp.headers:
            self.session_id = resp.headers["Mcp-Session-Id"]
        resp.raise_for_status()
        if not expect_response:
            return None
        return self._parse_response(resp)

    def _initialize(self):
        init_payload = {
            "jsonrpc": "2.0",
            "id": self._next_request_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "robinhood-direct-client", "version": "0.1.0"},
            },
        }
        self._post(init_payload)

        # Required notification, no response expected.
        self._post(
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            expect_response=False,
        )

    def call_tool(self, name: str, arguments: dict) -> dict:
        payload = {
            "jsonrpc": "2.0",
            "id": self._next_request_id(),
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
        print("--- JSON-RPC request sent ---")
        print(json.dumps(payload, indent=2))

        result = self._post(payload)

        print("--- JSON-RPC response received ---")
        print(json.dumps(result, indent=2))
        return result

    def place_order(
        self,
        ticker: str,
        side: str,
        account_number: str,
        shares: Optional[float] = None,
        dollar_amount: Optional[float] = None,
        order_type: str = "market",
        limit_price: Optional[str] = None,
        stop_price: Optional[str] = None,
        time_in_force: str = "gfd",
        market_hours: str = "regular_hours",
        dry_run: bool = True,
    ) -> dict:
        """
        Buy/sell equities through the Robinhood MCP server.

        Exactly one of `shares` or `dollar_amount` must be given
        (dollar_amount requires order_type='market'). dry_run=True (default)
        previews the order via review_equity_order without sending it; pass
        dry_run=False to place a real order.
        """
        if side not in ("buy", "sell"):
            raise ValueError("side must be 'buy' or 'sell'")
        if (shares is None) == (dollar_amount is None):
            raise ValueError("provide exactly one of shares or dollar_amount")

        arguments = {
            "account_number": account_number,
            "symbol": ticker.upper(),
            "side": side,
            "type": order_type,
            "time_in_force": time_in_force,
            "market_hours": market_hours,
        }
        if shares is not None:
            arguments["quantity"] = str(shares)
        if dollar_amount is not None:
            arguments["dollar_amount"] = f"{dollar_amount:.2f}"
        if limit_price is not None:
            arguments["limit_price"] = str(limit_price)
        if stop_price is not None:
            arguments["stop_price"] = str(stop_price)

        if dry_run:
            return self.call_tool("review_equity_order", arguments)

        arguments["ref_id"] = str(uuid.uuid4())
        return self.call_tool("place_equity_order", arguments)


if __name__ == "__main__":
    # Example: preview only. Pass dry_run=False to actually send it.
    client = RobinhoodMCPClient()
    client.place_order(
        ticker="NVDA",
        side="buy",
        account_number="582128930",
        shares=1,
        dry_run=False,
    )
