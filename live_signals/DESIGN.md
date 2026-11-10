# Live signal paper trader

A self-contained replacement for the once-a-day 8am batch on Alpaca paper account **PA32ZMSEI** ("Grok Recs", about $15k). Wirewatch already scores news through the day. This package decides, as each signal arrives, whether it is credible enough to trade, how many shares that conviction is worth, and when the position is done.

Nothing in this package is wired into the existing ATS. That system uses a different paper account (`PA3V3A1X0IX6`). Its pipeline, cron, and data files are not imported and not modified. The only edit outside `live_signals/` is the pytest `testpaths` entry.

This task does not deploy the service and does not send an order. The default mode is `log_only`. There is no live-money mode.

## How a cycle runs

Every 10 minutes (configurable, 5–15):

1. Read a batch of `NewsSignal`s and a market snapshot (quotes, ATR, optional confirmation, earnings/macro events). Both are injected. This package does not call Wirewatch and does not fetch quotes.
2. Classify the ET session. Closed sessions, including Friday 20:00 through Sunday 20:00 and the night before a market holiday, take no new risk.
3. If the kill switch, the daily-loss halt, or the drawdown breaker is on, flatten and skip every signal.
4. Otherwise review resting orders, then exits, then new entries.
5. Append one CSV row per signal (taken or skipped, with the reason) and one row per exit.
6. In `log_only` (the default) apply those fills to a local book so the next cycle can manage the hypothetical position. No broker method is called. In `paper`, submit only after the account guard passes.

State for one cycle is a single JSON document inside SQLite (`live_signals_state.db`), so a crash cannot save a position without its stop. The decision log is `live_signals_decisions.csv`.

## Signal interface

Wirewatch is not a dependency. One object is the whole contract:

| Field | Meaning |
| --- | --- |
| `ticker` | Symbol |
| `score` | 0–100 |
| `sentiment` | `bullish`, `bearish`, or `neutral` |
| `sources` | URL plus source type (`news`, `filing`, `wire`, `press`, `reddit`, `x`, `social`) |
| `headline`, `rationale` | Text, stored on the decision row, not sent to the broker |
| `timestamp` | Timezone-aware. Naive timestamps are rejected |
| `signal_id` | Stable id. Missing ids are hashed from ticker, time, headline, score, and sentiment |

`parse_wirewatch_payload` is the only function that knows Wirewatch's key names (`type` or `source_type`, `id` or `signal_id`). `WirewatchPushSource.fetch` is a stub and does not open a socket. Until Wirewatch can push, point `JsonFileSignalSource` at a JSON file the scanner overwrites.

Market data is the same shape: a `MarketSnapshot` the caller fills in. Confirmation, when present, is a return since the signal and a relative volume. ATR is in dollars. Sector and return-correlation are optional; when they are missing those caps do not fire (see below).

## 1. Qualification

Neutral is never a trade. A signal older than **45 minutes** is `stale`. A timestamp more than **2 minutes** in the future is rejected. Score below **70** is rejected before any of the source rules.

Source types in `news`, `filing`, `wire`, `press` are institutional. `reddit`, `x`, `social`, and anything unrecognized are social. Credibility uses the number of distinct URL domains, not the number of headlines from the same site.

| Source mix | Tradable when |
| --- | --- |
| One social domain | Score **≥ 88**, or score **≥ 78** and price/volume confirmation |
| Two or more social domains | Score **≥ 80**, or score **≥ 70** and confirmation |
| One institutional domain | Score **≥ 74** |
| Two or more institutional domains | Score **≥ 70** |

Confirmation means the price has moved at least **0.40%** in the signal's direction and relative volume is at least **1.5×**. A single Reddit post at 70, which is what the morning batch bought on GOOGL, fails this gate even if the price has ticked.

The credibility multiplier, used by sizing, is

```
c = 0.45 + 0.18 * (domains - 1)
    + 0.15  if any institutional source
    + 0.10  if confirmed
c = clamp(c, 0.40, 1.00)
```

So one unconfirmed social source is **0.45**, one news source is **0.60**, two news domains are **0.78**, and four news domains (or three confirmed news domains) saturate at **1.00**.

Other gates, in order:

- The same `signal_id` is a duplicate.
- Same ticker and direction within **120 minutes** is `cooldown`, unless the new score is at least **8 points** above the last fill (`score_upgrade`), in which case we buy only the extra shares.
- An opposite signal inside **180 minutes** is `conflict` unless it clears the flip rule in the exit section. Two opposite signals in the same batch with equal top scores are both `ambiguous_conflict`.
- A name already at its target size is `already_sized`.

## 2. Sizing

The target is a whole-share quantity. Size is the dollars at risk if the initial stop is hit, not a fraction of equity split equally across whatever scored 70.

### Conviction

A logistic curve, rescaled so score 70 maps to a floor and score 95 maps to 1. Below 70 the value is 0 (and qualification has already rejected the signal).

```
L(s) = 1 / (1 + exp(-0.18 * (s - 82)))
t(s) = (L(s) - L(70)) / (L(95) - L(70))
conviction(s) = 0                              if s < 70
               = 1                              if s >= 95
               = 0.15 + 0.85 * t(s)             otherwise
```

The curve is convex from the threshold through the midpoint: a 70 is a small position, an 80 is several times larger, and a 95 does not size past a 100. With these constants:

| Score | Conviction |
| --- | --- |
| 70 | 0.150 |
| 80 | 0.473 |
| 90 | 0.891 |
| 95 and above | 1.000 |

### Risk budget

```
stop_distance = max(1.5 * ATR, 0.8% * price)

vol_risk = 0.60% * conviction * credibility * session_multiplier
kelly_cap = min(0.25 * max(p - (1 - p) / 1.75, 0), 1.00%)
p = 0.42 + 0.18 * conviction

risk_fraction = min(vol_risk, kelly_cap)
risk_dollars  = risk_fraction * equity
shares        = floor(risk_dollars / stop_distance)
```

`p` is an assumed win rate, not a fitted probability. Quarter-Kelly is a ceiling. At the default 0.60% base risk it does not bind, because even full conviction stays under the 1% cap. Raising `base_risk_pct` in the config cannot put more than **1% of equity** at risk on one trade. The test `test_kelly_ceiling_binds_only_when_the_vol_budget_is_raised` locks that.

Session multiplier (also the per-position notional cap):

| Session | Risk multiplier | Position cap |
| --- | --- | --- |
| Regular hours 09:30–16:00 | 1.00 | 12% |
| Pre-market 04:00–09:30 | 0.55 | 7% |
| After-hours 16:00–20:00 | 0.50 | 6% |
| Overnight 20:00–04:00 | 0.35 | 4% |
| Closed | 0 | 0 |

The share count is then cut by the tightest of:

| Cap | Default |
| --- | --- |
| Per position | The session cap above |
| Gross exposure | 85% of equity |
| Net exposure | 55% of equity |
| Open positions | 8 names, counting resting entries |
| Sector | 30% of equity, only when a sector is supplied |
| Correlation cluster | 25% of equity for names with correlation ≥ 0.75, only when a correlation is supplied |
| New risk opened today | 2.0% of equity, summed as `shares * stop_distance`, released if the entry is cancelled unfilled |
| New entries today | 6, not counting exits or the scale-in clip |

A missing sector or a missing correlation does not block the trade. Unknown names are exempt rather than dumped into one bucket, because a missing map would otherwise cap the whole book at 30%. Net at 55% is deliberate: the morning batch deploys 90% of equity in one direction, and this book will not.

In regular hours and pre-market the target is split **60% / 40%**. The second clip is sent only if price moves **0.35 ATR** in favor within **45 minutes**. After-hours and overnight send the target in one clip. The second clip is dropped if the daily risk budget no longer fits.

### Worked examples

Equity **$15,000**, price **$100**, credibility **1.00**, regular hours. Stop = `max(1.5 × ATR, $0.80)`.

"Budget" is `equity × 0.60% × conviction` before the share floor and the caps. "At risk" is `final shares × stop`, which is what the book can actually lose.

Low vol is ATR **$1.20** (stop **$1.80**). High vol is ATR **$4.00** (stop **$6.00**).

| Score | Conviction | Vol | Budget | Uncapped shares | Cap | Final shares | Notional | At risk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 70 | 0.150 | low | $13.50 | 7 | 18 | 7 | $700 | $12.60 |
| 80 | 0.473 | low | $42.59 | 23 | 18 | 18 | $1,800 | $32.40 |
| 90 | 0.891 | low | $80.19 | 44 | 18 | 18 | $1,800 | $32.40 |
| 70 | 0.150 | high | $13.50 | 2 | 18 | 2 | $200 | $12.00 |
| 80 | 0.473 | high | $42.59 | 7 | 18 | 7 | $700 | $42.00 |
| 90 | 0.891 | high | $80.19 | 13 | 18 | 13 | $1,300 | $78.00 |

At low vol the 12% cap pins both 80 and 90 at 18 shares, so the extra conviction cannot be spent. At high vol the stop is wide and the score still shows up: 2, then 7, then 13.

The same score-90 high-vol name with a single unconfirmed social source (credibility **0.45**) in the overnight session (multiplier **0.35**) has a budget of about **$12.63** and a final size of **2 shares / $200**, against 13 shares in regular hours at full credibility. These counts are locked by `test_high_vol_shares_rise_with_score`, `test_low_vol_hits_the_position_cap`, and `test_credibility_and_session_shrink_size`.

In regular hours that 13-share target is not sent as one order. The first clip is 8 shares and 5 are reserved for the scale-in. Overnight sends all 4.

## 3. Entry

Every order this engine builds is a **limit**, **whole shares**, time in force **`day`**. No market, bracket, notional, or broker trailing-stop order is representable: `OrderRequest` rejects them. Outside regular hours `extended_hours` is true. Alpaca accepts `day` or `gtc` on those orders; we send `day` because a news limit must not rest for days. A day order placed in the overnight session can stay live at the broker until 20:00 ET, so the engine cancels on its own timer instead of trusting the time in force.

Limit vs the mid, then capped vs the **original** decision mid so a replace cannot chase a spike:

```
buy:  min(ask, max(mid * (1 + offset), bid)), then ≤ decision_mid * (1 + max_chase)
sell: max(bid, min(mid * (1 - offset), ask)), then ≥ decision_mid * (1 - max_chase)
```

| Session | Hours ET | extended_hours | Offset | Max chase | Max spread | Cancel / replace |
| --- | --- | --- | --- | --- | --- | --- |
| Overnight | 20:00–04:00, Sun evening through Fri morning | true | 20 bp | 50 bp | 1.50% | 300s, one replace |
| Pre-market | 04:00–09:30 | true | 10 bp | 30 bp | 0.80% | 180s, one replace |
| Regular | 09:30–16:00 | false | 5 bp | 15 bp | 0.40% | 120s, one replace |
| After-hours | 16:00–20:00 | true | 10 bp | 35 bp | 1.00% | 180s, one replace |
| Closed | Weekends, holidays, Fri 20:00–Sun 20:00, holiday eves | no new orders | | | | |

Early-close days are an injected map of date → regular-session end. After-hours then runs four hours, capped at 20:00. That four-hour assumption is not an exchange feed.

Skip the entry, and write the reason, when:

- The spread is wider than the session max, the quote is crossed, or the symbol is halted.
- The last price is already more than **1.25 ATR** past the prior regular close, in the signal's direction (`gap_already_extended`).
- The side is short and the name is not shortable or not easy to borrow.
- The session is closed, or the clock would flatten a position of this score immediately (Friday from 15:50, a sub-85 score from 15:50, the night before a holiday).
- Earnings for this ticker are inside **12 hours before through 2 hours after**.
- A macro print (kind `macro`, `cpi`, or `fomc`) is inside **30 minutes before through 15 minutes after**.

Prices under $1 are skipped. Sub-penny names are out of scope; every limit is rounded to a penny.

Client order ids are `ww-` plus a hash of purpose, ticker, signal id, tranche, and attempt. The same logical order always gets the same id, and a retry does not double-fill.

## 4. Exits

The broker is not asked to trail. The stop lives on the position and is checked against the latest bar each cycle, which is what makes it valid in extended hours. On a bar that both runs in favor and trades through the stop, the **existing** stop wins. The trail is updated only for the next bar, so a bar cannot stop itself out with a stop it just invented.

Initial risk per share is the same `stop_distance` as sizing, measured from the fill.

| Rule | Long | Short | Default |
| --- | --- | --- | --- |
| Initial stop | fill − R | fill + R | R = max(1.5 × ATR, 0.8% of price) |
| Arm the trail | high ≥ entry + 1.0 R | low ≤ entry − 1.0 R | Trail distance = max(1.0 × ATR, 0.6 × R). The stop only tightens |
| Trail exit | bar low ≤ stop | bar high ≥ stop | Reason `trail_stop` once the stop has moved, else `stop` |
| Scale out, qty ≥ 3 | Sell 40% at +1.5 R and 30% at +2.5 R, leave the rest | Mirror | Fractions of the shares held when the level is hit. Both levels on one bar sell both slices |
| Small position | Full exit at +2.0 R | Mirror | Qty under 3 |
| Alpha decay | Flat if peak favorable excursion is still under 0.4 R | Same | After **8 hours** of non-closed session time, not wall-clock. Weekends do not count |
| Time stop | Flat | Same | **36 hours** of wall-clock time, even if the trade worked |
| Reversal | Flat | Same | A fresh opposite signal that itself passes the source gate and scores at least as high as the position (and at least 65) |
| Invalidation | Flat | Same | A fresh `neutral` signal with score ≥ 75 |
| Flip | After that exit, open the other side | Same | Only if the new score is ≥ **80** and at least **5 points** above the position. Otherwise flatten and stay flat |
| Into the close | Flat from 15:50 ET | Same | Score under **85**, any Friday, or any evening whose overnight session will not run (weekend or the night before a holiday) |
| Overnight hold | Allowed | Same | Score ≥ 85, and only Sun evening through Thu evening when the next regular session exists |
| Pre-open | Flat at 09:20–09:30 if still red | Same | Only positions that were opened outside regular hours and scored under 85. A winner is allowed to convert to the regular session |
| Earnings | Flat | Same | From **4 hours** before the print until **30 minutes** after, for that ticker |
| Macro, score under 90 | Flat | Same | From **15 minutes** before CPI/FOMC until the print |
| Macro, score ≥ 90 | Hold, stop pulled in to 0.75 R | Mirror | Never loosened. New entries are still blocked through 15 minutes after the print |

Exits are priced at the touch (sell the bid, buy the ask) even when the spread is wide. Getting flat matters more than saving the spread. Entries still refuse that spread.

## 5. Risk controls

| Control | Default | What it does |
| --- | --- | --- |
| Daily loss | 1.5% of the equity snapshotted at the first cycle of the ET date | Flattens and blocks new entries until the next ET date. The first cycle of a process does not trip on losses that happened before it was watching |
| Max drawdown | 8% from the persisted equity peak | Flattens and stays halted until `LIVE_SIGNALS_RESET_DRAWDOWN=1` or the reset file is present. The reset is consumed (file deleted, env var popped) and the peak is rebased to current equity. A reset file that is already sitting there does not clear a breaker in the same cycle that trips it |
| Kill switch | `LIVE_SIGNALS_KILL=1` or a file at `LIVE_SIGNALS_KILL` | Flattens, including adopted positions, and skips every signal. Only the exact value `1` counts. The flag stays until the operator removes it |
| Startup reconciliation | Broker quantities win | Adopted positions are recorded and, by default, **not managed** (`manage_adopted_positions: false`). Orphan open orders whose client id starts with `ww-` are cancelled. Any other open order is left alone |
| Account guard | `PA32ZMSEI`, account type `paper` | Checked in code, not in the config file, before any cancel, replace, or submit. `PA3V3A1X0IX6` is rejected. `log_only` and `live` cannot submit. The Alpaca adapter's only base URL is `https://paper-api.alpaca.markets` |

Keys are read from `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` (or `ALPACA_API_SECRET`) inside `AlpacaPaperBroker.from_env`. They are scrubbed out of exception text and they are not in `__repr__`. The CLI never constructs that class.

## 6. Modes

| Mode | Behavior |
| --- | --- |
| `log_only` | Default. Full decisions, local fills, no broker calls. `python -m live_signals` forces this even if the config file says `paper` |
| `paper` | Submits through the injected broker after the account guard. Tests use `InMemoryBroker`. A real paper session has to construct `AlpacaPaperBroker` on purpose |
| anything else, including `live` | Rejected at config load and again at submission |

## Architecture

```
live_signals/
  default_config.json    every number in this document
  qualify.py             source gate, cooldown, conflict
  sizing.py              conviction, Kelly ceiling, caps
  pricing.py             limit, spread, gap, borrow, cancel/replace
  exits.py               stop, trail, scale-out, time, clock
  events.py              earnings and CPI/FOMC
  risk.py                daily loss, drawdown, kill switch
  planner.py             one pure cycle
  engine.py              log_only simulation or paper submission
  broker.py              protocol, in-memory broker, account guard
  alpaca_broker.py       paper REST adapter, transport-injected
  state.py               SQLite document
  journal.py             decision CSV
  sources.py             JSON file reader and the Wirewatch stub
  notify.py              in-memory hook; no socket
```

Notification hooks are `emit(event, payload)` for `halt`, `take`, and `exit`. `MemoryNotifier` stores them. A mail or Slack sender can implement the same method later. Payloads are scrubbed for key names that look like credentials.

Polling entry point:

```
python -m live_signals --signals signals.json --market market.json --once
python -m live_signals --signals signals.json --market market.json
```

The loop re-reads both files each poll. `--once` is a single cycle.

### Suggested systemd unit

Do not install this as part of the change. It is here for the VPS when the log-only week starts. Paths assume the repo and a dedicated env file that holds `ALPACA_*` only if you have promoted to paper; log-only does not need them.

```
[Unit]
Description=Wirewatch live-signal paper trader (log_only until promoted)
After=network-online.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/algo-ats
EnvironmentFile=-/home/ubuntu/wirewatch/live-signals.env
ExecStart=/home/ubuntu/.local/bin/uv run python -m live_signals \
  --signals /home/ubuntu/wirewatch/signals.json \
  --market /home/ubuntu/wirewatch/market.json \
  --state /home/ubuntu/wirewatch/live_signals_state.db \
  --log /home/ubuntu/wirewatch/live_signals_decisions.csv \
  --kill-file /home/ubuntu/wirewatch/LIVE_SIGNALS_KILL \
  --reset-file /home/ubuntu/wirewatch/LIVE_SIGNALS_RESET_DRAWDOWN
Restart=on-failure
RestartSec=15
NoNewPrivileges=true

[Install]
WantedBy=multi-user.target
```

`signals.json` is the Wirewatch drop. `market.json` is whatever quote/ATR feed is chosen later. Neither file is fetched by this package.

## Rollout

**Days 1–7, log-only.** Run the unit on the VPS. No Alpaca keys. Each morning, read the CSV: how many signals qualified, the skip reasons (especially `single_social_unconfirmed`), the share counts, and whether the same ticker is firing every poll. Compare the takes with what the 8am batch would have done. The batch keeps running on PA32ZMSEI during this week; log-only does not touch it.

**Days 8–14, paper, only if the CSV looks sane.** Switch the engine to `paper` in code (the CLI will not do it) pointed at PA32ZMSEI, and turn the 8am batch off for that account on the same day. Leave `manage_adopted_positions` false so leftover morning-batch positions are recorded and not flattened. Keep the kill file path on the box and the 6-trade / 1.5% daily-loss limits. Review fills against the limits, extended-hours rejects, and short-locate skips.

Promotion past that is a separate decision. It is not this change.

## Open questions for Or

1. Can Wirewatch overwrite a JSON file each scan, or is the dashboard the only output? The engine polls a file. The HTTP stub does not call anything.
2. Where do quote, spread, ATR, shortable, and relative volume come from on the VPS? This package refuses to import the other ATS feed. Alpaca market data is the obvious later source and is not wired here.
3. When paper mode is turned on, should it adopt and manage the positions the 8am routine already has in PA32ZMSEI? The default is to record them and leave them alone, except that a kill switch, a daily-loss halt, or the drawdown breaker will flatten them too.
4. Sector labels and return correlations are optional and currently no-ops without a feed. Is a static sector map enough, or should those caps stay off until there is one?
5. Who supplies the earnings time and the CPI/FOMC clock? The engine will not download a calendar. An empty calendar means those rules never fire.
6. Is "hold overnight only at score ≥ 85, and always flat by Friday 15:50" the policy you want, given the overnight session runs Sunday night through Friday morning?
7. The net cap is 55% of equity, so a one-way book deploys less than today's 90% equal split. Confirm that is the point of the change.
8. A single Reddit post now needs 88, or 78 plus a 0.4% move and 1.5× volume. The GOOGL 70 would be skipped. Is 88 the right bar?
9. The 8am job and this engine must not both send orders to PA32ZMSEI. Confirm the batch is switched off only after the log-only week, on a day you are watching.
