# Live signal paper trader

A self-contained replacement for the once-a-day 8am batch on Alpaca paper account **PA32ZMSEI** ("Grok Recs", about $15k). Wirewatch already scores news through the day. This package decides, as each signal arrives, whether it is credible enough to trade, how many shares that conviction is worth, and when the position is done.

Nothing in this package is wired into the existing ATS. That system uses a different paper account (`PA3V3A1X0IX6`). Its pipeline, cron, and data files are not imported and not modified. The only edit outside `live_signals/` is the pytest `testpaths` entry.

This task does not deploy the service and does not send an order. The default mode is `log_only`. There is no live-money mode.

## How a cycle runs

Every 10 minutes (configurable, 5–15):

1. Read a batch of `NewsSignal`s. Quotes and ATR come from a local file when it already has them, otherwise from Alpaca paper data and then Yahoo. Earnings and CPI/FOMC are still an injected list. This package does not call Wirewatch.
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

Market data is a `MarketSnapshot`. If `--market` already has a quote and an ATR for the ticker, that file wins and nothing is fetched. Otherwise the loop asks Alpaca with the paper keys (`data.alpaca.markets` for the quote and daily bars, `paper-api.alpaca.markets` for shortable / easy-to-borrow and the session calendar) and, if that call fails, Yahoo Finance daily bars. Yahoo has no bid/ask and no locate, so the fallback prints the last trade as both sides of the quote and will not open a short. Relative volume is the latest bar divided by the mean of the bars before it. The return versus the previous close is what confirmation uses when no one passed a return in by hand.

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
| Sector | 30% of equity. Sector comes from the snapshot if present, otherwise from the static map in `sectors.py` |
| Correlation cluster | 25% of equity for names with correlation ≥ 0.75, only when a correlation is supplied |
| New risk opened today | 2.0% of equity, summed as `shares * stop_distance`, released if the entry is cancelled unfilled |
| New entries today | 6, not counting exits or the scale-in clip |

The sector map is a checked-in table of liquid US tickers to a GICS-style sector (Technology, Financials, Health Care, and the rest). It exists so two megacap tech names cannot silently become the whole book just because nobody passed a sector on the quote. A ticker that is not in the table is not capped. Correlation still needs a number from outside; a missing pair is not treated as correlated. Net at 55% is deliberate: the morning batch deploys 90% of equity in one direction, and this book will not.

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

Early-close days come from Alpaca's calendar when the feed is used (`close` before 16:00). After-hours then runs four hours, capped at 20:00. A file can still pass the map directly.

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
| Score faded | Flat | Same | Any fresh signal on this ticker with score **< 60**. The position stays open through a neutral 75 or a same-direction 70. This replaces the old neutral-75 invalidation |
| 2% trail | Stop = high-water mark × 0.98 | Mirror, × 1.02 | Used when the entry score is **85 or lower**, including an adopted position with no score. It is active immediately and never loosens the initial stop. A score **above 85** keeps the ATR trail instead |
| Reversal | Flat | Same | A fresh opposite signal that itself passes the source gate and scores at least as high as the position (and at least 65) |
| Flip | After that exit, open the other side | Same | Only if the new score is ≥ **80** and at least **5 points** above the position. Otherwise flatten and stay flat |
| Into the close | Flat from 15:50 ET | Same | Score **85 or lower**. Also any Friday, and any evening whose overnight session will not run (weekend or the night before a holiday), regardless of score |
| Overnight hold | Allowed | Same | Score **strictly above 85**, Sunday evening through Thursday evening, and only when that overnight session exists |
| Time stop / alpha decay | Off | Off | `time_exits_enabled` is false. A high-score position is not closed just for being old; it waits for a score under 60, a stop, or a daily-loss halt |
| Earnings | Flat | Same | From **4 hours** before the print until **30 minutes** after, for that ticker, when an event is supplied |
| Macro, score under 90 | Flat | Same | From **15 minutes** before CPI/FOMC until the print, when an event is supplied |
| Macro, score ≥ 90 | Hold, stop pulled in to 0.75 R | Mirror | Never loosened. New entries are still blocked through 15 minutes after the print |

Exits are priced at the touch (sell the bid, buy the ask) even when the spread is wide. Getting flat matters more than saving the spread. Entries still refuse that spread.

## 5. Risk controls

| Control | Default | What it does |
| --- | --- | --- |
| Daily loss | 1.5% of the equity snapshotted at the first cycle of the ET date | Flattens and blocks new entries until the next ET date. The first cycle of a process does not trip on losses that happened before it was watching |
| Max drawdown | 8% from the persisted equity peak | Flattens and stays halted until `LIVE_SIGNALS_RESET_DRAWDOWN=1` or the reset file is present. The reset is consumed (file deleted, env var popped) and the peak is rebased to current equity. A reset file that is already sitting there does not clear a breaker in the same cycle that trips it |
| Kill switch | `LIVE_SIGNALS_KILL=1` or a file at `LIVE_SIGNALS_KILL` | Flattens, including adopted positions, and skips every signal. Only the exact value `1` counts. The flag stays until the operator removes it |
| Startup reconciliation | Broker quantities win | Adopted positions are managed: they stay open until a fresh score is under 60, the 2% trail, or a halt. Orphan open orders whose client id starts with `ww-` are cancelled. Any other open order is left alone |
| Account guard | `PA32ZMSEI`, account type `paper` | Checked in code, not in the config file, before any cancel, replace, or submit. `PA3V3A1X0IX6` is rejected. `log_only` and `live` cannot submit. The Alpaca adapter's only base URL is `https://paper-api.alpaca.markets` |

Keys are read from `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` (or `ALPACA_API_SECRET`). They are scrubbed out of exception text and they are not in `__repr__`. Log-only still does not submit orders. Omitting `--market` is the path that calls Alpaca for quotes.

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
  marketdata.py          Alpaca quotes/ATR/calendar, Yahoo fallback
  sectors.py             static ticker to sector map
  sources.py             JSON file reader and the Wirewatch stub
  notify.py              in-memory hook; no socket
```

Notification hooks are `emit(event, payload)` for `halt`, `take`, and `exit`. `MemoryNotifier` stores them. A mail or Slack sender can implement the same method later. Payloads are scrubbed for key names that look like credentials.

Polling entry point:

```
python -m live_signals --signals signals.json --once
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

`signals.json` is the Wirewatch drop. Leave `--market` off and the process loads quotes from Alpaca, then Yahoo. A market file, when present, overrides those feeds for any ticker it already contains.

## Rollout

**Days 1–7, log-only.** Run the unit on the VPS. No Alpaca keys. Each morning, read the CSV: how many signals qualified, the skip reasons (especially `single_social_unconfirmed`), the share counts, and whether the same ticker is firing every poll. Compare the takes with what the 8am batch would have done. The batch keeps running on PA32ZMSEI during this week; log-only does not touch it.

**Days 8–14, paper, only if the CSV looks sane.** Switch the engine to `paper` in code (the CLI will not do it) pointed at PA32ZMSEI, and turn the 8am batch off for that account on the same day. Positions already in the account are managed from then on: they stay open until a fresh score is under 60, unless the 2% trail, a stop, or the 1.5% daily-loss halt closes them. Keep the kill file path on the box and the 6-trade cap. Review fills against the limits, extended-hours rejects, and short-locate skips.

Promotion past that is a separate decision. It is not this change.

## Decisions from review

Wirewatch does not have to push. The loop re-reads a JSON file. If the scanner only has a dashboard today, the missing piece is a few lines on the VPS that write the latest signals to that file. This package will not scrape the dashboard.

Quotes, ATR, borrow flags, and the NYSE session calendar come from Alpaca with the paper keys. Yahoo Finance daily bars are the fallback for price and ATR only. Alpaca's corporate-actions feed is splits, dividends, and mergers. It does not include an earnings timestamp or a CPI/FOMC clock, so those rules run only when an event list is passed in. Holidays and early closes do come from `GET /v2/calendar`.

A position stays open until a fresh signal on that ticker scores under 60, or a stop, the 2% trail, the 15:50 rule, or the daily-loss halt says otherwise. Adopted positions from the morning batch are managed under those same rules. Daily loss, the kill switch, and the drawdown breaker still flatten everything.

Overnight is allowed only when the entry score is above 85. At 85 or below, a 2% trail runs during the day and the position is flat from 15:50 ET. Friday 15:50 is flat for every score, because Friday night is not an overnight session.

The 55% net cap and the single-source bar of 88 (or 78 with confirmation) stay. The 8am batch and this engine must not both send orders to PA32ZMSEI. The batch stays on through the log-only week and comes off only when paper mode is pointed at that account.
