# mm_at_the_touch_gate

At-the-touch market maker on **Gate spot** (FIL, INJ, ZRO) with an **$800 USDT** account. A fitted order-book model
decides when each side quotes; the Condor agent keeps the bot running and reports on it.

**Before either quickstart:** copy this folder into Condor's `agents/` (keep the folder name) and connect a Gate spot
account (`gate_io`) holding USDT. No coins are needed; the bot buys its own.

---

## Quickstart: test

A small live test on FIL-USDT (about $6 orders, under $30 of capital). Paste into a Condor chat:

```
Run the live test for agent mm_at_the_touch_gate. Sync its controller mm_at_the_touch to the server (if the server
copy differs, replace it with the agent's file), upload sample config gtouch_livetest_fil under the same name, and
deploy it on its own as bot mm_at_the_touch_gate-gate_touch-livetest with max_global_drawdown_quote 30 and
max_controller_drawdown_quote 30. Then show me its newest status line (log messages containing " status state=").
Do not change any config or place any order. Whenever I ask for status, show the newest status line again.
```

Approve the one deploy confirmation, then watch. **To stop:** `Stop the bot mm_at_the_touch_gate-gate_touch-livetest.`

---

## Quickstart: competition

The full run: 3 pairs, about $720 of the $800. Stop the test bot first. Paste into a Condor chat:

```
Start the loop mm_at_the_touch_gate.gate_touch in loop mode with its default config.
```

Approve the one start confirmation. From then on it needs nothing: it deploys the bot, keeps it running, redeploys it
if it ever disappears, and sends an hourly summary. **To stop:** `Shut down the mm_at_the_touch_gate agent.` This
stops the loop and the bot. The coins stay in the account; nothing is sold.

---

## What to expect

- **First 30-90 minutes: little or no quoting.** Each pair's model warms up (`fit=0` in the status line), then buys
  its starting coin with **bids only**. Both are normal, not failures.
- **Then both sides quote as the model decides.** It may sit a side out; that's the policy working.
- **Only a pair with no status line for an hour is a problem** (no market data reaching it).

## If it isn't trading

Ask Condor, e.g. *"Ask the mm_at_the_touch_gate agent why the bot isn't quoting."* The agent reads each pair's
status line and the bot's logs. What the status line means:

| Status line shows | What's blocking |
|---|---|
| No status line at all | Bot not running or crashed. Check the bot's error logs. |
| `fit=0 short_on=mos` | Warm-up: waiting for market orders (see below). Normal. |
| `fit=0 short_on=visits` | Warm-up: waiting for the book to move in and out of a regime. Normal. |
| `err=1` | A refit failed. The last good policy keeps quoting; the "refit failed" log line says why. |
| `fit=1 post=-` | The policy sees no edge right now, so it isn't quoting. The model working, not a fault. |
| `fit=1 post=B`, `q` well below 0 | Buying its starting coin (bids only). Normal at the start. |
| `post=BA` but `live_bid_s` / `live_ask_s` not growing | Orders are being rejected (balance, minimum size or rate limits). Check the error logs. |
| Quoting, but `fills` stays 0 | Orders are live but nobody is trading against them yet. |
| `age_s` over ~30 | The order-book feed has stalled. |

**Warm-up needs**, per pair, in each of 5 order-book regimes: **10 buy and 10 sell market orders** (prints within
50 ms count as one) and **10 visits** to that regime. That's at least 100 market orders in total. It usually takes
longer because the rarest regime sets the pace.

**Useful questions:**
- "What does each pair's latest status line say, and what is each one waiting for?"
- "Is each pair fitted yet? If not, how many regimes is it short, and short of what?"
- "Is the policy quoting both sides on each pair right now? If not, why?"
- "Over the last hour, what share of the time has each side had an order resting?"
- "Any errors in the bot's logs: rejected orders, insufficient balance, rate limits, failed refits?"
- "Has each pair finished buying its starting coin (is `q` near 0)?"
- "What are each pair's volume, fills and PnL so far?"
- "Is the bot running? What did the loop journal on its last few ticks?"

## Reference

| Config | Pair | Order | Range | Coin held |
|---|---|---|---|---|
| gtouch_fil | FIL-USDT | 14.8 FIL (~$16) | ±7 orders | 118.4 FIL |
| gtouch_inj | INJ-USDT | 2.07 INJ (~$16) | ±7 orders | 16.56 INJ |
| gtouch_zro | ZRO-USDT | 7.6 ZRO (~$16) | ±7 orders | 60.8 ZRO |
| gtouch_livetest_fil | FIL-USDT | 5.6 FIL (~$6) | ±2 orders | 16.8 FIL |

- **Fees:** 0% maker, 0.058% taker (it only places maker orders).
- **Capital per pair:** (2 × range + 1) × order size: the coin held plus USDT to buy up to the top of the range.
  3 × 15 × $16 = $720.
- **Requirements:** `hummingbot/hummingbot:latest` (verified; Python 3.12+), with numpy, pandas and scipy, all in the
  image. No candles feed, no extra files. Controller type **generic**, installed by the loop to
  `bots/controllers/generic/mm_at_the_touch.py`.
- **Logs:** one line per pair per minute, e.g. `mm_at_the_touch pair=FIL-USDT status fit=1 q=+1 post=BA vol=950.00
  pnl=-0.42 …`. `fit` = model ready, `q` = inventory vs the coin held, `post` = sides quoting, `vol` / `pnl` =
  session totals. All fields are in CONTROLLER.md.
- **Limits:** keeps its market history in memory, sized for about 48 hours per deploy. The API keeps ~100 log lines
  per bot, so a pair can briefly show "no reading".

```
mm_at_the_touch_gate/
├── README.md, AGENT.md, shutdown.md
├── controllers/mm_at_the_touch/   mm_at_the_touch.py, CONTROLLER.md, sample_configs/*.yml
└── loops/gate_touch/loop.md
```
