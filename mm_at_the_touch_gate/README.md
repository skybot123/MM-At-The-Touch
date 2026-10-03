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
