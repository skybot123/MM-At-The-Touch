---
type: generic
description: Order-book-imbalance market maker; one LIMIT_MAKER per side at the touch, posted by a fitted model + solver policy
---
# mm_at_the_touch

- Rests at most ONE LIMIT_MAKER order per side at the best bid/ask. A fitted order-book-imbalance model + solver decides, at every moment, whether each side posts. That policy is the strategy: nothing overrides it.
- SPOT: inventory is measured against `baseline_holding` (coin units that count as q = 0), in units of `order_amount`, inside [`q_min`, `q_max`]. Starting with no coin, it reads itself as below the band and only bids, so it buys its baseline itself at 0% maker once the model is fitted.
- It quotes nothing until each of 5 imbalance regimes has `min_mos_per_regime` buy AND sell market orders and `min_sojourns_per_regime` completed visits (warm-up). No quotes during warm-up is expected, not a failure.
- No candles feed: it uses only its pair's order book and public trades.
- Log: ONE line per pair every `status_log_interval` seconds, for example:
  `mm_at_the_touch pair=FIL-USDT status state=active fit=1 short=0 short_on=- need=3/3/10 regimes=ss:ok,ms:ok,neu:ok,mb:ok,sb:ok err=0 q=+1 post=BA edge_bps=1.40 est_vol_h=9000 vol=950.00 fills=48 live_bid_s=3000 live_ask_s=2800 up_s=3600 vol_h=4000 pnl=-0.4200 upnl=-0.1000 fees=0.0000 age_s=0.4`
  - `fit` = 1 once a policy exists; `short` / `short_on` = regimes still warming up and what they lack. `need` = floor as buys/sells/visits; `regimes` = each regime (ss, ms, neu, mb, sb = strong sell .. strong buy) as buys/sells/visits so far, or `ok` once met, e.g. `neu:1/0/3`.
  - `q` = inventory units vs the baseline. `post` = sides the policy posts now (B, A, BA or -).
  - `vol`, `fills`, `live_bid_s`, `live_ask_s`, `up_s` = session totals since the bot started. `vol_h` = last minute's rate.
  - `pnl` = net incl. unrealized; `fees`. `age_s` = age of the latest order-book snapshot.

Schema has no `candles_*` and no `unwind_*` fields.

## Styles
| style | pair | order_amount | baseline_holding | q range | total_amount_quote |
|---|---|---|---|---|---|
| gtouch_fil | FIL-USDT | 14.8 | 118.4 | -7..7 | 240 |
| gtouch_inj | INJ-USDT | 2.07 | 16.56 | -7..7 | 240 |
| gtouch_zro | ZRO-USDT | 7.6 | 60.8 | -7..7 | 240 |
| gtouch_livetest_fil | FIL-USDT | 5.6 | 16.8 | -2..2 | 30 (standalone smoke test only) |
