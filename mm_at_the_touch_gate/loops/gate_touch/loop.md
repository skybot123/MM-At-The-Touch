---
name: Gate Touch
description: Keeps one mm_at_the_touch bot running FIL, INJ and ZRO on Gate spot and
  reports on it. Never interferes with the model's quoting.
agent_key: null
skills: []
default_config:
  frequency_sec: 1800
  execution_mode: loop
  bot_mode: bot
  total_amount_quote: 800
  max_ticks: 0
  restart_on_boot: true
  risk_limits:
    max_position_size_quote: 900
    max_open_executors: 30
default_trading_context: ''
---

## What this loop does
Keep one bot running the 3 pairs and report on it. The model decides all quoting; you never interfere with it.

## Unattended (overrides everything else)
From the moment it is started until I say stop, this loop needs no human:
- Never stop, pause, shut down or exit the loop yourself. Never ask a question, request confirmation or wait for a reply. Notifications are information only.
- If anything fails (a tool call, a deploy, a log read), journal it and carry on; try again next tick, every tick, for as long as it keeps failing.
- If the bot disappears for any reason, redeploy it (step 1). The new bot reads the coins already in the account and carries on from there.
- Never gate anything on the account balance. If it falls below the capital figure, keep trading.

NS = mm_at_the_touch_gate-gate_touch (this loop's bot namespace). Bot NS-a runs gtouch_fil, gtouch_inj, gtouch_zro. A deployed bot runs under its requested name plus a "-YYYYMMDD-HHMMSS" suffix: resolve its instance name every tick from manage_bots(action="status") as the running bot whose name starts with "NS-a". (A bot named NS-livetest is a separate manual smoke test: ignore it, never touch it, never deploy it.)

## Each tick
1. If NS-a has no running instance:
   (If the manage_bots status call itself fails, the instance is UNKNOWN, not absent: journal it and end the tick without deploying.)
   a. manage_agent_controllers(action="status", name="mm_at_the_touch"): missing -> sync. Drift -> this playbook explicitly authorises replacing the server copy with this agent's file: run sync without overwrite (the preview), then immediately sync with overwrite=true. Unreachable -> retry next tick.
   b. Upload the 3 sample configs unchanged: manage_agent_controllers(action="upload_config", name="mm_at_the_touch", sample="<id>", config_name="<id>", overwrite=true).
   c. manage_bots(action="deploy", bot_name="NS-a", controllers_config=["gtouch_fil","gtouch_inj","gtouch_zro"], max_global_drawdown_quote=800, max_controller_drawdown_quote=800). (The platform requires a drawdown figure; it is set to the whole account on purpose.) If it fails, retry next tick.
   d. Notify that the bot was deployed, and end the tick.
   Never deploy a bot that already has a running instance.
2. Read the bot's general logs with run_code (client.bot_orchestration.get_bot_status(<instance>)["data"]["general_logs"]), keep messages containing " status state=", and take the newest line per pair=. A pair with no line this tick is simply reported as "no reading".
3. Journal one line per pair: fit, short, q, post, vol, vol since last tick, fills, pnl, fees, age_s.
4. Once an hour, notify a summary: per pair vol and pnl, combined volume since the last summary, total pnl. Give ZRO-USDT's pnl against its vol on its own line. Record the time of each summary with a journal "state" entry; send the next one once at least 3600 seconds have passed since it.

You never call update_config, never change any field, and never place, cancel or close orders.
