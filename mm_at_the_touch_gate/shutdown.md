---
on_kill_switch: keep_all
---
Positions and coins stay as they are; nothing is sold.

Cleanup:
1. Find the running bot: the instance in manage_bots(action="status") whose name
   starts with mm_at_the_touch_gate-gate_touch-a.
2. Stop it: manage_bots(action="stop_bot", bot_name=<that instance name>).
3. Confirm it no longer appears in manage_bots(action="status").
4. Notify "stopped".
