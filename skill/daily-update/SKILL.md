---
name: daily-update
description: 台股全市場每日更新：抓當天行情、除權息、月營收，重跑回測與今日選股，更新看板並同步 GitHub。Use when the user asks to 更新看板、跑今天的選股、補資料，或排程每晚 22:00 觸發時。
---

# daily-update

融資餘額晚上才公布，所以每天台北時間 21:00 以後跑。所有指令在 repo 根目錄執行。

## 步驟

1. `python3 -u -m twscreener.twse`：補到今天。白天抓的檔案沒有 `margin_bal` 欄，最近 5 天內的會自動重抓。休市日存 `.none`。
2. `python3 -m twscreener.adjust`：補除權息。資料往前補到更早的月份時，用 `--start YYYY-MM-01`。
3. `python3 -u -m twscreener.fundamentals revenue`：最新一個月的月營收（OpenAPI），並重試還沒抓到的歷史月份（公開資訊觀測站時好時壞，抓不到就下次再試）。
4. `python3 -m twscreener.market all`：全市場回測、今日掃描、推薦對帳，輸出 `results/` 和 `dashboard/index.html`。接著 `python3 -m twscreener.report` 更新 `results/回測報告.md`（期望值、扣成本、分市況）。
5. 看板只放 GitHub Pages（步驟 6 會更新）。不要用 Artifact 工具發佈：使用者在上班，不想讓它在 App 側邊自動打開。
6. `bash scripts/sync_github.sh`：同步到 GitHub。腳本會擋下 `.env`。GitHub Pages 也從這裡更新（https://derpyhsu.github.io/stock-backtest/dashboard/，手機不用登入就能看）。
7. 照 `skill/agent-review/SKILL.md` 做 agent 複審，選出今日推薦。
7b. `python3 -m twscreener.notify`：把今天的選股推播到 Telegram。需要環境變數 `TELEGRAM_BOT_TOKEN`；沒有就略過。收件人 ID 存在 `.env`。先用 `--dry` 可以只看訊息不送。
8. 背景回補（可中斷續抓，不要和步驟 1 同時跑）：`data/daily` 最早一天晚於 2022-01-03 時跑 `python3 -u -m twscreener.twse --start 2022-01-01 --end 2025-01-19`（2022 是完整空頭年，驗證用）；月營收歷史沒補齊時跑 `python3 -u -m twscreener.fundamentals revenue --start 2021-01 --rounds 8`；估值缺的日子跑 `python3 -u -m twscreener.fundamentals valuation`（不要和抓日資料同時跑）。

## 回報

一兩句話：今天投信連買、動能、型態策略各抓到哪些股票（`results/picks_today.csv`），有錯就說是哪一步、什麼錯。休市日不回報。

## 出錯時

- `HTTP 307（被限流）`：正常，程式會等。連續失敗的那天會在下次補。
- 某天只有上市、沒有上櫃：刪掉那天的 csv 重跑步驟 1。
- 不要為了跑完而跳過驗證或改寫 `results/picks_log.csv` 的舊資料。
