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
7a-0. `python3 scripts/fetch_disposition.py --update`：抓今年的處置股公告到 `data/disposition.csv`（證交所、櫃買各一次查詢；要等步驟 1 的證交所抓取結束才跑，不要同時打證交所）。模擬帳戶 v4 的處置股那一份靠它，**一定要在下一步之前跑完**；失敗就重試，還是失敗要在回報裡講明今天處置股沒更新。
7a-1. `python3 scripts/fetch_tdcc.py`：抓集保股權分散表最新一週到 `data/tdcc/`（集保只給最新一週，要每週累積；同一週已存就自動略過）。失敗不影響其他步驟，回報裡提一句就好。
7a. `python3 -m twscreener.portfolio daily`：模擬帳戶（v6：營收加速 30%、一年新高 30%、處置股 40%；處置股第 2 天開盤買，另加處置跌深）（完全照規則，不經人手）執行今天開盤的委託、檢查停損與到期、排明天的委託，存在 `results/paper/`。滿 60 天的持股收盤還在 60 日線上就續抱（最多 120 天）；每檔續抱或賣出的依據寫在 `results/paper/hold_review.csv`，回報時把「滿 60 天續抱」和「明天賣」的幾檔連同依據列出來。推播的模擬帳戶那則會自動加「明天留意」（離停損不到 5%、再 5 天內滿 60 或 120 天、續抱中離 60 日線不到 3%、已賺 30% 以上）。不要手動改裡面的檔案。
7a. `python3 scripts/orb_test.py fetch`：累積成交最大 300 檔的 5 分 K 到 `data/intraday/`（Yahoo 只保留 60 天，每天抓才不會斷）。當沖策略目前都沒過回測，資料是為了日後重測；失敗就略過，不影響其他步驟。
7c. 每月 15 號後第一次更新：`python3 scripts/fetch_revenue_dates.py --refresh`（背景跑，FinMind 免費帳號每小時 600 次），累積營收實際公布日到 `data/revenue_dates.csv`，給 2027-01 檢討「營收提早進場」用。
7b. `python3 -m twscreener.notify`：把今天的選股推播到 Telegram。需要環境變數 `TELEGRAM_BOT_TOKEN`；沒有就略過。收件人 ID 存在 `.env`。先用 `--dry` 可以只看訊息不送。
8. 背景回補（可中斷續抓，不要和步驟 1 同時跑）：`data/daily` 最早一天晚於 2022-01-03 時跑 `python3 -u -m twscreener.twse --start 2022-01-01 --end 2025-01-19`（2022 是完整空頭年，驗證用）；月營收歷史沒補齊時跑 `python3 -u -m twscreener.fundamentals revenue --start 2021-01 --rounds 8`；估值缺的日子跑 `python3 -u -m twscreener.fundamentals valuation`（不要和抓日資料同時跑）。

## 回報

一兩句話：今天投信連買、動能、型態策略各抓到哪些股票（`results/picks_today.csv`），有錯就說是哪一步、什麼錯。休市日不回報。

## 出錯時

- `HTTP 307（被限流）`：正常，程式會等。連續失敗的那天會在下次補。
- 某天只有上市、沒有上櫃：刪掉那天的 csv 重跑步驟 1。
- 不要為了跑完而跳過驗證或改寫 `results/picks_log.csv` 的舊資料。

9. 最後一步（2021 年補齊前才需要）：`timeout 5400 python3 -u scripts/backfill_daily.py`，把 2021 年日線補到 `data/backfill_daily/`（Derpy 2026-11-26 要求補完；用來讓帳戶回測涵蓋 2022 年空頭）。一定要等前面所有打證交所的步驟都結束才跑，背景跑即可，不用等它。印出「2021 已補齊」後照該腳本開頭的說明併回 `data/daily/`、重算還原價、重跑帳戶回測，並把這一步從這裡刪掉。
