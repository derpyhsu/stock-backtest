# 台股選股策略回測系統

每天收盤後抓台股全市場資料，用固定規則篩出值得關注的股票，附上歷史勝率，發佈成手機也能看的看板。用 Codex、Claude Code 或 Cursor 打開這個資料夾都能接手；三邊的 skill 都指向 `skill/` 裡同一份正本。

先讀 `專案總覽.md`：已定案的方向、目前的發現、資料來源、排程與搬家步驟都在那裡。

## Skills

- `skill/daily-update/SKILL.md`：每日更新（抓資料、回測、看板、同步 GitHub）。
- `skill/strategy-backtest/SKILL.md`：新增或修改策略、加基本面條件、驗證是否真的有用。
- `skill/agent-review/SKILL.md`：每晚從候選股查新聞、選出 3~5 檔推薦並存檔。
- `skill/stock-check/SKILL.md`：使用者問個股（好不好、要不要賣、買在某價位怎麼辦）時，用固定規則回測回答，不給心得。
- `skill/commit/SKILL.md`：commit 規範。每次 commit 前讀。

## Boundaries

- 選股和回測只用固定規則（`twscreener/`），agent 不參與初篩：成本高、答案不一致，回測時還會「偷看未來」。agent 只從 `results/candidates_today.csv` 裡挑（消息面複審），推薦紀錄 `results/agent_picks_log.csv` 和 `picks_log.csv` 一樣不可改寫過去的列。
- 資料一律用證交所、櫃買的「全市場一天一檔」批次端點，不要逐檔抓。證交所會限流（HTTP 307 安全頁），程式會自己等待重試；不要同時跑兩支打證交所的程式。
- 回測不能用到訊號當天之後才知道的資訊。月營收在次月 10 號後才能用（`revenue_features` 的 `available`）。
- 策略要和「隨機買股」「買大盤」一起比，60 個交易日勝率與報酬中位數為主要指標。只贏在一段期間的結果不算數。
- `.env`（FinMind token）和任何 token 都不可以 commit、印出或寫進檔案。這個 repo 是公開的。
- 策略手冊原文（專案的 `../strategy/`）是別人的內容，不放進 repo。
- `results/picks_log.csv` 是每天推薦的紀錄，用來事後對帳，不要刪或改寫過去的列。
- 看板只從 `dashboard/template.html` 產生（`python -m twscreener.market all`），不要手改 `dashboard/index.html`。
