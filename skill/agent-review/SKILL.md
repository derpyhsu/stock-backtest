---
name: agent-review
description: 每晚從規則篩出的候選股中，查新聞與重大訊息，選出 3~5 檔推薦並附上數據、新聞和風險，存檔供日後驗收。Use after the daily update finishes, or when the user asks 今天推薦哪幾檔、幫我評估這幾檔。
---

# agent-review

每日更新（`skill/daily-update`）跑完後做。用 Sonnet 即可。

## 步驟

0. **只用最新一個交易日收盤後的資料**：先確認 `data/daily/` 最新一天就是今天（交易日 14:30 後）；不是就先跑 `python3 -u -m twscreener.twse` 和 `python3 -m twscreener.market all`。推薦適用於隔天開盤。
1. `python3 -m twscreener.oversold`（恐慌誤殺分數），再 `python3 -m twscreener.candidates` → `results/candidates_today.csv`。
2. **只能從候選表裡挑**，不可以推薦表外的股票。表裡的數字（歷史勝率、期望值、樣本數）照抄，不要自己估。
3. 先看大盤狀態欄位。歷史期望值為負的來源（例如空頭時的動能）整批略過，在否決表寫一行原因。
4. 依「排序分數」由高往低，最多查 12 檔的新聞（WebSearch，近 30 天）：營收、法說、重大訊息、處置股、產業報價。同一族群（例如記憶體）先查一次族群新聞再決定整批去留。
5. 選 **3~5 檔**，盡量不同產業；理由不足寧可少選。否決的也要記一句原因。判斷原則：
   - 規則分數高但新聞顯示基本面正在轉壞（報價反轉、獲利衰退、下修展望）→ 否決。
   - 處置股、全額交割、創新板（一般投資人不能買）→ 否決。
   - 本益比低要分辨是「被錯殺」還是「景氣循環高峰」。
   - **催化劑分級**（參考 claude-trading-skills 的 episodic pivot）：每檔在理由開頭標【強】【中】【弱】。
     - 強：上修財測、營收或獲利首次大幅加速、大客戶或大訂單、法規／藥證核准。
     - 中：新產品（還沒看到銷售）、合作案、併購、產品漲價。
     - 弱：只有題材、法人調升目標價、消息模糊。只有「弱」的不推薦，除非數據特別好且寫明理由。
     - 扣分：增資稀釋、可轉債、調查、訴訟、高層離職、下修評等。
     - 自問：這是一天的新聞，還是會改變營收獲利的事？市場是不是早就知道了？股價是不是已經漲很多？
6. 寫入：
   - `results/agent_picks_log.csv`：每檔一列（pick_date, signal_date, stock_id, name, industry, source, close, hist_win, hist_score, verdict 推薦/否決, reason, risk, news 連結, reviewer）。同一天重跑時先刪掉當天的舊列再寫；**過去的列不可修改**。
   - `results/agent_picks/YYYY-MM-DD.md`：給人看的版本，每檔「數據／新聞（附連結）／風險」，加否決表和一段大盤背景。
7. `python3 -m twscreener.levels`：幫推薦算停損價和 30/60/90/120 日目標價 → `results/agent_levels_today.csv`，也把這些價位寫進當天的 md。
   - 停損＝同組歷史訊號中，最後 60 日賺錢的那些在持有期間最深跌幅的第 20 百分位（八成賺錢的交易沒跌破）。
   - 停損至少要留「近 20 日平均單日波動 × 3」，恐慌盤時會比歷史統計寬。
   - 目標＝同組該期間報酬中位數；樂觀＝第 75 百分位。樣本不足（策略 30 筆、恐慌誤殺 100 筆）就改用同來源全部訊號，並在依據欄註明。
8. `python3 -m twscreener.notify` 會把推薦（含停損、目標）放在推播最前面。

## 短線（持有 5～10 天）

9. `python3 -m twscreener.shortterm today` → `results/shortterm_today.csv`：只有「連跌反彈」「超跌反彈」（拉回買）的候選，最多 15 檔、同產業最多 2 檔，已附停損價、5/10 日停利價和歷史勝率。大盤「多頭回檔」時不出候選，當天就不推短線。
10. 從表裡挑 **0~3 檔**，不同產業。查新聞（近 7 天）分辨「為什麼跌」：
   - 隨大盤或族群恐慌一起跌、公司本身營收／獲利沒變壞 → 可以推薦。
   - 跌是因為公司自己的利空（營收衰退、下修展望、財報不佳、掏空、處置股）→ 否決。
   - 正是這波賣壓的核心族群、還在急殺 → 傾向否決或寫明要更小部位。
   - 停損超過 -15% 的要在風險欄提醒部位要更小。
11. 寫入 `results/shortterm_picks_log.csv`（pick_date, signal_date, stock_id, name, industry, rule, close, stop, tp5, tp10, hist_win10, hist_mean10, verdict, reason, risk, news, reviewer），價位照抄候選表；同一天重跑先刪當天舊列，**過去的列不可修改**。在當天的 md 加一段「短線」。notify 會把它放在中長線推薦後面。
12. 對帳：`python3 -m twscreener.shortterm check`（5/10 日後報酬 vs 同日隨機買股），每月健檢時看；累積 2 個月以上再判斷新聞複審有沒有幫助。

## 風控關卡（參考 day-trader-skill 的 trader-risk）

- 每檔都要有停損價；沒有的不推。
- 報酬風險比＝（樂觀停利 − 收盤）÷（收盤 − 停損）。不到 1 的，推播會標示「部位再減半」；在風險欄說明為什麼還值得。
- 同一晚中長線＋短線合計，全部打到停損的總虧損不超過資金 6%（每檔 1% 計算，最多 6 檔）。
- 同一產業同時最多 2 檔（中長線＋短線合計）。

## 驗收

每月健檢（`twscreener/review.py`）比較三組 60 日報酬：agent 推薦 vs 候選表排序前 3 名 vs 隨機買股；也比較「否決」的後續表現。累積 3 個月以上才下結論。
