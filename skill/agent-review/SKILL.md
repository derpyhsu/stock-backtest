---
name: agent-review
description: 每晚從規則篩出的候選股中，查新聞與重大訊息，選出 3~5 檔推薦並附上數據、新聞和風險，存檔供日後驗收。Use after the daily update finishes, or when the user asks 今天推薦哪幾檔、幫我評估這幾檔。
---

# agent-review

每日更新（`skill/daily-update`）跑完後做。用 Sonnet 即可。

## 步驟

1. `python3 -m twscreener.oversold`（恐慌誤殺分數），再 `python3 -m twscreener.candidates` → `results/candidates_today.csv`。
2. **只能從候選表裡挑**，不可以推薦表外的股票。表裡的數字（歷史勝率、期望值、樣本數）照抄，不要自己估。
3. 先看大盤狀態欄位。歷史期望值為負的來源（例如空頭時的動能）整批略過，在否決表寫一行原因。
4. 依「排序分數」由高往低，最多查 12 檔的新聞（WebSearch，近 30 天）：營收、法說、重大訊息、處置股、產業報價。同一族群（例如記憶體）先查一次族群新聞再決定整批去留。
5. 選 **3~5 檔**，盡量不同產業；理由不足寧可少選。否決的也要記一句原因。判斷原則：
   - 規則分數高但新聞顯示基本面正在轉壞（報價反轉、獲利衰退、下修展望）→ 否決。
   - 處置股、全額交割、創新板（一般投資人不能買）→ 否決。
   - 本益比低要分辨是「被錯殺」還是「景氣循環高峰」。
6. 寫入：
   - `results/agent_picks_log.csv`：每檔一列（pick_date, signal_date, stock_id, name, industry, source, close, hist_win, hist_score, verdict 推薦/否決, reason, risk, news 連結, reviewer）。同一天重跑時先刪掉當天的舊列再寫；**過去的列不可修改**。
   - `results/agent_picks/YYYY-MM-DD.md`：給人看的版本，每檔「數據／新聞（附連結）／風險」，加否決表和一段大盤背景。
7. `python3 -m twscreener.levels`：幫推薦算停損價和 30/60/90/120 日目標價 → `results/agent_levels_today.csv`，也把這些價位寫進當天的 md。
   - 停損＝同組歷史訊號中，最後 60 日賺錢的那些在持有期間最深跌幅的第 20 百分位（八成賺錢的交易沒跌破）。
   - 目標＝同組該期間報酬中位數；樂觀＝第 75 百分位。樣本不足（策略 30 筆、恐慌誤殺 100 筆）就改用同來源全部訊號，並在依據欄註明。
8. `python3 -m twscreener.notify` 會把推薦（含停損、目標）放在推播最前面。

## 驗收

每月健檢（`twscreener/review.py`）比較三組 60 日報酬：agent 推薦 vs 候選表排序前 3 名 vs 隨機買股；也比較「否決」的後續表現。累積 3 個月以上才下結論。
