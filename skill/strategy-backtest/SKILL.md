---
name: strategy-backtest
description: 新增或修改選股策略、加入基本面或大盤濾網條件，並用全市場回測驗證是否真的比隨機買股和大盤好。Use when the user asks to 加策略、調參數、測某個條件有沒有用，或問某策略的勝率。
---

# strategy-backtest

## 程式在哪

- 策略規則：`twscreener/strategies.py`（`STRATEGIES` 字典，每個函式吃 `df, mkt` 回傳布林序列；`MIN_HIST` 設最少歷史天數）。
- 指標：`twscreener/features.py`。基本面：`twscreener/fundamentals.py`（估值 `data/valuation/`、月營收 `data/revenue.csv`）。
- 回測與看板：`twscreener/market.py`。出場一律用「三破」（`backtest.simulate_exit`）。

## 驗證流程

1. 先寫成規則，再跑 `python3 -m twscreener.market backtest`。
2. 和 `基準_隨機進場`、`基準_大盤指數` 比 60 日勝率與報酬中位數，看 `results/market_summary.csv`（共同期間）和 `market_summary_long.csv`（長期間）。
3. 看 `walk_forward`（每季勝率）：只贏在一兩季的，不算有效。
4. 用「假裝三個月前」驗收：只用當時看得到的資料選股，對照之後的實際報酬。
5. 樣本少於約 30 筆的結果，回報時要說樣本太少。
6. 除了勝率，也要報**期望值**（勝率 × 平均賺 − 敗率 × 平均賠）、平均賺賠比、最大虧損（`max_dd_60d`）。勝率高但期望值為負的不算好策略。
7. 報酬要**扣掉交易成本**：一進一出約 0.6%（手續費 0.1425% × 2 + 證交稅 0.3%），換股越頻繁影響越大。
8. 要看**不同市況**：多頭、回檔、急跌（例如 2024-08、2025-04、2026-10 起）、空頭（2022）分開列，至少涵蓋一次完整的多空循環。

## 規則

- 不能用到訊號日之後的資訊：月營收用 `available`（次月 10 號後），估值用當天檔案。
- 不要為了讓數字好看而針對歷史反覆調門檻；調整要能用常識解釋，並在新期間再驗一次。
- 結論和數字寫進 `專案總覽.md` 第 3 節，註明期間與樣本數。都是歷史統計，不保證未來。
