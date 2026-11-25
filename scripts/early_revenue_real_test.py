"""營收提早進場，用實際公布日（data/revenue_dates.csv，FinMind create_time，2026-05 起才有）。

事件：營收加速（年增 >= 30%、比前 3 月平均多 10 點、創 12 個月新高、流動性夠）。
提早：公布日「隔一個交易日」開盤買（公布時間可能在盤後，保守一天）。
現在：11 號後第一個交易日的隔天開盤買（v2 的時間點）。
比較：(1) 提早買到 v2 買點之間多賺的漲幅；(2) 各自抱 20 個交易日；都和同日隨機買股比，扣 0.6% 成本。
    python scripts/early_revenue_real_test.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from twscreener.backtest import COST  # noqa: E402
from twscreener.fundamentals import revenue_features  # noqa: E402
from twscreener.market import load_panel  # noqa: E402
from twscreener.simulate import prices  # noqa: E402

raw, adj, opn, vol = prices(load_panel())
days = adj.index
liquid = (vol.rolling(20).mean() >= 500) & (raw >= 10)
rev = revenue_features(pd.read_csv(ROOT / "data/revenue.csv", dtype={"stock_id": str})).sort_values(["stock_id", "ym"])
rev["prev3"] = rev.groupby("stock_id")["yoy"].transform(lambda s: s.shift(1).rolling(3).mean())
ev = rev[(rev["yoy"] >= 0.3) & (rev["yoy"] - rev["prev3"] >= 0.1) & rev["rev_high12"].astype(bool) & (rev["ym"] >= "2026-04")]
dates = pd.read_csv(ROOT / "data/revenue_dates.csv", dtype=str).dropna()
dates["announced"] = pd.to_datetime(dates["announced"])
ev = ev.merge(dates, on=["stock_id", "ym"], how="inner")
# 只留合理的公布日：次月 1～15 號（更早的是補登資料的日期）
m1 = pd.to_datetime(ev["ym"] + "-01") + pd.DateOffset(months=1)
ev = ev[(ev["announced"] >= m1) & (ev["announced"] <= m1 + pd.Timedelta(days=14))].copy()
H = 20
rows = []
for r in ev.itertuples():
    if r.stock_id not in adj.columns:
        continue
    a = days[days > r.announced]
    m1_ = pd.Timestamp(r.ym + "-01") + pd.DateOffset(months=1)
    v2d = days[days >= m1_ + pd.Timedelta(days=10)]
    if len(a) == 0 or len(v2d) == 0:
        continue
    ie, iv = days.get_loc(a[0]), days.get_loc(v2d[0]) + 1  # 提早：公布日後第一個交易日開盤；v2：訊號日隔天開盤
    if iv + H >= len(days) or not liquid.iat[ie - 1, adj.columns.get_loc(r.stock_id)]:
        continue
    s = r.stock_id
    pe, pv = opn.iat[ie, adj.columns.get_loc(s)], opn.iat[iv, adj.columns.get_loc(s)]
    be = (opn.iloc[iv] / opn.iloc[ie] - 1)[liquid.iloc[ie - 1]].mean()
    re = adj.iat[ie + H - 1, adj.columns.get_loc(s)] / pe - 1 - COST
    rv = adj.iat[iv + H - 1, adj.columns.get_loc(s)] / pv - 1 - COST
    bre = (adj.iloc[ie + H - 1] / opn.iloc[ie] - 1 - COST)[liquid.iloc[ie - 1]].mean()
    brv = (adj.iloc[iv + H - 1] / opn.iloc[iv] - 1 - COST)[liquid.iloc[iv - 1]].mean()
    rows.append({"ym": r.ym, "stock_id": s, "公布日": r.announced.date(), "提早天數": iv - ie,
                 "提早到v2之間": pv / pe - 1, "同期隨機": be, "提早20日": re, "提早隨機": bre, "v2時點20日": rv, "v2隨機": brv})
t = pd.DataFrame(rows)
t = t[t["提早天數"] > 0]
lines = [f"事件 {len(t)} 筆（營收月份 {t['ym'].min()} ~ {t['ym'].max()}），平均提早 {t['提早天數'].mean():.1f} 個交易日",
         f"提早買到 v2 買點之間：平均 {t['提早到v2之間'].mean():+.2%}（同期隨機 {t['同期隨機'].mean():+.2%}），中位數 {t['提早到v2之間'].median():+.2%}",
         f"抱 20 日：提早買 {t['提早20日'].mean():+.2%}（比隨機 {(t['提早20日'] - t['提早隨機']).mean():+.2%}），"
         f"v2 時點 {t['v2時點20日'].mean():+.2%}（比隨機 {(t['v2時點20日'] - t['v2隨機']).mean():+.2%}）", "", "每月："]
for ym, g in t.groupby("ym"):
    lines.append(f"{ym}：{len(g)} 筆，提早 {g['提早天數'].mean():.1f} 天，提早到 v2 之間 {g['提早到v2之間'].mean():+.2%}（隨機 {g['同期隨機'].mean():+.2%}），"
                 f"20 日比隨機 提早 {(g['提早20日'] - g['提早隨機']).mean():+.2%} / v2 {(g['v2時點20日'] - g['v2隨機']).mean():+.2%}")
lines.append("\n公布日分布（次月幾號）：" + str(pd.to_datetime(t["公布日"]).dt.day.value_counts().sort_index().to_dict()))
text = "\n".join(lines)
print(text)
(ROOT / "results" / "重測_2022" / "early_revenue_real.txt").write_text(text, encoding="utf-8")
