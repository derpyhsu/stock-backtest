"""月營收公布後續漲（台股版 PEAD）回測。參考 tradermonty/claude-trading-skills 的 pead-screener 概念。

事件：營收月份 M 的資料在 M+1 月 10 號後才能用。訊號日 = M+1 月 10 號之後第一個交易日收盤，隔天開盤買。
A 營收加速：年增 >= 30%、比前 3 個月平均年增多 10 個百分點以上、創 12 個月新高。
B A＋股價已經反應：M+1 月 1 號到訊號日，股票比加權指數多漲 5% 以上（市場認帳）。
C 只看股價反應（對照組）。基準：同一天所有流動性夠的股票。報酬扣 0.6% 成本。
    python scripts/revenue_drift_test.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from twscreener.backtest import COST  # noqa: E402
from twscreener.fundamentals import revenue_features  # noqa: E402
from twscreener.market import load_market_index, load_panel  # noqa: E402
from twscreener.review import market_context  # noqa: E402
from twscreener.simulate import prices  # noqa: E402

raw, adj, opn, vol = prices(load_panel())
taiex = load_market_index()["taiex"].reindex(adj.index).ffill()
rev = revenue_features(pd.read_csv(Path(__file__).resolve().parent.parent / "data/revenue.csv", dtype={"stock_id": str}))
rev = rev.sort_values(["stock_id", "ym"])
rev["yoy_prev3"] = rev.groupby("stock_id")["yoy"].transform(lambda s: s.shift(1).rolling(3).mean())
liquid = (vol.rolling(20).mean() >= 500) & (raw >= 10)
days = adj.index
rows = []
for ym, g in rev.groupby("ym"):
    avail = g["available"].iat[0]
    after = days[days >= avail]
    if len(after) == 0:
        continue
    d = after[0]
    i = days.get_loc(d)
    if i + 61 >= len(days):
        continue
    month_start = days[days >= avail - pd.Timedelta(days=10)][0]
    j = days.get_loc(month_start) - 1
    if j < 0:
        continue
    ids = [s for s in g["stock_id"] if s in adj.columns and liquid.at[d, s]]
    gg = g.set_index("stock_id").loc[ids]
    entry = opn.iloc[i + 1][ids]
    react = adj.iloc[i][ids] / adj.iloc[j][ids] - 1 - (taiex.iat[i] / taiex.iat[j] - 1)
    out = pd.DataFrame({"date": d, "stock_id": ids, "yoy": gg["yoy"].values, "accel": (gg["yoy"] - gg["yoy_prev3"]).values,
                        "high12": gg["rev_high12"].values, "react": react.values})
    for h in (20, 60):
        out[f"r{h}"] = (adj.iloc[i + h][ids] / entry - 1 - COST).values
    rows.append(out)
d = pd.concat(rows).dropna(subset=["r20", "r60"])
d = d.merge(market_context()[["date", "大盤狀態"]], on="date", how="left")
A = (d["yoy"] >= 0.3) & (d["accel"] >= 0.1) & d["high12"].astype(bool)
B = A & (d["react"] >= 0.05)
C = d["react"] >= 0.05
d["year"] = d["date"].dt.year
cut = d["date"].quantile(0.5)
d["half"] = np.where(d["date"] <= cut, "前", "後")
pd.set_option("display.width", 250)
print(f"事件期間 {d['date'].min():%Y-%m} ~ {d['date'].max():%Y-%m}，{d['date'].nunique()} 個月")


def show(m, name):
    x = d[m]
    t = x.groupby("half")[["r20", "r60"]].mean()
    print(f"{name:10s} 筆數 {len(x):5d}  20日 {x['r20'].mean():+.3f}  60日 {x['r60'].mean():+.3f}  60日勝率 {(x['r60'] > 0).mean():.2f}"
          f"  去前5% {x['r60'][x['r60'] <= x['r60'].quantile(.95)].mean():+.3f}  前段60日 {t.at['前', 'r60']:+.3f}  後段60日 {t.at['後', 'r60']:+.3f}")
    return x


for m, n in [(pd.Series(True, index=d.index), "隨機買股"), (A, "A營收加速"), (B, "B加速+反應"), (C, "C只看反應")]:
    show(m, n)
print("\n逐年 60 日平均（扣成本）：")
print(pd.DataFrame({n: d[m].groupby("year")["r60"].mean() for m, n in [(pd.Series(True, index=d.index), "隨機"), (A, "A"), (B, "B"), (C, "C")]}).round(3))
print("\n分大盤狀態 60 日：")
print(pd.DataFrame({n: d[m].groupby("大盤狀態")["r60"].agg(lambda s: f"{s.mean():+.3f} ({len(s)})") for m, n in [(pd.Series(True, index=d.index), "隨機"), (A, "A"), (B, "B")]}))
print("\n壓力測試（B 的門檻 50%~150%）：")
for k in (0.5, 0.75, 1, 1.25, 1.5):
    m = (d["yoy"] >= 0.3 * k) & (d["accel"] >= 0.1 * k) & d["high12"].astype(bool) & (d["react"] >= 0.05 * k)
    x = d[m]
    print(f"  x{k}: 筆數 {len(x)}，60日 {x['r60'].mean():+.3f}，超額 {x['r60'].mean() - d['r60'].mean():+.3f}")
