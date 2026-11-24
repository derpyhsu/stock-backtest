"""籌碼類策略（T1 投信連買、S5 逆勢籌碼）vs 同日隨機買股：60 日勝率、平均、超額，逐年、前後段、每月一票。
報酬取 results/market_signals.csv 的 ret_60d（隔天開盤進場），扣成本 0.6%；隨機 = 同日所有流動性合格股票的 60 日平均（同樣扣成本）。
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from twscreener.market import load_panel  # noqa: E402
from twscreener.simulate import prices  # noqa: E402
ROOT = Path(__file__).resolve().parent.parent
raw, adj, opn, vol = prices(load_panel())
liquid = (vol.rolling(20).mean() >= 500) & (raw >= 10)
f60 = opn.shift(-61) / opn.shift(-1) - 1        # 隔天開盤進場、60 日後
f60 = adj.shift(-61) / opn.shift(-1) - 1
base = f60.where(liquid).mean(axis=1) - 0.006
bwin = (f60.where(liquid) > 0.006).sum(axis=1) / liquid.sum(axis=1)
s = pd.read_csv(ROOT / "results/market_signals.csv", dtype={"stock_id": str}, parse_dates=["signal_date"])
s = s[s["strategy"].isin(["T1_投信連買", "S5_逆勢籌碼"])].dropna(subset=["ret_60d"])
s["r"] = s["ret_60d"] - 0.006
s["b"] = s["signal_date"].map(base)
s["x"] = s["r"] - s["b"]
s["年"] = s["signal_date"].dt.year
s["月"] = s["signal_date"].dt.to_period("M")
mid = s["signal_date"].min() + (s["signal_date"].max() - s["signal_date"].min()) / 2
s["段"] = np.where(s["signal_date"] < mid, "前段", "後段")
def tab(g):
    mm = g.groupby("月")["x"].mean()
    return pd.Series({"件數": len(g), "勝率": f"{(g['r'] > 0).mean():.0%}", "平均": f"{g['r'].mean():+.1%}", "隨機": f"{g['b'].mean():+.1%}",
                      "超額": f"{g['x'].mean():+.1%}", "每月一票超額": f"{mm.mean():+.1%}", "贏的月份": f"{(mm > 0).mean():.0%}"})
pd.set_option("display.width", 200)
for k, g in s.groupby("strategy"):
    print(f"\n== {k}（{g['signal_date'].min().date()} ~ {g['signal_date'].max().date()}，前後段分界 {mid.date()}）")
    print(pd.concat({"全期": tab(g), **{str(y): tab(gg) for y, gg in g.groupby("年")}, **{p: tab(gg) for p, gg in g.groupby("段")}}, axis=1).T.to_string())
    new = g[g["signal_date"] <= "2024-05-31"]
    if len(new):
        print("新補資料期間（~2024-05）：", tab(new).to_dict())
