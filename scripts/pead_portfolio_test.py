"""PEAD-TW（月營收加速＋股價已反應，規則 B）放進模擬帳戶回測，和 v1、加權指數比。

訊號日 = 營收可用日（M+1 月 10 號後）第一個交易日；隔天開盤買，同一套帳戶規則（每筆風險 1%、單檔 20%、最多 10 檔、60 天或跌破停損賣）。
當天訊號超過空位時，依營收年增加速度（accel）排序。只用訊號日當天以前的資料。
    python scripts/pead_portfolio_test.py 2022-08-01
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import twscreener.portfolio as P  # noqa: E402
from twscreener.fundamentals import revenue_features  # noqa: E402
from twscreener.market import load_market_index  # noqa: E402

start = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else "2022-08-01")
inputs = P._inputs()
raw, adj, opn, vol = inputs[0], inputs[1], inputs[2], inputs[3]
days = adj.index
taiex = load_market_index()["taiex"].reindex(days).ffill()
rev = revenue_features(pd.read_csv(Path(__file__).resolve().parent.parent / "data/revenue.csv", dtype={"stock_id": str}))
rev = rev.sort_values(["stock_id", "ym"])
rev["yoy_prev3"] = rev.groupby("stock_id")["yoy"].transform(lambda s: s.shift(1).rolling(3).mean())
liquid = (vol.rolling(20).mean() >= 500) & (raw >= 10)
ind = inputs[7]

ev = {}
for ym, g in rev.groupby("ym"):
    after = days[days >= g["available"].iat[0]]
    if len(after) == 0:
        continue
    d = after[0]
    i = days.get_loc(d)
    ms = days[days >= g["available"].iat[0] - pd.Timedelta(days=10)]
    j = days.get_loc(ms[0]) - 1
    if j < 0:
        continue
    ids = [s for s in g["stock_id"] if s in adj.columns and liquid.at[d, s]]
    gg = g.set_index("stock_id").loc[ids]
    react = adj.iloc[i][ids] / adj.iloc[j][ids] - 1 - (taiex.iat[i] / taiex.iat[j] - 1)
    accel = gg["yoy"] - gg["yoy_prev3"]
    m = (gg["yoy"] >= 0.3) & (accel >= 0.1) & gg["rev_high12"].astype(bool) & (react >= 0.05)
    sel = accel[m].sort_values(ascending=False)
    if len(sel):
        ev[d] = sel


def make_pick(stop, skip_bear=False, mix=False):
    orig = P.pick

    def pk(d, *a):
        c, reg = orig(d, *a) if mix else (pd.DataFrame(), None)
        if reg is None:
            reg = a[1].set_index("date")["大盤狀態"].get(d)
        if d in ev and not (skip_bear and reg == "空頭"):
            s = ev[d]
            pe = pd.DataFrame({"stock_id": s.index, "name": s.index, "strategy": "營收加速", "industry": [ind.get(x, "其他") for x in s.index],
                               "score": 1 + s.values, "stop": stop, "close_at_pick": raw.loc[d, s.index].values})
            c = pd.concat([pe, c], ignore_index=True).drop_duplicates("stock_id") if len(c) else pe
        return c, reg
    return pk, orig


def run(pk):
    P.pick = pk
    a = P.run(days[days >= start], P.Account(), inputs, version="v1")
    n = pd.DataFrame(a.nav)
    return n.set_index(pd.to_datetime(n["date"]))["淨值"]


V = {"PEAD 停損-15%": dict(stop=-0.15), "PEAD 停損-20%": dict(stop=-0.20), "PEAD 停損-30%": dict(stop=-0.30),
     "PEAD -20%＋空頭不買": dict(stop=-0.20, skip_bear=True), "v1＋PEAD（-20%）": dict(stop=-0.20, mix=True)}
navs = {}
for k, kw in V.items():
    pk, orig = make_pick(**kw)
    navs[k] = run(pk)
    P.pick = orig
navs["v1"] = run(P.pick)
navs = pd.DataFrame(navs)
navs["加權指數"] = taiex.reindex(navs.index)
navs.to_csv(Path(__file__).resolve().parent.parent / "results/重測_2022/pead_portfolio_navs.csv")
seg = navs[navs.index >= "2023-04-01"]


def summ(s):
    s = s.dropna() / s.dropna().iat[0]
    return f"{s.iat[-1] - 1:+.1%}／{(s / s.cummax() - 1).min():.1%}"


print(f"期間 {seg.index[0].date()} ~ {seg.index[-1].date()}（報酬／最大回撤），PEAD 事件月數 {len(ev)}")
h = seg.index[len(seg) // 2]
for k in seg.columns:
    yr = "  ".join(f"{y} {summ(g[k])}" for y, g in seg.groupby(seg.index.year))
    print(f"{k:18s} 全期 {summ(seg[k])}  前段 {summ(seg[k][seg.index < h])}  後段 {summ(seg[k][seg.index >= h])}  {yr}")
