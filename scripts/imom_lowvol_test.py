"""日內動能 IMOM 前段班裡改挑波動小的（imom_portfolio_test.py 的追加，2026-11-26）。

波動＝近 20 日日報酬標準差。排序：同一週的 IMOM 前段班，波動由小到大。

IMOM：近 N 天「開盤到收盤」報酬加總，流動性股票中排名前 q；每週最後一個交易日選，隔天開盤買。
帳戶規則和 v2 一樣（每筆風險 1%、單檔 20%、最多 10 檔、停損 -15%、60 天賣、開盤漲停買不到）。
對照「近 20 日漲幅前 10%」（一般動能），看 IMOM 有沒有比單純追漲好。
    python scripts/imom_lowvol_test.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import twscreener.pead as pead  # noqa: E402
import twscreener.portfolio as P  # noqa: E402
from twscreener.market import load_market_index  # noqa: E402

inputs = P._inputs()
raw, adj, opn, vol, raw_open = inputs[:5]
days = raw.index
taiex = load_market_index()["taiex"].reindex(days).ffill()
liquid = (vol.rolling(20).mean() >= 500) & (raw >= 10)
oc = (raw / raw_open - 1).where(lambda x: x.abs() <= 0.11)
wk = pd.Series(days.to_period("W"), index=days)
weekend = set(days[(wk != wk.shift(-1)).values])
ind = inputs[7]


volat = adj.pct_change(fill_method=None).rolling(20).std()


def ranker(score, q, vmax=1.0):
    rk = score.where(liquid).rank(axis=1, pct=True)
    vr = volat.where(liquid).rank(axis=1, pct=True)
    out = {}
    for d in weekend:
        keep = (rk.loc[d] >= q) & (vr.loc[d] <= vmax)
        s = (-volat.loc[d][keep]).dropna().sort_values(ascending=False)
        if len(s):
            out[d] = s
    return out


i20, i40 = oc.rolling(20).sum(), oc.rolling(40).sum()
SIGS = {"IMOM20前10%・低波動先": ranker(i20, 0.9), "IMOM20前20%・低波動先": ranker(i20, 0.8),
        "IMOM40前10%・低波動先": ranker(i40, 0.9), "IMOM20前10%・波動後半不買": ranker(i20, 0.9, 0.5)}
orig = pead.pick


def make(sig, with_v2):
    def pk(day, s, ctx, raw_, adj_, vol_, ind_, tx, names):
        c, reg = orig(day, s, ctx, raw_, adj_, vol_, ind_, tx, names) if with_v2 else (pd.DataFrame(), None)
        if sig is not None and day in sig:
            x = sig[day]
            e = pd.DataFrame({"stock_id": x.index, "name": x.index, "strategy": "IMOM", "industry": [ind.get(k, "其他") for k in x.index],
                              "score": -x.values, "stop": -0.15, "close_at_pick": raw.loc[day, x.index].values})
            c = pd.concat([c, e], ignore_index=True).drop_duplicates("stock_id") if len(c) else e
        return c, reg
    return pk


start = pd.Timestamp("2022-08-01")
navs, trades = {}, {}
V = {"v2": (None, True)}
for k, s in SIGS.items():
    V[k] = (s, False)
    V["v2＋" + k] = (s, True)
for k, (s, w) in V.items():
    pead.pick = make(s, w)
    a = P.run(days[days >= start], P.Account(), inputs, version="v2")
    n = pd.DataFrame(a.nav)
    navs[k] = n.set_index(pd.to_datetime(n["date"]))["淨值"]
    trades[k] = len(a.trades) if hasattr(a, "trades") else np.nan
    print(k, "done", flush=True)
pead.pick = orig
navs = pd.DataFrame(navs)
navs["加權指數"] = taiex.reindex(navs.index)
out = ROOT / "results" / "網路策略"
navs.to_csv(out / "imom_lowvol_navs.csv")


def summ(s):
    s = s.dropna() / s.dropna().iat[0]
    return f"{s.iat[-1] - 1:+.1%}／{(s / s.cummax() - 1).min():.1%}"


lines = []
for st in ("2022-08-01", "2023-04-01"):
    seg = navs[navs.index >= st]
    h = seg.index[len(seg) // 2]
    lines.append(f"\n期間 {seg.index[0].date()} ~ {seg.index[-1].date()}（報酬／最大回撤）")
    for k in seg.columns:
        yr = "  ".join(f"{y} {summ(g[k])}" for y, g in seg.groupby(seg.index.year))
        lines.append(f"{k:24s} 全期 {summ(seg[k])}  前段 {summ(seg[k][seg.index < h])}  後段 {summ(seg[k][seg.index >= h])}  {yr}  交易 {trades.get(k, '')}")
text = "\n".join(lines)
print(text)
(out / "imom_lowvol.txt").write_text(text, encoding="utf-8")
