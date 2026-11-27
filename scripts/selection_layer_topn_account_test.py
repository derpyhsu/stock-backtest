"""選股條件（熱門、大盤好、主流類股領頭）用在一年新高那份，放進 v5 模擬帳戶實測（2026-11-27）成交值名次換不同數字的穩健性檢查見本檔。。逐筆結果見 scripts/selection_layer_test.py。
先印一年新高訊號加上各條件後的逐筆 60 天結果，再跑帳戶：
  A 現在的 v5（同一天多檔時 20 日漲幅小的先買）
  B 一年新高：成交值前 100 名的先買
  C 一年新高：成交值前 100 名而且是主流類股領頭的先買
  D 一年新高：只買成交值前 100 名的
  E 一年新高：只在大盤好（指數在月線上且月線上揚）的日子買
  F 一年新高：再加「熱門＋大盤好＋主流領頭＋近 20 天創過一年新高」進池第一天的股票，而且排最前面
  G 同 F，但不排前面（照原本 20 日漲幅小的先買）
    python scripts/selection_layer_account_test.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import twscreener.portfolio as P  # noqa: E402
from twscreener.market import load_market_index  # noqa: E402

inputs = P._inputs()
raw, adj, opn, vol, raw_open, sig0 = inputs[:6]
days = raw.index
c = adj
liquid = (vol.rolling(20).mean() >= 500) & (raw >= 10)
tx = load_market_index()["taiex"].reindex(days).ffill()
txma = tx.rolling(20).mean()
mkt = (tx > txma) & (txma > txma.shift(5))
vrank = (raw * vol).rank(axis=1, ascending=False)
hot = vrank <= 100
ind = pd.Series(P.load_industry()).reindex(raw.columns).fillna("其他")
r20 = (c / c.shift(20) - 1).where(liquid)
ir = r20.T.groupby(ind.values).mean().T
top = (ir.rank(axis=1, ascending=False) <= 5).reindex(columns=ind.values).set_axis(raw.columns, axis=1).fillna(False).astype(bool)
lead = top & (r20.T.groupby(ind.values).rank(ascending=False).T <= 3)
hi240 = (c >= c.rolling(240).max()).astype(float).rolling(20).max().fillna(0).astype(bool)
pool = hot & lead & hi240 & liquid
pool = pool.mul(mkt.values, axis=0).astype(bool)
newin = pool & ~pool.astype(float).rolling(20).max().shift(1).fillna(0).astype(bool)

# 一年新高訊號逐筆（60 天、扣 0.6%、比同一天流動性股票平均）
fwd = adj.shift(-60) / opn.shift(-1) - 1 - 0.006
base = fwd.where(liquid).mean(axis=1)
s1 = sig0[sig0["strategy"] == "S1_一年新高"].copy()
s1 = s1[s1["date"].isin(days) & s1["stock_id"].isin(raw.columns)]
di, ci = days.get_indexer(s1["date"]), raw.columns.get_indexer(s1["stock_id"])
s1["ret"] = fwd.values[di, ci] - base.values[di]
for nm, arr in (("hot", hot.values), ("lead", lead.values)):
    s1[nm] = arr[di, ci]
s1["mkt"] = mkt.values[di]
s1 = s1.dropna(subset=["ret"])
lines = ["== 一年新高訊號逐筆（60 天比隨機）"]
for nm, m in (("全部", slice(None)), ("成交值前 100", s1["hot"]), ("不在前 100", ~s1["hot"]), ("主流類股領頭", s1["lead"]), ("不是領頭", ~s1["lead"]),
              ("大盤好", s1["mkt"]), ("大盤不好", ~s1["mkt"]), ("前 100＋領頭", s1["hot"] & s1["lead"]), ("前 100＋領頭＋大盤好", s1["hot"] & s1["lead"] & s1["mkt"])):
    x = s1.loc[m]
    yr = x.groupby(x["date"].dt.year)["ret"].mean()
    lines.append(f"{nm:14s} {len(x):5d} 筆 平均 {x['ret'].mean():+.1%} 中位 {x['ret'].median():+.1%} 贏的年數 {(yr > 0).sum()}/{len(yr)}  " + " ".join(f"{y % 100}:{v:+.1%}" for y, v in yr.items()))
print("\n".join(lines), flush=True)


NS = {"D50": 50, "D150": 150, "D200": 200, "D300": 300, "D100均": 100, "B200": 200, "B300": 300}
avgv = (raw * vol).rolling(20).mean().rank(axis=1, ascending=False)
MODE = {"m": "A"}
orig_s1 = P.s1_pick


def s1_pick(day, sig, raw_, adj_, vol_, ind_, names):
    m = MODE["m"]
    gq = orig_s1(day, sig, raw_, adj_, vol_, ind_, names)
    if m == "A" or day not in days or not len(gq):
        return gq
    rk = avgv if m.endswith("均") else vrank
    h = gq["stock_id"].map(lambda x: bool(rk.at[day, x] <= NS[m]) if x in rk.columns else False)
    if m[0] == "D":
        return gq[h.values].reset_index(drop=True)
    return gq.assign(_p=(~h).astype(int).values).sort_values(["_p", "score"]).drop(columns="_p").reset_index(drop=True)


P.s1_pick = s1_pick
navs, info = {}, {}
for m in ["A"] + list(NS):
    MODE["m"] = m
    a = P.run(days[days >= "2023-01-01"], P.Account(), inputs, version="v5")
    n = pd.DataFrame(a.nav)
    navs[m] = n.set_index(pd.to_datetime(n["date"]))["淨值"]
    t = pd.DataFrame(a.trades)
    info[m] = "；".join(f"{k} {len(s)} 筆、勝率 {(s['報酬'] > 0).mean():.0%}、平均 {s['報酬'].mean():+.1%}"
                       for k, s in t.groupby("strategy") if k in ("一年新高", "營收加速"))
    print(m, "done", flush=True)
navs = pd.DataFrame(navs)
seg = navs[navs.index >= "2023-04-01"]


def summ(s):
    s = s.dropna() / s.dropna().iat[0]
    return f"{s.iat[-1] - 1:+.1%}／{(s / s.cummax() - 1).min():.1%}"


half = seg.index[len(seg) // 2]
lines = [f"== 一年新高只買（D）或先買（B）成交值前 N 名，N 換不同數字；「均」是用 20 日平均成交值排名。期間 {seg.index[0].date()} ~ {seg.index[-1].date()}（報酬／最大回撤）"]
for k in seg.columns:
    yr = "  ".join(f"{y} {summ(x[k])}" for y, x in seg.groupby(seg.index.year))
    lines.append(f"{k:6s} 全期 {summ(seg[k])}  前半 {summ(seg.loc[seg.index < half, k])}  後半 {summ(seg.loc[seg.index >= half, k])}  {yr}  {info.get(k, '')}")
text = "\n".join(lines)
print(text)
(ROOT / "results" / "選股層" / "account_topN.txt").write_text(text, encoding="utf-8")
