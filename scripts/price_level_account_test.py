"""點位條件放進 v4 帳戶實測（2026-11-27）。逐筆結果見 scripts/price_level_test.py。
營收加速那一份，訊號日收盤（還原價）要符合條件才買：
  A 現在的 v4
  B 收盤在 60 日線上
  C 離一年最高收盤 15% 以內
  D B 且 C
  E 不在 120 日大量區下方 5% 以上（上面沒有大量套牢區）
  F 離一年最高收盤 10% 以內（門檻收緊）
  G 離一年最高收盤 20% 以內（門檻放寬）
一年新高那一份：
  H 離 20 日線 20% 以上的不買（乖離太大）
    python scripts/price_level_account_test.py
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
raw, adj, opn, vol = inputs[:4]
days = raw.index
ma60, ma20 = adj.rolling(60).mean(), adj.rolling(20).mean()
hi = adj.rolling(250, min_periods=120).max()
A, V = adj.values, vol.fillna(0).values


def poc(i, j, w=120):
    c, v = A[max(0, i - w + 1):i + 1, j], V[max(0, i - w + 1):i + 1, j]
    m = np.isfinite(c) & (v > 0)
    if m.sum() < 60:
        return np.nan
    b = np.floor(np.log(c[m]) / np.log(1.02)).astype(int)
    return 1.02 ** (pd.Series(v[m]).groupby(b).sum().idxmax() + 0.5)


MODE = {"m": "A"}
orig_pead, orig_s1 = P.pead.pick, P.s1_pick


def pead_pick(day, sig, ctx, raw_, adj_, vol_, ind, taiex, names):
    c, regime = orig_pead(day, sig, ctx, raw_, adj_, vol_, ind, taiex, names)
    m = MODE["m"]
    if m in "AH" or not len(c):
        return c, regime
    ids = c["stock_id"]
    x = adj.loc[day, ids].values
    up60 = x > ma60.loc[day, ids].values
    dist = x / hi.loc[day, ids].values - 1
    if m == "E":
        i = days.get_loc(day)
        q = np.array([poc(i, adj.columns.get_loc(s)) for s in ids])
        keep = ~(x / q - 1 < -0.05)
    else:
        keep = {"B": up60, "C": dist >= -0.15, "D": up60 & (dist >= -0.15), "F": dist >= -0.10, "G": dist >= -0.20}[m]
    return c[keep].reset_index(drop=True), regime


def s1_pick(day, sig, raw_, adj_, vol_, ind, names):
    g = orig_s1(day, sig, raw_, adj_, vol_, ind, names)
    if MODE["m"] != "H" or not len(g):
        return g
    ids = g["stock_id"]
    return g[(adj.loc[day, ids].values / ma20.loc[day, ids].values - 1) < 0.20].reset_index(drop=True)


P.pead.pick, P.s1_pick = pead_pick, s1_pick
taiex = load_market_index()["taiex"].reindex(days).ffill()
navs, info = {}, {}
for m in "ABCDEFGH":
    MODE["m"] = m
    a = P.run(days[days >= "2023-01-01"], P.Account(), inputs, version="v4")
    n = pd.DataFrame(a.nav)
    navs[m] = n.set_index(pd.to_datetime(n["date"]))["淨值"]
    t = pd.DataFrame(a.trades)
    info[m] = "；".join(f"{k} {len(s)} 筆、勝率 {(s['報酬'] > 0).mean():.0%}、平均 {s['報酬'].mean():+.1%}、中位 {s['報酬'].median():+.1%}"
                       for k, s in t.groupby("strategy") if k != "處置股")
navs = pd.DataFrame(navs)
navs["加權指數"] = taiex.reindex(navs.index)
seg = navs[navs.index >= "2023-04-01"]


def summ(s):
    s = s.dropna() / s.dropna().iat[0]
    return f"{s.iat[-1] - 1:+.1%}／{(s / s.cummax() - 1).min():.1%}"


half = seg.index[len(seg) // 2]
lines = [f"期間 {seg.index[0].date()} ~ {seg.index[-1].date()}（報酬／最大回撤）"]
for k in seg.columns:
    yr = "  ".join(f"{y} {summ(x[k])}" for y, x in seg.groupby(seg.index.year))
    lines.append(f"{k:5s} 全期 {summ(seg[k])}  前半 {summ(seg.loc[seg.index < half, k])}  後半 {summ(seg.loc[seg.index >= half, k])}  {yr}  {info.get(k, '')}")
text = "\n".join(lines)
print(text)
(ROOT / "results" / "規則依據" / "price_level_account.txt").write_text(text, encoding="utf-8")
