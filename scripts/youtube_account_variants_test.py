"""選股影片的兩個新條件放進 v4 模擬帳戶實際回測（2026-11-27）。逐筆比隨機的結果見 results/選股影片規則/README.md。
  A 現在的 v4
  B 營收加速那份：營收連續第 2 個月以上創歷史新高的先買（不過濾）
  C 營收加速那份：只買連續創歷史新高的
  D 營收加速那份：再加「連續創歷史新高但不在營收加速裡」的（排在後面）
  E 一年新高那份：再加「投信賣轉買＋收紅＋收 5 日最高」的訊號
  F 一年新高那份：整份換成投信賣轉買
  G B＋E
「歷史新高」只看得到 2021-01 之後，而且要有 24 個月以上，所以 2023 年初以前沒有這個條件。
    python scripts/youtube_account_variants_test.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import twscreener.portfolio as P  # noqa: E402
from twscreener.fundamentals import load_revenue, revenue_features  # noqa: E402
from twscreener.market import load_market_index, load_panel  # noqa: E402

inputs = P._inputs()
raw, adj, opn, vol = inputs[:4]
days = raw.index
p = load_panel()


def piv(col):
    return p.pivot_table(index="date", columns="stock_id", values=col).reindex(index=days, columns=raw.columns)


rf = revenue_features(load_revenue())
g = rf.groupby("stock_id")
rf["hi_all"] = (rf["rev"] >= g["rev"].cummax()) & (g.cumcount() + 1 >= 24)
rf["streak"] = rf["hi_all"] & g["hi_all"].shift(1).fillna(False).astype(bool)
pos = days.searchsorted(rf["available"])
rf = rf[(pos < len(days)) & rf["streak"] & rf["stock_id"].isin(raw.columns)].assign(day=lambda x: days[pos[x.index.map(lambda i: rf.index.get_loc(i))]])
streak = {d: set(x["stock_id"]) for d, x in rf.groupby("day")}

tn, ro = piv("trust_net").fillna(0), piv("open")
turn = ((tn.shift(1) < 0) & (tn.shift(2) < 0) & (tn > -(tn.shift(1) + tn.shift(2))) & (raw > ro) & (adj >= adj.rolling(5).max()))
MODE = {"m": "A"}
orig_pead, orig_s1 = P.pead.pick, P.s1_pick


def frame(ids, day, strategy, sleeve, score, stop, ind, names):
    return pd.DataFrame({"stock_id": ids, "name": [names.get(x, x) for x in ids], "strategy": strategy,
                         "industry": [ind.get(x, "其他") for x in ids], "score": score, "stop": stop,
                         "close_at_pick": raw.loc[day, ids].values, "sleeve": sleeve})


def liq(day):
    i = days.get_loc(day)
    return (vol.iloc[max(0, i - 19):i + 1].mean() >= 500) & (raw.iloc[i] >= 10)


def pead_pick(day, sig, ctx, raw_, adj_, vol_, ind, taiex, names):
    m = MODE["m"]
    c, regime = orig_pead(day, sig, ctx, raw_, adj_, vol_, ind, taiex, names)
    st = streak.get(day, set())
    if m in "BCG" and len(c):
        s = c["stock_id"].isin(st)
        c = c[s] if m == "C" else pd.concat([c[s], c[~s]])
        c = c.reset_index(drop=True)
    if m == "D" and st:
        lq = liq(day)
        have = set(c["stock_id"]) if len(c) else set()
        ids = sorted(x for x in st if x not in have and lq.get(x, False))
        if ids:
            extra = frame(ids, day, "營收加速", "v2", 0.0, P.pead.STOP, ind, names).drop(columns="sleeve")
            c = pd.concat([c, extra], ignore_index=True) if len(c) else extra
    return c, regime


def s1_pick(day, sig, raw_, adj_, vol_, ind, names):
    m = MODE["m"]
    gq = orig_s1(day, sig, raw_, adj_, vol_, ind, names)
    if m not in "EFG" or day not in turn.index:
        return gq
    i = days.get_loc(day)
    lq = liq(day)
    have = set(gq["stock_id"]) if len(gq) and m != "F" else set()
    ids = [x for x in turn.columns[turn.loc[day].fillna(False)] if lq.get(x, False) and x not in have]
    if not ids or i < 20:
        return pd.DataFrame() if m == "F" else gq
    r20 = (adj.iloc[i][ids] / adj.iloc[i - 20][ids] - 1).dropna().sort_values()
    extra = frame(list(r20.index), day, "一年新高", "s1", r20.values, P.S1_STOP, ind, names)
    if m == "F" or not len(gq):
        return extra
    return pd.concat([gq, extra], ignore_index=True).sort_values("score").reset_index(drop=True)


P.pead.pick, P.s1_pick = pead_pick, s1_pick
taiex = load_market_index()["taiex"].reindex(days).ffill()
navs, info = {}, {}
for m in "ABCDEFG":
    MODE["m"] = m
    a = P.run(days[days >= "2023-01-01"], P.Account(), inputs, version="v4")
    n = pd.DataFrame(a.nav)
    navs[m] = n.set_index(pd.to_datetime(n["date"]))["淨值"]
    t = pd.DataFrame(a.trades)
    info[m] = "；".join(f"{k} {len(s)} 筆、勝率 {(s['報酬'] > 0).mean():.0%}、平均 {s['報酬'].mean():+.1%}"
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
out = ROOT / "results" / "選股影片規則"
out.mkdir(parents=True, exist_ok=True)
(out / "account_variants.txt").write_text(text, encoding="utf-8")
