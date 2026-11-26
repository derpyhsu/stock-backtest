"""v4 帳戶裡「一年新高」那 1/3 換幾種選法，用 portfolio.run 實際回測比較（2026-11-26）。
來源：scripts/handbook_untested_test.py 每筆比隨機的結果（一年新高＋外資 60 日買超、高檔量縮後帶量突破）。
  A 現在的 v4（同一天多檔時 20 日漲幅小的先買）
  B 只買創高前 60 日外資買超的
  C 外資 60 日買超占成交量大的先買（不過濾）
  D 一年新高 再加「高檔量縮後帶量突破」的訊號
  E D 再只留外資 60 日買超的
  F 一年新高 再加「年線將翻揚、站穩 120 日線＋外資 120 日買超」的訊號
  G F 再加 D 的訊號，外資買超多的先買
    python scripts/s1_sleeve_variants_test.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import twscreener.portfolio as P  # noqa: E402
from twscreener.market import load_market_index, load_panel  # noqa: E402

inputs = P._inputs()
raw, adj, opn, vol = inputs[:4]
days = raw.index
p = load_panel()
fn = p.pivot_table(index="date", columns="stock_id", values="foreign_net").reindex(index=days, columns=raw.columns).fillna(0)
force = fn.rolling(60).sum() / vol.rolling(60).sum()
c = adj
ma240 = c.rolling(240).mean()
a_up = ma240 > ma240.shift(20)
v5, v20, v120 = vol.rolling(5).mean(), vol.rolling(20).mean(), vol.rolling(120).mean()
choke = (v5 <= 0.45 * v120) & (c.rolling(10).max() / c.rolling(10).min() - 1 <= 0.08) & (c >= 0.9 * c.rolling(250).max()) & a_up
s9 = (choke.astype(float).rolling(15).max().shift(1) > 0) & (vol >= 1.5 * v20.shift(1)) & (c > c.shift(1).rolling(20).max())
ma120 = c.rolling(120).mean()
xb = (ma240 < ma240.shift(20)) & (c > c.shift(200).rolling(40).max()) & (c > ma120) & (ma120 > ma120.shift(10)) & (c.rolling(10).min() > ma120.rolling(10).min() * 0.98) & (fn.rolling(120).sum() / vol.rolling(120).sum() > 0.02)
xbc = xb & ~(xb.astype(float).rolling(40).max().shift(1) > 0)
orig = P.s1_pick
MODE = {"m": "A"}


def pick(day, sig, raw_, adj_, vol_, ind, names):
    m = MODE["m"]
    g = orig(day, sig, raw_, adj_, vol_, ind, names)
    extra_sig = {"D": s9, "E": s9, "F": xbc, "G": s9 | xbc}.get(m)
    if extra_sig is not None and day in extra_sig.index:
        i = adj_.index.get_loc(day)
        liq = (vol_.iloc[max(0, i - 19):i + 1].mean() >= 500) & (raw_.iloc[i] >= 10)
        have = set(g["stock_id"]) if len(g) else set()
        ids = [x for x in extra_sig.columns[extra_sig.loc[day].fillna(False)] if liq.get(x, False) and x not in have]
        if ids and i >= 20:
            r20 = (adj_.iloc[i][ids] / adj_.iloc[i - 20][ids] - 1).dropna()
            extra = pd.DataFrame({"stock_id": r20.index, "name": [names.get(x, x) for x in r20.index], "strategy": "一年新高",
                                  "industry": [ind.get(x, "其他") for x in r20.index], "score": r20.values, "stop": P.S1_STOP,
                                  "close_at_pick": raw_.loc[day, r20.index].values, "sleeve": "s1"})
            g = pd.concat([g, extra], ignore_index=True).sort_values("score") if len(g) else extra.sort_values("score")
    if len(g) and m in ("B", "C", "E", "G"):
        f = force.loc[day].reindex(g["stock_id"]).values
        g = g.assign(_f=f)
        g = g[g["_f"] > 0] if m in ("B", "E") else g.sort_values("_f", ascending=False)
        g = g.drop(columns="_f").reset_index(drop=True)
    return g


P.s1_pick = pick
taiex = load_market_index()["taiex"].reindex(days).ffill()
navs, info = {}, {}
for m in "ABCDEFG":
    MODE["m"] = m
    a = P.run(days[days >= "2023-01-01"], P.Account(), inputs, version="v4")
    n = pd.DataFrame(a.nav)
    navs[m] = n.set_index(pd.to_datetime(n["date"]))["淨值"]
    t = pd.DataFrame(a.trades)
    s = t[t["strategy"] == "一年新高"]
    info[m] = f"一年新高那份 {len(s)} 筆、勝率 {(s['報酬'] > 0).mean():.0%}、平均 {s['報酬'].mean():+.1%}"
navs = pd.DataFrame(navs)
navs["加權指數"] = taiex.reindex(navs.index)
seg = navs[navs.index >= "2023-04-01"]


def summ(s):
    s = s.dropna() / s.dropna().iat[0]
    return f"{s.iat[-1] - 1:+.1%}／{(s / s.cummax() - 1).min():.1%}"


lines = [f"期間 {seg.index[0].date()} ~ {seg.index[-1].date()}（報酬／最大回撤）"]
for k in seg.columns:
    yr = "  ".join(f"{y} {summ(g[k])}" for y, g in seg.groupby(seg.index.year))
    lines.append(f"{k:5s} 全期 {summ(seg[k])}  {yr}  {info.get(k, '')}")
text = "\n".join(lines)
print(text)
out = ROOT / "results" / "手冊未測型態"
out.mkdir(parents=True, exist_ok=True)
(out / "s1_sleeve_variants.txt").write_text(text, encoding="utf-8")
