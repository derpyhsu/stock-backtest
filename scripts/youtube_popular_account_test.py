"""高觀看數影片裡逐筆看起來最好的兩條，放進 v5 模擬帳戶實測（2026-11-27）。逐筆結果見 scripts/youtube_popular_rules_test.py。
  A 現在的 v5
  B 一年新高那份：加 3 倍 ATR 移動停利（停損價每天往上調到 收盤 ×(1−3×ATR%)，只升不降）
  C 一年新高、營收加速兩份都加 3 倍 ATR 移動停利
  D 一年新高那份：再加「多頭排列中第一次回到 20 日線」的訊號
  E 一年新高那份：整份換成「多頭排列中第一次回到 20 日線」
    python scripts/youtube_popular_account_test.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import twscreener.portfolio as P  # noqa: E402
from twscreener.market import load_market_index, load_panel  # noqa: E402

inputs = P._inputs()
raw, adj, opn, vol = inputs[:4]
days = raw.index
p = load_panel()


def piv(col):
    return p.pivot_table(index="date", columns="stock_id", values=col).reindex(index=days, columns=raw.columns)


rh, rl = piv("high"), piv("low")
pc = raw.shift(1)
atr = pd.concat([rh - rl, (rh - pc).abs(), (rl - pc).abs()]).groupby(level=0).max().rolling(14).mean() / raw
ma20, ma60, ma120 = adj.rolling(20).mean(), adj.rolling(60).mean(), adj.rolling(120).mean()
bull = (ma20 > ma60) & (ma60 > ma120)
held = bull.astype(float).rolling(20).min().fillna(0).astype(bool) & (ma60 > ma60.shift(5))
touch = held & (rl <= raw / adj * ma20 * 1.01) & (adj > ma20)
pull = touch & ~touch.astype(float).rolling(20).max().shift(1).fillna(0).astype(bool)
MODE = {"m": "A"}
orig_plan, orig_s1 = P.Account.plan, P.s1_pick
TRAIL = {"B": {"一年新高"}, "C": {"一年新高", "營收加速"}}


def plan(self, day, close, cands, prev=None, trend=None):
    names = TRAIL.get(MODE["m"])
    if names and len(self.pos) and day in atr.index:
        for idx, h in self.pos.iterrows():
            if h["strategy"] not in names:
                continue
            c, a = close.get(h["stock_id"], np.nan), atr.at[day, h["stock_id"]] if h["stock_id"] in atr.columns else np.nan
            if np.isfinite(c) and np.isfinite(a):
                self.pos.at[idx, "stop"] = max(h["stop"], round(c * (1 - 3 * a), 2))
    return orig_plan(self, day, close, cands, prev, trend=trend)


def s1_pick(day, sig, raw_, adj_, vol_, ind, names):
    m = MODE["m"]
    gq = orig_s1(day, sig, raw_, adj_, vol_, ind, names)
    if m not in "DE" or day not in pull.index:
        return gq
    i = days.get_loc(day)
    lq = (vol.iloc[max(0, i - 19):i + 1].mean() >= 500) & (raw.iloc[i] >= 10)
    have = set(gq["stock_id"]) if len(gq) and m != "E" else set()
    ids = [x for x in pull.columns[pull.loc[day].fillna(False)] if lq.get(x, False) and x not in have]
    if not ids or i < 20:
        return pd.DataFrame() if m == "E" else gq
    r20 = (adj.iloc[i][ids] / adj.iloc[i - 20][ids] - 1).dropna().sort_values()
    ids = list(r20.index)
    extra = pd.DataFrame({"stock_id": ids, "name": [names.get(x, x) for x in ids], "strategy": "一年新高",
                          "industry": [ind.get(x, "其他") for x in ids], "score": r20.values, "stop": P.S1_STOP,
                          "close_at_pick": raw.loc[day, ids].values, "sleeve": "s1"})
    if m == "E" or not len(gq):
        return extra
    return pd.concat([gq, extra], ignore_index=True).sort_values("score").reset_index(drop=True)


P.Account.plan, P.s1_pick = plan, s1_pick
taiex = load_market_index()["taiex"].reindex(days).ffill()
navs, info = {}, {}
for m in "ABCDE":
    MODE["m"] = m
    a = P.run(days[days >= "2023-01-01"], P.Account(), inputs, version="v5")
    n = pd.DataFrame(a.nav)
    navs[m] = n.set_index(pd.to_datetime(n["date"]))["淨值"]
    t = pd.DataFrame(a.trades)
    info[m] = "；".join(f"{k} {len(s)} 筆、勝率 {(s['報酬'] > 0).mean():.0%}、平均 {s['報酬'].mean():+.1%}"
                       for k, s in t.groupby("strategy") if k in ("一年新高", "營收加速"))
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
(out / "popular_account.txt").write_text(text, encoding="utf-8")
