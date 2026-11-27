"""阿魯米「大盤大跌後最先漲回起跌點＋創 120 日新高」放進 v5 模擬帳戶實測（2026-11-27）。逐筆結果見 scripts/youtube_nocc_rules_test.py。
  A 現在的 v5
  B 一年新高那份：再加這個訊號（大盤從 120 日高點跌過 10%、目前還差 5% 以上）
  C 同 B，但門檻改成跌過 15%、還差 8% 以上
另外印出這個訊號和「創 240 日新高」重疊多少。
    python scripts/youtube_nocc_account_test.py
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
tx = load_market_index()["taiex"].reindex(days).ffill()
W = 120
txv, cv = tx.values, adj.values
pk = np.array([i - W + 1 + int(np.argmax(txv[i - W + 1:i + 1])) if i >= W - 1 else -1 for i in range(len(days))])
ok = pk >= 0
p0 = np.full(cv.shape, np.nan)
p0[ok] = cv[pk[ok]]
txpk = np.where(ok, txv[np.clip(pk, 0, None)], np.nan)
age = np.arange(len(days)) - pk
dd = pd.Series(txv / txpk - 1, index=days)
since_low = pd.Series([txv[pk[i]:i + 1].min() / txpk[i] - 1 if pk[i] >= 0 else np.nan for i in range(len(days))], index=days)
rec = adj > pd.DataFrame(p0, index=days, columns=raw.columns)
hi120 = adj >= adj.rolling(120).max()
hi240 = adj >= adj.rolling(240).max()


def build(a, b):
    m = rec.mul(((since_low <= a) & (dd <= b) & (age >= 10)).values, axis=0).astype(bool)
    return m & ~m.astype(float).rolling(20).max().shift(1).fillna(0).astype(bool) & hi120


SIG = {"B": build(-0.10, -0.05), "C": build(-0.15, -0.08)}
for k, s in SIG.items():
    n = int(s.values.sum())
    print(k, "訊號", n, "筆；同時創 240 日新高", f"{(s & hi240).values.sum() / max(n, 1):.0%}", "；有訊號的天數", int(s.any(axis=1).sum()), flush=True)
MODE = {"m": "A"}
orig_s1 = P.s1_pick


def s1_pick(day, sig, raw_, adj_, vol_, ind, names):
    m = MODE["m"]
    gq = orig_s1(day, sig, raw_, adj_, vol_, ind, names)
    if m not in SIG or day not in days:
        return gq
    i = days.get_loc(day)
    lq = (vol.iloc[max(0, i - 19):i + 1].mean() >= 500) & (raw.iloc[i] >= 10)
    have = set(gq["stock_id"]) if len(gq) else set()
    ids = [x for x in SIG[m].columns[SIG[m].loc[day].fillna(False)] if lq.get(x, False) and x not in have]
    if not ids or i < 20:
        return gq
    r20 = (adj.iloc[i][ids] / adj.iloc[i - 20][ids] - 1).dropna().sort_values()
    ids = list(r20.index)
    extra = pd.DataFrame({"stock_id": ids, "name": [names.get(x, x) for x in ids], "strategy": "一年新高",
                          "industry": [ind.get(x, "其他") for x in ids], "score": r20.values, "stop": P.S1_STOP,
                          "close_at_pick": raw.loc[day, ids].values, "sleeve": "s1"})
    if not len(gq):
        return extra
    return pd.concat([gq, extra], ignore_index=True).sort_values("score").reset_index(drop=True)


P.s1_pick = s1_pick
navs, info = {}, {}
for m in "ABC":
    MODE["m"] = m
    a = P.run(days[days >= "2023-01-01"], P.Account(), inputs, version="v5")
    n = pd.DataFrame(a.nav)
    navs[m] = n.set_index(pd.to_datetime(n["date"]))["淨值"]
    t = pd.DataFrame(a.trades)
    info[m] = "；".join(f"{k} {len(s)} 筆、勝率 {(s['報酬'] > 0).mean():.0%}、平均 {s['報酬'].mean():+.1%}"
                       for k, s in t.groupby("strategy") if k in ("一年新高", "營收加速"))
    print(m, "done", flush=True)
navs = pd.DataFrame(navs)
navs["加權指數"] = tx.reindex(navs.index)
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
(ROOT / "results" / "選股影片規則" / "nocc_account.txt").write_text(text, encoding="utf-8")
