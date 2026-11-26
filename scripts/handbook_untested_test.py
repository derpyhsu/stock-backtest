"""手冊裡還沒測過的型態（2026-11-26）：N 年大量、高檔量縮不跌、窒息量平台與突破、均線糾結突破、
一年新高再加「推升力量沒撤」。做法和 horizon_matrix_test.py 一樣：

訊號日收盤判斷、隔天開盤買（開盤漲停買不到）、抱 N 個交易日收盤賣，扣 0.6%。
比隨機＝每天一票：同一天訊號平均減同一天所有流動性股票平均。流動性：20 日均量 >= 500 張、股價 >= 10。
資料 2022-01 起，所以「N 年大量」只能測 N=2、3，而且訊號從 2024、2025 才開始。
    python scripts/handbook_untested_test.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from twscreener.market import load_panel  # noqa: E402
from twscreener.simulate import prices  # noqa: E402

H = (5, 10, 30, 60, 90, 120)
COST = 0.006
p = load_panel()
raw, adj, opn, vol = prices(p)


def piv(col):
    return p.pivot_table(index="date", columns="stock_id", values=col).reindex(index=raw.index, columns=raw.columns)


ro = piv("open")
days = raw.index
liquid = (vol.rolling(20).mean() >= 500) & (raw >= 10)
can_buy = ro.shift(-1) < raw * 1.095
entry = opn.shift(-1)
fwd = {h: adj.shift(-h) / entry - 1 - COST for h in H}
base = {h: fwd[h].where(liquid).mean(axis=1) for h in H}

c = adj
ma = {n: c.rolling(n).mean() for n in (5, 10, 20, 60, 120, 240)}
above_all = pd.concat([(c > m) for m in ma.values()], axis=0).groupby(level=0).all().reindex(days)
a_up = ma[240] > ma[240].shift(20)  # A：240 日線往上
v5, v20, v120 = vol.rolling(5).mean(), vol.rolling(20).mean(), vol.rolling(120).mean()
hi250 = c.rolling(250).max()
rng10 = c.rolling(10).max() / c.rolling(10).min() - 1
choke = (v5 <= 0.45 * v120) & (rng10 <= 0.08)  # B2：量縮到極小、價不跌
recent = lambda s, n: s.rolling(n).max().shift(1).fillna(0).astype(bool)  # noqa: E731
vol_up = vol >= 1.5 * v20.shift(1)
brk20 = c > c.shift(1).rolling(20).max()
new_high = c >= hi250
first_high = new_high & ~recent(new_high.astype(float), 20)

sig = {}
sig["S1 一年新高（20 日內第一次）"] = first_high & a_up
sig["S9 高檔量縮不跌"] = choke & (c >= 0.9 * hi250) & a_up & ~recent(choke.astype(float), 10)
sig["S9 後帶量突破"] = recent((choke & (c >= 0.9 * hi250) & a_up).astype(float), 15) & vol_up & brk20
sig["B2 窒息量平台（長多）"] = choke & a_up & (c > ma[240]) & ~recent(choke.astype(float), 10)
sig["B2 後帶量站上所有均線"] = recent((choke & a_up).astype(float), 15) & vol_up & above_all & brk20
spread = pd.concat(ma.values(), axis=0).groupby(level=0).max().reindex(days) / pd.concat(ma.values(), axis=0).groupby(level=0).min().reindex(days) - 1
box = (c.rolling(250).max() / c.rolling(250).min() <= 1.4)
sig["S4 均線糾結後帶量突破"] = recent((spread <= 0.06).astype(float), 10) & box.shift(1).fillna(False) & vol_up & brk20 & above_all
for n_year in (2, 3):
    w = 250 * n_year
    v20sum = vol.rolling(20).sum()
    nyr = v20sum >= v20sum.rolling(w, min_periods=w).max()
    sig[f"S8 {n_year} 年大量＋一年新高"] = nyr & new_high & ~recent((nyr & new_high).astype(float), 60)
    waited = recent((nyr & new_high).astype(float), 120) & ~recent((nyr & new_high).astype(float), 20)
    sig[f"S8 {n_year} 年大量後量縮再突破"] = waited & recent(choke.astype(float), 20) & vol_up & brk20

# 一年新高，依創高前 60 日哪股力量在買分組（B1：推升力量）
cols = set(p.columns)
force = {}
for k, col in (("外資", "foreign_net"), ("投信", "trust_net")):
    if col in cols:
        force[k] = piv(col).fillna(0).rolling(60).sum() / vol.rolling(60).sum()
mcol = next((x for x in ("margin_balance", "margin_bal", "margin") if x in cols), None)
if mcol:
    mb = piv(mcol)
    force["融資"] = (mb - mb.shift(60)) / vol.rolling(60).sum()
s1 = sig["S1 一年新高（20 日內第一次）"]
for k, f in force.items():
    sig[f"S1＋{k} 60 日買超占量 >= 3%"] = s1 & (f >= 0.03)
    sig[f"S1＋{k} 60 日賣超"] = s1 & (f < 0)
if "外資" in force and "投信" in force:
    sig["S1＋外資投信都買"] = s1 & (force["外資"] > 0) & (force["投信"] > 0)

half = days[len(days) // 2]
rows = []
for name, s in sig.items():
    m = (s & liquid & can_buy).fillna(False)
    for h in H:
        r = fwd[h].where(m)
        st = r.stack().dropna()
        if len(st) < 30:
            continue
        x = r.sub(base[h], axis=0).mean(axis=1).dropna()
        yr = x.groupby(x.index.year).mean()
        rows.append({"策略": name, "天數": h, "筆數": len(st), "勝率": (st > 0).mean(), "平均": st.mean(), "中位數": st.median(),
                     "比隨機": x.mean(), "前半": x[x.index < half].mean(), "後半": x[x.index >= half].mean(),
                     "贏的年數": f"{(yr > 0).sum()}/{len(yr)}", "每年": " ".join(f"{y % 100}:{v:+.1%}" for y, v in yr.items())})
t = pd.DataFrame(rows)
out = ROOT / "results" / "手冊未測型態"
out.mkdir(parents=True, exist_ok=True)
t.to_csv(out / "handbook_untested.csv", index=False, encoding="utf-8-sig")
order = list(sig)
lines = [f"{days[0]:%Y-%m-%d} ~ {days[-1]:%Y-%m-%d}，成本 {COST:.1%}；比隨機＝每天一票的平均超額；力量欄位：{list(force)}"]
for title, col, fmt in (("比隨機（每筆超額）", "比隨機", lambda v: f"{v:+.1%}"), ("筆數", "筆數", lambda v: f"{v:.0f}"),
                        ("勝率", "勝率", lambda v: f"{v:.0%}"), ("贏的年數", "贏的年數", str)):
    pv = t.pivot(index="策略", columns="天數", values=col).reindex([o for o in order if o in set(t["策略"])])
    lines += ["", f"== {title}", pv.map(lambda v: fmt(v) if pd.notna(v) else "").to_string()]
lines += ["", "== 60 天逐年超額"]
for _, r in t[t["天數"] == 60].iterrows():
    lines.append(f"{r['策略']}：{r['每年']}（前半 {r['前半']:+.1%}、後半 {r['後半']:+.1%}）")
text = "\n".join(lines)
print(text)
(out / "handbook_untested.txt").write_text(text, encoding="utf-8")
