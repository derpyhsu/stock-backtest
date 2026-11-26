"""文章裡三個較細的條件（2026-11-26）：殺盤賣出量占外資一年買超的比例（籌碼有沒有鬆動）、
回檔後第一次與第二次帶量站上全部均線、賺錢不走的融資。

做法和 handbook_untested_test.py 一樣：訊號日收盤判斷、隔天開盤買（開盤漲停買不到）、
抱 N 個交易日收盤賣，扣 0.6%。比隨機＝每天一票：同一天訊號平均減同一天所有流動性股票平均。
    python scripts/mda_fine_conditions_test.py
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
a_up = ma[240] > ma[240].shift(20)
v20 = vol.rolling(20).mean()
vol_up = vol >= 1.5 * v20.shift(1)
recent = lambda s, n: s.astype(float).rolling(n).max().shift(1).fillna(0).astype(bool)  # noqa: E731
first = lambda s, n: s & ~recent(s, n)  # noqa: E731

fn = piv("foreign_net").fillna(0)
mb = piv("margin_bal")
sig = {}

# 6. 長多股從 60 日高點回檔 15% 以上，依外資有沒有鬆動分組
b250 = fn.rolling(250).sum()
s20 = fn.rolling(20).sum()
dd = a_up & (c <= 0.85 * c.rolling(60).max())
big = b250 / vol.rolling(250).sum() > 0.02
keep = big & (s20 > -b250 / 3)
sig["長多回檔 15%（20 日內第一次）"] = first(dd, 20)
sig["　外資一年買超大、近 20 日賣不到一年買超的 1/3"] = first(dd, 20) & keep
sig["　外資一年買超大、近 20 日賣超過 1/3（鬆動）"] = first(dd, 20) & big & ~keep
sig["　外資一年沒有買超"] = first(dd, 20) & ~big

# 7. 回檔後帶量站上全部均線：第一次、第二次
reclaim = a_up & above_all & ~above_all.shift(1).fillna(False).astype(bool) & vol_up & recent(dd, 60)
sig["回檔後第一次帶量站上全部均線"] = reclaim & ~recent(reclaim, 60)
sig["回檔後第二次以上帶量站上全部均線"] = reclaim & recent(reclaim, 60)
sig["　第一次＋外資沒鬆動"] = reclaim & ~recent(reclaim, 60) & keep
sig["　第二次以上＋外資沒鬆動"] = reclaim & recent(reclaim, 60) & keep

# 8. 賺錢不走的融資：120 日融資增加占量 >= 2%，期間股價曾回檔 20% 以上，融資仍在 120 日高點的九成以上
m120 = (mb - mb.shift(120)) / vol.rolling(120).sum()
shaken = (c / c.rolling(120).max()).rolling(60).min() <= 0.8
sticky = (m120 >= 0.02) & shaken & (mb >= 0.9 * mb.rolling(120).max()) & a_up
sig["融資黏著（60 日內第一次）"] = first(sticky, 60)
sig["　融資黏著＋當天帶量站上全部均線"] = sticky & above_all & vol_up & ~recent(sticky & above_all & vol_up, 20)
sig["　融資黏著＋一年新高"] = sticky & (c >= c.rolling(250).max()) & ~recent(sticky & (c >= c.rolling(250).max()), 20)

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
        rows.append({"策略": name, "天數": h, "筆數": len(st), "訊號天數": len(x), "勝率": (st > 0).mean(), "平均": st.mean(),
                     "中位數": st.median(), "比隨機": x.mean(), "前半": x[x.index < half].mean(), "後半": x[x.index >= half].mean(),
                     "贏的年數": f"{(yr > 0).sum()}/{len(yr)}", "每年": " ".join(f"{y % 100}:{v:+.1%}" for y, v in yr.items())})
t = pd.DataFrame(rows)
out = ROOT / "results" / "手冊未測型態"
out.mkdir(parents=True, exist_ok=True)
t.to_csv(out / "mda_fine_conditions.csv", index=False, encoding="utf-8-sig")
order = list(sig)
lines = [f"{days[0]:%Y-%m-%d} ~ {days[-1]:%Y-%m-%d}，成本 {COST:.1%}；比隨機＝每天一票的平均超額"]
for title, col, fmt in (("比隨機（每筆超額）", "比隨機", lambda v: f"{v:+.1%}"), ("筆數", "筆數", lambda v: f"{v:.0f}"),
                        ("勝率", "勝率", lambda v: f"{v:.0%}"), ("贏的年數", "贏的年數", str)):
    pv = t.pivot(index="策略", columns="天數", values=col).reindex([o for o in order if o in set(t["策略"])])
    lines += ["", f"== {title}", pv.map(lambda v: fmt(v) if pd.notna(v) else "").to_string()]
lines += ["", "== 60 天逐年超額"]
for _, r in t[t["天數"] == 60].iterrows():
    lines.append(f"{r['策略']}：{r['每年']}（前半 {r['前半']:+.1%}、後半 {r['後半']:+.1%}；訊號 {r['訊號天數']} 天）")
text = "\n".join(lines)
print(text)
(out / "mda_fine_conditions.txt").write_text(text, encoding="utf-8")
