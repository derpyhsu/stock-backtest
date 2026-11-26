"""2026-11-26 重讀 M大文章後整理出、用日資料就能測的新條件。做法同 horizon_matrix_test.py：
訊號日收盤判斷、隔天開盤買（開盤漲停買不到）、抱 N 日收盤賣，扣 0.6%；比隨機＝每天一票的同日超額。
    python scripts/mda_factors_test.py
"""
import sys
from pathlib import Path

import numpy as np
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


ro, lo = piv("open"), piv("low")
days = raw.index
liquid = (vol.rolling(20).mean() >= 500) & (raw >= 10)
can_buy = ro.shift(-1) < raw * 1.095
entry = opn.shift(-1)
fwd = {h: adj.shift(-h) / entry - 1 - COST for h in H}
base = {h: fwd[h].where(liquid).mean(axis=1) for h in H}
c = adj
f = adj / raw
low = lo * f
ma = {n: c.rolling(n).mean() for n in (5, 20, 60, 120, 240)}
v5, v20, v60 = vol.rolling(5).mean(), vol.rolling(20).mean(), vol.rolling(60).mean()
a_up = ma[240] > ma[240].shift(20)
up = c > c.shift(1)
B = lambda x: x.fillna(False).astype(bool)  # noqa: E731
rec = lambda s, n: B(s).astype(float).rolling(n).max().shift(1) > 0  # noqa: E731
first = lambda s, n: B(s) & ~rec(s, n)  # noqa: E731
sig = {}

# 1 底底高（格友的 M哥分數）：60 日切三段，每段最低點墊高；站上月線季線、季線上揚、半年線不彎；量增
l1, l2, l3 = low.rolling(20).min().shift(40), low.rolling(20).min().shift(20), low.rolling(20).min()
hl = (l3 > l2) & (l2 > l1)
trend = (c > ma[20]) & (c > ma[60]) & (ma[60] > ma[60].shift(10)) & (ma[120] >= ma[120].shift(10) * 0.995)
volup = (B(vol >= 2.5 * v60.shift(1)).astype(float).rolling(15).max() > 0) | (v5 >= 1.5 * v60.shift(5))
sig["底底高＋站上月季線＋量增"] = first(hl & trend & volup, 20)
sig["底底高＋站上月季線（不看量）"] = first(hl & trend, 20)

# 2 20 日扣抵量 SOP：長多、回檔後量縮價穩不破低、20 日均量正要扣掉大量、5 日均量上穿 20 日均量且當天上漲
cross = (v5 > v20) & (v5.shift(1) <= v20.shift(1))
stable = c.rolling(10).min() > c.rolling(40).min() * 1.0001
pull = c < c.rolling(60).max() * 0.92
deduct = vol.shift(20).rolling(5).mean() > 1.3 * v20
sig["扣抵量：長多回檔後均量黃金交叉"] = a_up & pull & stable & cross & up & (v5.shift(3) < 0.8 * v20.shift(3))
sig["扣抵量：同上＋正在扣抵大量"] = sig["扣抵量：長多回檔後均量黃金交叉"] & deduct.shift(3)
sig["（對照）任何均量黃金交叉且上漲"] = cross & up

# 3 法人愛不完：一年內外資多數日子在買、累計買超占成交量高；在量縮價穩後轉強時進
fn = piv("foreign_net").fillna(0)
tn = piv("trust_net").fillna(0)
buy_days = (fn > 0).rolling(250).mean()
f250 = fn.rolling(250).sum() / vol.rolling(250).sum()
love = (buy_days >= 0.55) & (f250 >= 0.03)
quiet = (v5 <= 0.6 * v60) & (c.rolling(10).max() / c.rolling(10).min() - 1 <= 0.08)
sig["外資愛不完（狀態第一天）"] = first(love, 60)
sig["外資愛不完＋量縮價穩後轉強"] = love & rec(quiet, 10) & cross & up
sig["外資愛不完＋位階不高"] = love & rec(quiet, 10) & cross & up & (c <= c.rolling(250).min() * 1.6)
t250 = tn.rolling(250).sum() / vol.rolling(250).sum()
sig["投信愛不完＋量縮價穩後轉強"] = ((tn > 0).rolling(250).mean() >= 0.35) & (t250 >= 0.02) & rec(quiet, 10) & cross & up

# 4 外資不論晴雨一直買＋大錢停在窄區間（外資版又熱又平又擠）
narrow = c.rolling(250).max() / c.rolling(250).min() <= 1.6
sig["外資一年大買＋股價窄區間＋帶量突破"] = (f250 >= 0.05) & narrow.shift(1) & (vol >= 1.5 * v20.shift(1)) & (c > c.shift(1).rolling(60).max())

# 5 XBC／左側：年線還在下彎，但 240 日扣抵價未來 40 天都低於現價、站穩上揚的 120 日線、外資 120 日買超
d240 = c.shift(200).rolling(40).max()  # t-240 ~ t-201，未來 40 天要扣掉的價
xbc = (ma[240] < ma[240].shift(20)) & (c > d240) & (c > ma[120]) & (ma[120] > ma[120].shift(10)) & (c.rolling(10).min() > ma[120].rolling(10).min() * 0.98)
f120 = fn.rolling(120).sum() / vol.rolling(120).sum()
sig["XBC：年線將翻揚＋站穩 120 日線"] = first(xbc, 40)
sig["XBC＋外資 120 日買超"] = first(xbc & (f120 > 0.02), 40)

# 6 底部爆大量＋站上大量區：年線下彎的股票出現兩年最大量，之後 60 日內收盤站上那天的高點
hi = piv("high") * f
big = (ma[240] < ma[240].shift(20)) & (c < ma[240]) & (vol >= vol.rolling(500, min_periods=400).max()) & (vol >= 4 * v60.shift(1))
bh = hi.where(big).ffill(limit=60)
sig["底部兩年大量"] = big
sig["底部大量後站上大量區"] = (c > bh) & (c.shift(1) <= bh.shift(1)) & ~big

# 7 慣性改變：以前出量後破底，這次出量後 10 日不破底且底部墊高（空頭或整理中）
spike = vol >= 2.5 * v60.shift(1)
sp10 = B(spike).shift(10).fillna(False)
hold = c.rolling(10).min() >= low.shift(10) * 0.99
sig["出量後 10 日不破底（年線下）"] = sp10 & hold & (c.shift(10) < ma[240].shift(10)) & (l3 > l2)

half = days[len(days) // 2]
rows = []
for name, s in sig.items():
    m = (B(s) & liquid & can_buy).fillna(False)
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
t.to_csv(out / "mda_factors.csv", index=False, encoding="utf-8-sig")
order = [o for o in sig if o in set(t["策略"])]
lines = [f"{days[0]:%Y-%m-%d} ~ {days[-1]:%Y-%m-%d}，成本 {COST:.1%}；比隨機＝每天一票的平均超額"]
for title, col, fmt in (("比隨機（每筆超額）", "比隨機", lambda v: f"{v:+.1%}"), ("筆數", "筆數", lambda v: f"{v:.0f}"),
                        ("勝率", "勝率", lambda v: f"{v:.0%}"), ("贏的年數", "贏的年數", str)):
    pv = t.pivot(index="策略", columns="天數", values=col).reindex(order)
    lines += ["", f"== {title}", pv.map(lambda v: fmt(v) if pd.notna(v) else "").to_string()]
lines += ["", "== 60 天逐年超額"]
for _, r in t[t["天數"] == 60].iterrows():
    lines.append(f"{r['策略']}：{r['每年']}（前半 {r['前半']:+.1%}、後半 {r['後半']:+.1%}）")
text = "\n".join(lines)
print(text)
(out / "mda_factors.txt").write_text(text, encoding="utf-8")
