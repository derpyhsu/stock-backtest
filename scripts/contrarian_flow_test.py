"""大盤被外資或融資大砍時，逆勢被買的股票（2026-11-26）。另外補測兩個條件：
一年新高依「離兩年低點多遠」分組、長多股回檔到 120 日線與 240 日線之間後帶量站回所有均線。

做法和 handbook_untested_test.py 一樣：訊號日收盤判斷、隔天開盤買（開盤漲停買不到）、
抱 N 個交易日收盤賣，扣 0.6%。比隨機＝每天一票：同一天訊號平均減同一天所有流動性股票平均。

「大盤被砍」只用當天以前的資料判斷：全市場外資 10 日買賣超金額、全市場融資餘額 5 日變動率，
落在自己過去 250 個交易日最低的一成。
    python scripts/contrarian_flow_test.py
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

# 大盤層級：外資 10 日買賣超金額、融資餘額 5 日變動率
mkt_f10 = (fn * raw).sum(axis=1).rolling(10).sum()
mkt_m = mb.sum(axis=1)
mkt_m5 = mkt_m / mkt_m.shift(5) - 1
f_dump = mkt_f10 <= mkt_f10.rolling(250, min_periods=120).quantile(0.1).shift(1)
m_dump = mkt_m5 <= mkt_m5.rolling(250, min_periods=120).quantile(0.1).shift(1)

f10 = fn.rolling(10).sum() / vol.rolling(10).sum()
m5 = (mb - mb.shift(5)) / vol.rolling(5).sum()
r10 = c / c.shift(10) - 1
strong = r10.gt(r10.where(liquid).median(axis=1), axis=0)
f250 = fn.rolling(250).sum() / vol.rolling(250).sum()

sig = {}
fd = pd.DataFrame({k: f_dump for k in raw.columns}, index=days)
md = pd.DataFrame({k: m_dump for k in raw.columns}, index=days)
f_buy = fd & (f10 >= 0.05)
m_buy = md & (m5 >= 0.03)
sig["外資大賣時 個股外資 10 日買超占量 >= 5%"] = first(f_buy, 20)
sig["　外資逆勢買＋年線上揚"] = first(f_buy & a_up, 20)
sig["　外資逆勢買＋比大盤強（10 日漲幅高於中位數）"] = first(f_buy & strong, 20)
sig["　外資逆勢買＋外資過去一年也是買超"] = first(f_buy & (f250 > 0.02), 20)
sig["對照：平常日 個股外資 10 日買超占量 >= 5%"] = first(~fd & (f10 >= 0.05), 20)
sig["融資大減時 個股融資 5 日增加占量 >= 3%"] = first(m_buy, 20)
sig["　融資逆勢增＋年線上揚"] = first(m_buy & a_up, 20)
sig["　融資逆勢增＋比大盤強"] = first(m_buy & strong, 20)
sig["　融資逆勢增＋外資 10 日也買超"] = first(m_buy & (f10 > 0), 20)
sig["對照：平常日 個股融資 5 日增加占量 >= 3%"] = first(~md & (m5 >= 0.03), 20)

# 一年新高，依離兩年低點多遠分組
new_high = c >= c.rolling(250).max()
s1 = first(new_high, 20) & a_up
mult = c / c.rolling(500, min_periods=400).min()
sig["一年新高（20 日內第一次）"] = s1
sig["　離兩年低點不到 1 倍（低基期）"] = s1 & (mult < 2)
sig["　離兩年低點 1~2 倍"] = s1 & (mult >= 2) & (mult < 3)
sig["　離兩年低點 2 倍以上（高基期）"] = s1 & (mult >= 3)

# 長多股回檔到 120 日線與 240 日線之間，之後帶量站回所有均線
zone = a_up & (c < ma[120]) & (c > ma[240])
back = recent(zone, 20) & above_all & vol_up
sig["長多回檔到半年線與年線之間後 帶量站回所有均線"] = first(back, 20)
sig["　回檔站回＋外資過去一年買超占量 > 2%"] = first(back & (f250 > 0.02), 20)

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
t.to_csv(out / "contrarian_flow.csv", index=False, encoding="utf-8-sig")
order = list(sig)
lines = [f"{days[0]:%Y-%m-%d} ~ {days[-1]:%Y-%m-%d}，成本 {COST:.1%}；比隨機＝每天一票的平均超額",
         f"外資大賣日 {int(f_dump.sum())} 天、融資大減日 {int(m_dump.sum())} 天（各取過去 250 日最兇的一成）"]
for title, col, fmt in (("比隨機（每筆超額）", "比隨機", lambda v: f"{v:+.1%}"), ("筆數", "筆數", lambda v: f"{v:.0f}"),
                        ("勝率", "勝率", lambda v: f"{v:.0%}"), ("贏的年數", "贏的年數", str)):
    pv = t.pivot(index="策略", columns="天數", values=col).reindex([o for o in order if o in set(t["策略"])])
    lines += ["", f"== {title}", pv.map(lambda v: fmt(v) if pd.notna(v) else "").to_string()]
lines += ["", "== 60 天逐年超額"]
for _, r in t[t["天數"] == 60].iterrows():
    lines.append(f"{r['策略']}：{r['每年']}（前半 {r['前半']:+.1%}、後半 {r['後半']:+.1%}；訊號 {r['訊號天數']} 天）")
text = "\n".join(lines)
print(text)
(out / "contrarian_flow.txt").write_text(text, encoding="utf-8")
