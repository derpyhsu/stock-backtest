"""所有做多訊號 × 持有天數（5、10、30、60、90、120）的對照表，找每個長度最適合的策略（2026-11-26）。

訊號日收盤判斷、隔天開盤買（開盤漲停買不到）、抱 N 個交易日收盤賣，扣 0.6%。
比隨機＝每天一票：同一天訊號平均減同一天所有流動性股票平均。流動性：20 日均量 >= 500 張、股價 >= 10。
「贏的年數」：每年平均超額 > 0 的年數／有訊號的年數。資料 2022-01 起（規則策略 2023 起才有訊號）。
    python scripts/horizon_matrix_test.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import twscreener.pead as pead  # noqa: E402
from twscreener.market import load_market_index, load_panel  # noqa: E402
from twscreener.simulate import prices  # noqa: E402

H = (5, 10, 30, 60, 90, 120)
COST = 0.006
p = load_panel()
raw, adj, opn, vol = prices(p)
ro = p.pivot_table(index="date", columns="stock_id", values="open").reindex(index=raw.index, columns=raw.columns)
days = raw.index
liquid = (vol.rolling(20).mean() >= 500) & (raw >= 10)
can_buy = ro.shift(-1) < raw * 1.095
entry = opn.shift(-1)
fwd = {h: adj.shift(-h) / entry - 1 - COST for h in H}
base = {h: fwd[h].where(liquid).mean(axis=1) for h in H}
empty = pd.DataFrame(False, index=days, columns=raw.columns)


def from_list(dates_ids):
    m = empty.copy()
    for d, s in dates_ids:
        if d in m.index and s in m.columns:
            m.at[d, s] = True
    return m


sig = {}
ms = pd.read_csv(ROOT / "results/market_signals.csv", dtype={"stock_id": str}, parse_dates=["signal_date"])
for k, g in ms.groupby("strategy"):
    sig[k] = from_list(zip(g["signal_date"], g["stock_id"]))
ev = pead.events(raw, adj, vol, load_market_index()["taiex"])
sig["v2_營收加速"] = from_list((d, s) for d, x in ev.items() for s in x.index)
oc = (raw / ro - 1).where(lambda x: x.abs() <= 0.11)
wk = pd.Series(days.to_period("W"), index=days)
wk = pd.DataFrame(np.repeat((wk != wk.shift(-1)).values[:, None], raw.shape[1], 1), index=days, columns=raw.columns)
sig["日內動能IMOM(每週)"] = wk & (oc.rolling(20).sum().where(liquid).rank(axis=1, pct=True) >= 0.9)
chg = (raw / raw.shift(1) - 1).where(lambda x: x.abs() <= 0.11)
v20 = vol.rolling(20).mean()
sig["帶量長紅收最高"] = (chg >= 0.05) & (raw >= p.pivot_table(index="date", columns="stock_id", values="high").reindex(index=days, columns=raw.columns) * 0.99) & (vol >= 2 * v20)
c = adj
d_ = c.diff()
rsi = 100 - 100 / (1 + d_.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean() / (-d_.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean())
sig["RSI<30超賣"] = rsi < 30
sig["突破20日高"] = c > c.shift(1).rolling(20).max()
sig["站回20日線"] = (c.shift(1) < c.rolling(20).mean().shift(1)) & (c > c.rolling(20).mean())

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
        rows.append({"策略": name, "天數": h, "筆數": len(st), "勝率": (st > 0).mean(), "平均": st.mean(),
                     "比隨機": x.mean(), "前半": x[x.index < half].mean(), "後半": x[x.index >= half].mean(),
                     "贏的年數": f"{(yr > 0).sum()}/{len(yr)}", "每年": " ".join(f"{y % 100}:{v:+.1%}" for y, v in yr.items())})
t = pd.DataFrame(rows)
out = ROOT / "results" / "持有天數"
out.mkdir(parents=True, exist_ok=True)
t.to_csv(out / "horizon_matrix.csv", index=False, encoding="utf-8-sig")
lines = [f"{days[0]:%Y-%m-%d} ~ {days[-1]:%Y-%m-%d}，成本 {COST:.1%}；比隨機＝每天一票的平均超額"]
piv = t.pivot(index="策略", columns="天數", values="比隨機")
lines += ["", "== 比隨機（每筆超額）", piv.map(lambda v: f"{v:+.2%}").to_string()]
lines += ["", "== 贏的年數", t.pivot(index="策略", columns="天數", values="贏的年數").to_string()]
lines += ["", "== 每個長度：前後段都贏、贏的年數最多，再依超額排序"]
for h in H:
    g = t[(t["天數"] == h) & (t["前半"] > 0) & (t["後半"] > 0)].copy()
    g["wy"] = g["贏的年數"].str.split("/").map(lambda a: int(a[0]) / int(a[1]))
    g = g.sort_values(["wy", "比隨機"], ascending=False).head(4)
    lines.append(f"{h} 天：" + "；".join(f"{r['策略']} {r['比隨機']:+.2%}（{r['贏的年數']}年，勝率 {r['勝率']:.0%}，{r['筆數']} 筆）" for _, r in g.iterrows()))
text = "\n".join(lines)
print(text)
(out / "horizon_matrix.txt").write_text(text, encoding="utf-8")
