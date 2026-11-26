"""網路上找到、有研究或明確規則、我們資料能測的台股策略，2022 起全市場日線回測（2026-11-26）。

長線（訊號日收盤判斷、隔天開盤買，抱 5／20／60 天收盤賣，扣 0.6%，和同日隨機買股比）：
 L1 投信初買：投信前 40 天都沒買，今天買超 >= 5000 萬元且 >= 當天成交量 10%（wistock、永豐）
 L2 漲停次數高（樂透股，Wang & Lien 2022，預期變差）：近 20 天漲停天數 − 跌停天數 >= 2
 L3 產業調整後短期反轉（每週最後一天）：近 20 日報酬減產業平均，最低 10%
 L4 高量溢酬（每週最後一天）：5 日均量是前 50 天 5 日均量最高、5 日漲跌在 ±5% 內
 L5 季底作帳：季底前 15 個交易日，投信近 20 日買超（佔 20 日量）前 20 名，抱到季底最後一天
 L6 日內動能 IMOM（Ho et al. 2023）：近 20 天「開盤到收盤」報酬加總，最高 10%（每週）
當沖（當天開盤買、收盤賣，扣 0.435%）：
 D1 跌停隔天買（NCU 論文）
 D2 開盤大跌（跳空 <= -4%）買，收盤賣
流動性：20 日均量 >= 500 張、股價 >= 10。
    python scripts/web_strategies_test.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from twscreener.fundamentals import load_industry  # noqa: E402
from twscreener.market import load_panel  # noqa: E402
from twscreener.simulate import prices  # noqa: E402

COST, DCOST = 0.006, 0.0015 + 0.001425 * 2
p = load_panel()
raw, adj, opn, vol = prices(p)
piv = lambda k: p.pivot_table(index="date", columns="stock_id", values=k).reindex(index=raw.index, columns=raw.columns)
ro, rh, rl = piv("open"), piv("high"), piv("low")
trust = piv("trust_net").fillna(0)
v20 = vol.rolling(20).mean()
liquid = (v20 >= 500) & (raw >= 10)
days = raw.index
chg = (raw / raw.shift(1) - 1).where(lambda x: x.abs() <= 0.11)
lim_up, lim_dn = chg >= 0.095, chg <= -0.095
weekend = pd.Series(days.to_period("W"), index=days)
weekend = weekend != weekend.shift(-1)
wk = pd.DataFrame(np.repeat(weekend.values[:, None], raw.shape[1], 1), index=days, columns=raw.columns)

sig = {}
sig["L1 投信初買"] = (trust.shift(1).rolling(40).max() <= 0) & (trust * raw * 1000 >= 5e7) & (trust >= 0.1 * vol)
sig["L2 樂透股(漲停多)"] = (lim_up.astype(int).rolling(20).sum() - lim_dn.astype(int).rolling(20).sum()) >= 2
r20 = adj / adj.shift(20) - 1
ind = pd.Series(load_industry()).reindex(raw.columns).fillna("其他")
resid = r20 - r20.T.groupby(ind).transform("mean").T
sig["L3 產業調整反轉"] = wk & (resid.where(liquid).rank(axis=1, pct=True) <= 0.1)
v5 = vol.rolling(5).mean()
sig["L4 高量溢酬"] = wk & (v5 >= v5.shift(5).rolling(50).max()) & ((adj / adj.shift(5) - 1).abs() <= 0.05)
oc = (raw / ro - 1).where(lambda x: x.abs() <= 0.11)
imom = oc.rolling(20).sum()
sig["L6 日內動能IMOM"] = wk & (imom.where(liquid).rank(axis=1, pct=True) >= 0.9)
# L5 季底作帳
qe = pd.Series(days.to_period("Q"), index=days)
qlast = qe != qe.shift(-1)
pos = pd.Series(np.arange(len(days)), index=days)
qend_pos = pos[qlast].reindex(days).bfill()
start_day = (qend_pos - pos) == 15
tflow = trust.rolling(20).sum() / vol.rolling(20).sum()
l5 = pd.DataFrame(False, index=days, columns=raw.columns)
for d in days[start_day.values]:
    top = tflow.loc[d].where(liquid.loc[d]).nlargest(20).index
    l5.loc[d, top] = True

entry = opn.shift(-1)
can_buy = ro.shift(-1) < raw * 1.095
fwd = {h: adj.shift(-h) / entry - 1 - COST for h in (5, 20, 60)}
base = {h: fwd[h].where(liquid).mean(axis=1) for h in fwd}
half = days[len(days) // 2]


def summarize(name, ret, basel):
    st = ret.stack().dropna()
    if st.empty:
        return f"{name}：沒有訊號"
    dd = st.index.get_level_values(0)
    x = ret.sub(basel, axis=0).mean(axis=1).dropna()  # 每天一票
    yr = x.groupby(x.index.year).mean()
    return (f"{name:<14}{len(st):>7}{(st > 0).mean():>6.0%}{st.mean():>+9.2%}{x.mean():>+9.2%}"
            f"{x[x.index < half].mean():>+9.2%}{x[x.index >= half].mean():>+9.2%}  每年超額 "
            + " ".join(f"{y}:{v:+.2%}" for y, v in yr.items()))


hdr = f"{'策略':<14}{'筆數':>7}{'勝率':>6}{'平均':>9}{'比隨機':>9}{'前半':>9}{'後半':>9}"
lines = [f"{days[0]:%Y-%m-%d} ~ {days[-1]:%Y-%m-%d}；長線成本 {COST:.1%}，比隨機＝每天一票的超額"]
for h in (5, 20, 60):
    lines += ["", f"== 抱 {h} 天", hdr]
    for name, s in sig.items():
        m = (s & liquid & can_buy).fillna(False)
        lines.append(summarize(name, fwd[h].where(m), base[h]))
# L5 抱到季底
r_q = pd.DataFrame(np.nan, index=days, columns=raw.columns)
for d in days[start_day.values]:
    i = pos[d]
    if i + 16 < len(days):
        r_q.loc[d] = adj.iloc[i + 15] / entry.loc[d] - 1 - COST
lines += ["", "== L5 季底作帳（季底前 15 天買、抱到季底）", hdr,
          summarize("L5 季底作帳", r_q.where(l5 & can_buy), r_q.where(liquid).mean(axis=1))]
# 當沖
d_oc = raw / ro - 1 - DCOST
dbase = (raw / ro - 1).where(liquid).mean(axis=1)
gap = ro / raw.shift(1) - 1
lines += ["", f"== 當沖（開盤買、收盤賣，成本 {DCOST:.3%}；比隨機＝同日隨機開盤買收盤賣，未扣成本）", hdr]
for name, s in {"D1 跌停隔天買": lim_dn.shift(1) & (gap > -0.095),
                "D1b 連2跌停隔天": lim_dn.shift(1) & lim_dn.shift(2) & (gap > -0.095),
                "D2 跳空 -4% 以下買": (gap <= -0.04) & (gap > -0.095)}.items():
    m = (s & liquid.shift(1)).fillna(False)
    lines.append(summarize(name, d_oc.where(m), dbase - DCOST))
text = "\n".join(lines)
print(text)
out = ROOT / "results" / "網路策略"
out.mkdir(parents=True, exist_ok=True)
(out / "web_strategies.txt").write_text(text, encoding="utf-8")
