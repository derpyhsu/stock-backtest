"""web_strategies_test.py 裡兩個看起來有效的（L6 日內動能、D2 開盤大跌買）做穩健度檢查。
    python scripts/web_strategies_robust.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from twscreener.market import load_panel  # noqa: E402
from twscreener.simulate import prices  # noqa: E402
COST, DCOST = 0.006, 0.0015 + 0.001425 * 2
p = load_panel()
raw, adj, opn, vol = prices(p)
ro = p.pivot_table(index="date", columns="stock_id", values="open").reindex(index=raw.index, columns=raw.columns)
v20 = vol.rolling(20).mean()
liquid = (v20 >= 500) & (raw >= 10)
days = raw.index
half = days[len(days) // 2]
lines = []
def row(name, ret, basel):
    st = ret.stack().dropna()
    x = ret.sub(basel, axis=0).mean(axis=1).dropna()
    yr = x.groupby(x.index.year).mean()
    return (f"{name:<22}{len(st):>7}{(st > 0).mean():>6.0%}{st.mean():>+9.2%}{x.mean():>+9.2%}{x[x.index < half].mean():>+9.2%}"
            f"{x[x.index >= half].mean():>+9.2%}  " + " ".join(f"{y}:{v:+.2%}" for y, v in yr.items()))
# D2
gap = ro / raw.shift(1) - 1
doc = raw / ro - 1
dbase = doc.where(liquid.shift(1)).mean(axis=1)
lines.append("== D2 開盤跳空下跌、開盤買收盤賣（扣 0.435%；比隨機未扣成本）")
for lo, hi in [(-0.02, -0.01), (-0.03, -0.02), (-0.04, -0.03), (-0.06, -0.04), (-0.095, -0.06)]:
    m = (gap > lo - 1e-9) & (gap <= hi) if False else (gap > lo) & (gap <= hi)
    m = (m & liquid.shift(1)).fillna(False)
    lines.append(row(f"跳空 {lo:.0%}~{hi:.0%}", (doc - DCOST).where(m), dbase - DCOST))
m4 = (gap <= -0.04) & (gap > -0.095)
for lab, cond in [("均量>=5000張", v20.shift(1) >= 5000), ("均量500~5000", v20.shift(1) < 5000),
                  ("前一天也跌>3%", raw.shift(1) / raw.shift(2) - 1 < -0.03), ("前一天沒跌", raw.shift(1) >= raw.shift(2))]:
    m = (m4 & cond & liquid.shift(1)).fillna(False)
    lines.append(row(f"跳空<=-4%・{lab}", (doc - DCOST).where(m), dbase - DCOST))
# 大盤跳空
mkt_gap = gap.where(liquid.shift(1)).median(axis=1)
for lab, cond in [("大盤中位數跳空<-1%", mkt_gap < -0.01), ("大盤沒大跌", mkt_gap >= -0.01)]:
    m = (m4 & liquid.shift(1)).fillna(False) & pd.DataFrame(np.repeat(cond.values[:, None], raw.shape[1], 1), index=days, columns=raw.columns)
    lines.append(row(f"跳空<=-4%・{lab}", (doc - DCOST).where(m), dbase - DCOST))
# L6 IMOM 參數
oc = doc.where(lambda x: x.abs() <= 0.11)
weekend = pd.Series(days.to_period("W"), index=days)
weekend = (weekend != weekend.shift(-1)).values
wk = pd.DataFrame(np.repeat(weekend[:, None], raw.shape[1], 1), index=days, columns=raw.columns)
entry = opn.shift(-1)
can_buy = ro.shift(-1) < raw * 1.095
f60 = adj.shift(-60) / entry - 1 - COST
f20 = adj.shift(-20) / entry - 1 - COST
b60, b20 = f60.where(liquid).mean(axis=1), f20.where(liquid).mean(axis=1)
lines.append("\n== L6 日內動能 IMOM 參數（每週最後一天，抱 60 天；比隨機＝每天一票）")
for lb in (10, 20, 40, 60):
    rk = oc.rolling(lb).sum().where(liquid).rank(axis=1, pct=True)
    for q in (0.9, 0.95):
        m = (wk & (rk >= q) & can_buy).fillna(False)
        lines.append(row(f"回看{lb}天・前{1 - q:.0%}", f60.where(m), b60))
rk = oc.rolling(20).sum().where(liquid).rank(axis=1, pct=True)
lines.append("\n回看 20 天各十分位（抱 60 天）")
for k in range(10):
    m = (wk & (rk > k / 10) & (rk <= (k + 1) / 10) & can_buy).fillna(False)
    lines.append(row(f"第{k + 1}組", f60.where(m), b60))
# 隔夜動能反向：近 20 天漲幅大多來自隔夜
on = (ro / raw.shift(1) - 1).where(lambda x: x.abs() <= 0.11)
rk_on = on.rolling(20).sum().where(liquid).rank(axis=1, pct=True)
m = (wk & (rk >= 0.9) & (rk_on <= 0.5) & can_buy).fillna(False)
lines.append(row("IMOM前10%＋隔夜後半", f60.where(m), b60))
text = "\n".join(lines)
print(text)
(ROOT / "results" / "網路策略" / "web_strategies_robust.txt").write_text(text, encoding="utf-8")
