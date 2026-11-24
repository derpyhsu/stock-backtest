"""v2 營收加速訊號：隔天開盤市價買 vs 掛限價（訊號日收盤 × (1-x)）。
限價單：隔天最低價 <= 限價才成交，成交價 = min(開盤, 限價)；沒成交就放棄（或第 3 天再追）。
持有 60 個交易日或跌破停損 -15%（以成交價算）出場，扣成本 0.6%。
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import twscreener.portfolio as P  # noqa: E402
from twscreener import pead  # noqa: E402
from twscreener.market import load_panel, load_market_index  # noqa: E402
panel = load_panel()
pv = lambda k: panel.pivot_table(index="date", columns="stock_id", values=k)
raw, op, lo, hi = pv("close"), pv("open"), pv("low"), pv("high")
raw_, adj, opn, vol = P.prices(panel) if hasattr(P, "prices") else __import__("twscreener.simulate", fromlist=["prices"]).prices(panel)
taiex = load_market_index()["taiex"]
ev = pead.events(raw, adj, vol, taiex)
days = raw.index
rows = []
for d, s in ev.items():
    i = days.get_loc(d)
    if i + 62 >= len(days):
        continue
    for sid in s.index:
        c0 = raw.at[d, sid]
        o1, l1 = op.iat[i + 1, op.columns.get_loc(sid)], lo.iat[i + 1, lo.columns.get_loc(sid)]
        if not (np.isfinite(o1) and np.isfinite(l1)) or o1 >= c0 * 1.095:
            continue
        a = adj[sid].values; r = raw[sid].values
        def outcome(k, px):
            # 以還原價推算：從第 k 天的成交價起算，60 日或停損
            base_adj = a[k] * px / r[k]
            for j in range(k + 1, min(k + 61, len(a))):
                if a[j] / base_adj - 1 <= pead.STOP:
                    return pead.STOP - 0.006
            return a[min(k + 60, len(a) - 1)] / base_adj - 1 - 0.006
        rec = {"date": d, "stock_id": sid, "市價": outcome(i + 1, o1), "gap": o1 / c0 - 1}
        for x in [0.0, 0.01, 0.02, 0.03, 0.05]:
            lim = c0 * (1 - x)
            rec[f"限{x:.0%}"] = outcome(i + 1, min(o1, lim)) if l1 <= lim else np.nan
        rows.append(rec)
d = pd.DataFrame(rows)
d["年"] = d["date"].dt.year
print(f"訊號 {len(d)} 筆（{d['date'].min().date()} ~ {d['date'].max().date()}）；隔天開盤平均跳空 {d['gap'].mean():+.1%}")
print("| 買法 | 成交率 | 成交那幾筆平均 | 每個訊號平均（沒成交算 0） | 沒成交那幾筆若市價買的平均 | 逐年每訊號平均 |")
print("|---|---|---|---|---|---|")
for k in ["市價", "限0%", "限1%", "限2%", "限3%", "限5%"]:
    fill = d[k].notna()
    per = d[k].fillna(0)
    yy = " ".join(f"{y % 100}:{v:+.1%}" for y, v in per.groupby(d["年"]).mean().items())
    miss = d.loc[~fill, "市價"].mean() if (~fill).any() else np.nan
    print(f"| {k} | {fill.mean():.0%} | {d.loc[fill, k].mean():+.1%} | {per.mean():+.1%} | {miss:+.1%} | {yy} |")
