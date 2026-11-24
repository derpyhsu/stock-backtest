"""v2 訊號：開盤市價買 vs 掛均線價（訊號日當天已知的 5／10／20 日均線）等 N 天。
等不到：A 放棄；B 第 N+1 天開盤市價買。也比「開盤跳空超過 x% 不買」。
持有 60 個交易日或跌破成交價 -15% 停損，扣成本 0.6%。
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from twscreener import pead  # noqa: E402
from twscreener.market import load_panel, load_market_index  # noqa: E402
from twscreener.simulate import prices  # noqa: E402
panel = load_panel()
pv = lambda k: panel.pivot_table(index="date", columns="stock_id", values=k)
raw, op, lo = pv("close"), pv("open"), pv("low")
_, adj, _, vol = prices(panel)
ev = pead.events(raw, adj, vol, load_market_index()["taiex"])
days = raw.index
MA = {n: raw.rolling(n).mean() for n in [5, 10, 20]}


def outcome(a, r, k, px):
    base = a[k] * px / r[k]
    for j in range(k + 1, min(k + 61, len(a))):
        if a[j] / base - 1 <= pead.STOP:
            return pead.STOP - 0.006, 1
    return a[min(k + 60, len(a) - 1)] / base - 1 - 0.006, 0


rows = []
for d, s in ev.items():
    i = days.get_loc(d)
    if i + 70 >= len(days):
        continue
    for sid in s.index:
        c = days.get_loc(d)
        a, r = adj[sid].values, raw[sid].values
        o, l = op[sid].values, lo[sid].values
        if not np.isfinite(o[i + 1]) or o[i + 1] >= r[i] * 1.095:
            continue
        rec = {"date": d, "gap": o[i + 1] / r[i] - 1}
        rec["市價"], rec["市價_停損"] = outcome(a, r, i + 1, o[i + 1])
        for n in [5, 10, 20]:
            lim = MA[n].at[d, sid]
            for wait in [1, 5]:
                got = None
                for k in range(i + 1, i + 1 + wait):
                    if np.isfinite(l[k]) and l[k] <= lim:
                        got = (k, min(o[k], lim)); break
                key = f"{n}日線等{wait}天"
                if got:
                    rec[key], rec[key + "_停損"] = outcome(a, r, *got)
                    rec[key + "B"], rec[key + "B_停損"] = rec[key], rec[key + "_停損"]
                else:
                    rec[key], rec[key + "_停損"] = np.nan, np.nan
                    k = i + 1 + wait
                    rec[key + "B"], rec[key + "B_停損"] = outcome(a, r, k, o[k]) if np.isfinite(o[k]) else (np.nan, np.nan)
        for g in [0.03, 0.05]:
            ok = rec["gap"] <= g
            rec[f"跳空>{g:.0%}不買"] = rec["市價"] if ok else np.nan
            rec[f"跳空>{g:.0%}不買_停損"] = rec["市價_停損"] if ok else np.nan
        rows.append(rec)
d = pd.DataFrame(rows)
d["年"] = d["date"].dt.year
print(f"訊號 {len(d)} 筆（{d['date'].min().date()} ~ {d['date'].max().date()}）。A=等不到就放棄，B=等不到就市價買")
print("| 買法 | 成交率 | 成交那幾筆平均 | 停損出場比例 | 每個訊號平均 | 逐年每訊號平均 |")
print("|---|---|---|---|---|---|")
keys = ["市價", "跳空>3%不買", "跳空>5%不買"] + [f"{n}日線等{w}天{b}" for n in [5, 10, 20] for w in [1, 5] for b in ["", "B"]]
for k in keys:
    if k not in d:
        continue
    f = d[k].notna()
    per = d[k].fillna(0)
    yy = " ".join(f"{y % 100}:{v:+.1%}" for y, v in per.groupby(d["年"]).mean().items())
    name = k.replace("天B", "天(B)") if k.endswith("B") else (k + "(A)" if "日線" in k else k)
    print(f"| {name} | {f.mean():.0%} | {d.loc[f, k].mean():+.1%} | {d.loc[f, k + '_停損'].mean():.0%} | {per.mean():+.1%} | {yy} |")
