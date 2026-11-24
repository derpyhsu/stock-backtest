"""股價從 120 日高點腰斬後，續抱還是賣？依當時最新月營收分組，看之後 20/60/120 日報酬，和同日隨機股票比。
事件：還原收盤 <= 120 日最高的 55%（每檔 60 個交易日內只算第一次），20 日均量 >= 500 張、股價 >= 10。
月營收只用事件日當天已公布的（available <= 事件日）。
    python scripts/crash_hold_test.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from twscreener.market import load_panel  # noqa: E402
from twscreener.simulate import prices  # noqa: E402
from twscreener.fundamentals import revenue_features  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
raw, adj, opn, vol = prices(load_panel())
adj = adj.loc["2022-01-01":]; raw = raw.reindex_like(adj); vol = vol.reindex_like(adj)
hi = adj.rolling(120, min_periods=100).max()
liquid = (vol.rolling(20).mean() >= 500) & (raw >= 10)
hit = (adj <= hi * 0.55) & liquid
ma20 = adj.rolling(20).mean()
H = [20, 60, 120]
fwd = {h: adj.shift(-h) / adj - 1 for h in H}
base = {h: fwd[h].where(liquid).mean(axis=1) for h in H}       # 同日隨機股票平均
rev = revenue_features(pd.read_csv(ROOT / "data/revenue.csv", dtype={"stock_id": str})).sort_values("available")

ev = []
days = adj.index
for s in adj.columns:
    idx = np.flatnonzero(hit[s].fillna(False).values)
    last = -999
    for i in idx:
        if i - last < 60:
            continue
        last = i
        ev.append((days[i], s))
ev = pd.DataFrame(ev, columns=["date", "stock_id"]).sort_values("date")
r = rev[["stock_id", "available", "yoy", "rev_high12"]].dropna(subset=["yoy"]).rename(columns={"available": "date"})
ev = pd.merge_asof(ev, r.sort_values("date"), on="date", by="stock_id", direction="backward")
ev["營收"] = np.select([ev["yoy"] >= 0.2, ev["yoy"] >= 0], ["年增>=20%", "年增0~20%"], "衰退")
ev.loc[ev["yoy"].isna(), "營收"] = "無資料"
for h in H:
    ev[f"r{h}"] = [fwd[h].at[d, s] for d, s in zip(ev["date"], ev["stock_id"])]
    ev[f"x{h}"] = ev[f"r{h}"] - [base[h].at[d] for d in ev["date"]]
def tab(df, by):
    g = df.groupby(by)
    out = pd.DataFrame({"件數": g.size()})
    for h in H:
        out[f"{h}日平均"] = g[f"r{h}"].mean().map("{:+.1%}".format)
        out[f"{h}日勝率"] = g[f"r{h}"].apply(lambda x: (x.dropna() > 0).mean()).map("{:.0%}".format)
        out[f"{h}日超額"] = g[f"x{h}"].mean().map("{:+.1%}".format)
    return out
pd.set_option("display.width", 250)
print("事件期間", ev["date"].min().date(), "~", ev["date"].max().date())
print("\n== 依營收分組（報酬未扣成本；超額 = 減同日所有股票平均）")
print(tab(ev, "營收").to_string())
ev["年"] = ev["date"].dt.year
print("\n== 年增>=20% 逐年")
print(tab(ev[ev["營收"] == "年增>=20%"], "年").to_string())
print("\n== 全部逐年")
print(tab(ev, "年").to_string())
ev.to_csv(ROOT / "results/腰斬續抱/events.csv", index=False)
