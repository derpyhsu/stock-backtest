"""處置期間跌深、解除後買（2026-11-27）。來源：results/處置股出場/README.md 第二節。
訊號：處置最後一天收盤，比處置首日開盤低 X% 以上 → 隔天（解除當天）開盤買，開盤漲停不買、開盤跌停也不買；抱 N 天收盤賣，扣 0.6%。
比隨機＝同一段期間所有流動性股票（20 日均量 >= 500 張、股價 >= 10）的平均。
    python scripts/disposition_rebound_test.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from twscreener.market import load_panel  # noqa: E402

COST = 0.006
p = load_panel()
piv = lambda k: p.pivot_table(index="date", columns="stock_id", values=k)  # noqa: E731
raw, ro, vol = piv("close"), piv("open"), piv("vol")
days = raw.index
ex = pd.read_csv(ROOT / "data/exrights.csv", dtype={"stock_id": str}, parse_dates=["date"])
ex = ex[ex["date"].isin(days) & ex["stock_id"].isin(raw.columns) & (ex["ref"] > 0)]
ret = raw.pct_change(fill_method=None)
for r in ex.itertuples():
    c = raw.at[r.date, r.stock_id]
    if np.isfinite(c):
        ret.at[r.date, r.stock_id] = c / r.ref - 1
ret = ret.where(ret.abs() <= 0.11, 0.0).fillna(0)
tri = (1 + ret).cumprod().where(raw.notna())
tro = tri / raw * ro
liquid = (vol.rolling(20).mean() >= 500) & (raw >= 10)
gap = ro / raw.shift(1) - 1
T, O, L, G = tri.values, tro.values, liquid.values, gap.values
col = {s: j for j, s in enumerate(raw.columns)}
d = pd.read_csv(ROOT / "data/disposition.csv", dtype={"stock_id": str}, parse_dates=["announce", "start", "end"])
d = d[d["stock_id"].isin(raw.columns) & d["stock_id"].str.fullmatch(r"\d{4}")]
d["again"] = d["measure"].str.contains("再次|第二次")
idx = pd.Series(np.arange(len(days)), index=days)
d["i_s"] = idx.reindex(d["start"]).values
d["i_e"] = np.searchsorted(days.values, d["end"].values, side="right") - 1
d = d.dropna(subset=["i_s"])
d["i_s"] = d["i_s"].astype(int)
N = len(days)
d = d[(d["i_s"] > 21) & (d["i_e"] >= d["i_s"]) & (d["i_e"] + 1 < N)]
d["j"] = d["stock_id"].map(col)
d = d[[bool(L[i - 1, j]) for i, j in zip(d["i_s"], d["j"])]]
d = d.sort_values(["i_e", "stock_id"]).drop_duplicates(["i_e", "stock_id"]).reset_index(drop=True)
d["pret"] = [T[e, j] / O[i, j] - 1 for i, e, j in zip(d["i_s"], d["i_e"], d["j"])]
d["g"] = [G[e + 1, j] for e, j in zip(d["i_e"], d["j"])]
d = d.dropna(subset=["pret", "g"])
d = d[(d["g"] < 0.095) & (d["g"] > -0.095)]
# 解除時是不是馬上又被處置（下一次處置的開始日在解除後 3 天內）
nxt = d.sort_values(["stock_id", "i_s"]).groupby("stock_id")["i_s"].shift(-1)
d["re"] = (nxt - d["i_e"]).between(0, 3)


def run(g, h):
    t = []
    for r in g.itertuples():
        i0, i1 = r.i_e + 1, r.i_e + h
        if i1 >= N or not np.isfinite(T[i1, r.j]) or not np.isfinite(O[i0, r.j]):
            continue
        x = T[i1, r.j] / O[i0, r.j] - 1
        m = L[i0 - 1]
        t.append((days[i0], x - COST, x - np.nanmean(T[i1][m] / O[i0][m] - 1)))
    return pd.DataFrame(t, columns=["d", "ret", "x"])


lines = [f"買得到的解除事件 {len(d)} 筆（{days[d['i_e'].min()]:%Y-%m} ~ {days[d['i_e'].max()]:%Y-%m}）；成本 {COST:.1%}",
         f"{'分組':<30}{'天數':>4}{'筆數':>6}{'勝率':>6}{'平均':>8}{'中位':>8}{'最差5%':>8}{'比隨機':>8}{'前半':>8}{'後半':>8}  每年比隨機"]
GROUPS = [("全部解除", d["pret"] < 9), ("處置期間跌 5% 以上", d["pret"] <= -0.05), ("跌 10% 以上", d["pret"] <= -0.10),
          ("跌 15% 以上", d["pret"] <= -0.15), ("跌 20% 以上", d["pret"] <= -0.20), ("跌 5%~10%", (d["pret"] <= -0.05) & (d["pret"] > -0.10)),
          ("跌 10% 以上・第一次處置", (d["pret"] <= -0.10) & ~d["again"]), ("跌 10% 以上・再次處置", (d["pret"] <= -0.10) & d["again"]),
          ("跌 10% 以上・解除時沒有馬上再被處置", (d["pret"] <= -0.10) & ~d["re"]),
          ("跌 10% 以上・2026-08-10 規則改後", (d["pret"] <= -0.10) & (d["start"] >= "2026-08-10"))]
rows = []
for lab, m in GROUPS:
    for h in (5, 10, 20, 40, 60):
        t = run(d[m], h)
        if len(t) < 30:
            lines.append(f"{lab:<30}{h:>4}  筆數不足（{len(t)}）")
            continue
        hf = t["d"].sort_values().iloc[len(t) // 2]
        yr = " ".join(f"{y % 100}:{v:+.1%}({n})" for (y, v), n in zip(t.groupby(t["d"].dt.year)["x"].mean().items(), t.groupby(t["d"].dt.year).size()))
        lines.append(f"{lab:<30}{h:>4}{len(t):>6}{(t['ret'] > 0).mean():>6.0%}{t['ret'].mean():>+8.2%}{t['ret'].median():>+8.2%}{t['ret'].quantile(.05):>+8.1%}"
                     f"{t['x'].mean():>+8.2%}{t[t['d'] < hf]['x'].mean():>+8.2%}{t[t['d'] >= hf]['x'].mean():>+8.2%}  {yr}")
        rows.append({"分組": lab, "天數": h, "筆數": len(t), "勝率": (t["ret"] > 0).mean(), "平均": t["ret"].mean(), "中位數": t["ret"].median(), "比隨機": t["x"].mean()})
text = "\n".join(lines)
print(text)
out = ROOT / "results" / "處置股出場"
(out / "rebound.txt").write_text(text, encoding="utf-8")
pd.DataFrame(rows).to_csv(out / "rebound.csv", index=False, encoding="utf-8-sig")
