"""處置股事件回測（2026-11-26）：公告後進場、處置期間、出關後的報酬，和同期間隨機買股比。

資料 data/disposition.csv（scripts/fetch_disposition.py；證交所、櫃買處置公告）。只看 4 碼普通股。
公告日收盤後才知道 → 最早隔天（處置開始日）開盤買。處置期間是分盤撮合（5～20 分鐘一次、可能要預收款券），
實際成交比回測差；2026-08-10 起處置天數規則改過。成本 0.6%。還原報酬（含除權息）。
分組：第一次處置／再次處置（含第二次處置：30 天內又被處置，撮合間隔更長）。
    python scripts/disposition_test.py
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
piv = lambda k: p.pivot_table(index="date", columns="stock_id", values=k)
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
amt20 = (raw * vol).rolling(20).mean() * 1000
liquid = (vol.rolling(20).mean() >= 500) & (raw >= 10)
T, O, L = tri.values, tro.values, liquid.values
col = {s: j for j, s in enumerate(raw.columns)}
d = pd.read_csv(ROOT / "data/disposition.csv", dtype={"stock_id": str}, parse_dates=["announce", "start", "end"])
d = d[d["stock_id"].isin(raw.columns) & d["stock_id"].str.fullmatch(r"\d{4}")]
d["again"] = d["measure"].str.contains("再次|第二次")
idx = pd.Series(np.arange(len(days)), index=days)
d["i_s"] = idx.reindex(d["start"]).values       # 處置開始日
d["i_e"] = np.searchsorted(days.values, d["end"].values, side="right") - 1   # 處置最後一天
d = d.dropna(subset=["i_s"])
d["i_s"] = d["i_s"].astype(int)
d = d[(d["i_s"] > 21) & (d["i_e"] >= d["i_s"])]
bc = {}


def base(i0, i1):
    if (i0, i1) not in bc:
        m = L[i0 - 1]
        bc[(i0, i1)] = np.nanmean(T[i1][m] / O[i0][m] - 1)
    return bc[(i0, i1)]


def run(name, g, entry, exit_):
    rows = []
    for r in g.itertuples():
        j = col[r.stock_id]
        i0, i1 = entry(r), exit_(r)
        if i1 >= len(days) or i0 > i1:
            continue
        x = T[i1, j] / O[i0, j] - 1
        if np.isfinite(x):
            rows.append((days[i0], x - COST, x - base(i0, i1)))
    t = pd.DataFrame(rows, columns=["d", "ret", "x"])
    if len(t) < 30:
        return f"{name:<24} 筆數不足（{len(t)}）"
    yr = t.groupby(t["d"].dt.year)["x"].mean()
    h = t["d"].sort_values().iloc[len(t) // 2]
    return (f"{name:<24}{len(t):>6}{(t['ret'] > 0).mean():>6.0%}{t['ret'].mean():>+8.2%}{t['ret'].median():>+8.2%}{t['x'].mean():>+8.2%}"
            f"{t[t['d'] < h]['x'].mean():>+8.2%}{t[t['d'] >= h]['x'].mean():>+8.2%}  " + " ".join(f"{y % 100}:{v:+.1%}" for y, v in yr.items()))


lines = [f"處置事件 {len(d)} 筆（{d['announce'].min():%Y-%m} ~ {d['announce'].max():%Y-%m}），第一次 {(~d['again']).sum()}、再次 {d['again'].sum()}；成本 {COST:.1%}",
         f"{'做法':<24}{'筆數':>6}{'勝率':>6}{'平均':>8}{'中位':>8}{'比隨機':>8}{'前半':>8}{'後半':>8}  每年比隨機"]
PLANS = [("處置首日開盤買→處置結束賣", lambda r: r.i_s, lambda r: r.i_e),
         ("處置首日開盤買→出關後 5 天", lambda r: r.i_s, lambda r: r.i_e + 5),
         ("處置首日開盤買→出關後 20 天", lambda r: r.i_s, lambda r: r.i_e + 20),
         ("處置最後 3 天買→結束賣", lambda r: r.i_e - 2, lambda r: r.i_e),
         ("出關日開盤買→抱 5 天", lambda r: r.i_e + 1, lambda r: r.i_e + 5),
         ("出關日開盤買→抱 10 天", lambda r: r.i_e + 1, lambda r: r.i_e + 10),
         ("出關日開盤買→抱 20 天", lambda r: r.i_e + 1, lambda r: r.i_e + 20),
         ("出關日開盤買→抱 60 天", lambda r: r.i_e + 1, lambda r: r.i_e + 60)]
big = d[[amt20.iat[i - 1, col[s]] >= 2e8 for i, s in zip(d["i_s"], d["stock_id"])]]
for lab, g in [("全部", d), ("第一次處置", d[~d["again"]]), ("再次／第二次處置", d[d["again"]]),
               ("再次處置・處置前 20 日均成交 >= 2 億", big[big["again"]]), ("第一次・處置前 20 日均成交 >= 2 億", big[~big["again"]])]:
    lines.append(f"-- {lab}")
    for name, a, b in PLANS:
        lines.append(run(name, g, a, b))
text = "\n".join(lines)
print(text)
(ROOT / "results/大盤以外/disposition.txt").write_text(text, encoding="utf-8")
