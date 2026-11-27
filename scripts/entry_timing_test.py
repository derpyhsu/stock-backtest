"""進場時間點有沒有依據（2026-11-27）：處置股和一年新高，各種「什麼時候買」放在同一批訊號上比。
營收加速的進場已測過（scripts/entry_limit_test.py、entry_ma_test.py：開盤市價買最好）。

一、處置股（出場都是處置最後一天收盤，扣 0.6%）
  進場：首日開盤（現在）、首日收盤、第 2 天開盤、第 2 天收盤、第 3 天開盤、第 5 天開盤；
  依首日開盤跳空分組；依首日收黑或收紅分組（第 2 天開盤才買的人看得到）。
二、一年新高（S1 訊號，抱 60 個交易日收盤賣，扣 0.6%）
  進場：隔天開盤（現在）、隔天收盤、第 2 天開盤、掛訊號日收盤價等 1 天／3 天（等不到放棄，或等不到就市價追）、
  隔天開高超過 3%／5% 不買。
    python scripts/entry_timing_test.py
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
raw, ro, vol, rl = piv("close"), piv("open"), piv("vol"), piv("low")
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
f = tri / raw
liquid = (vol.rolling(20).mean() >= 500) & (raw >= 10)
gap = ro / raw.shift(1) - 1
T, O, LO, L, G, R = tri.values, (f * ro).values, (f * rl).values, liquid.values, gap.values, raw.values
N = len(days)
col = {s: j for j, s in enumerate(raw.columns)}
lines = []


def row(lab, t, extra=""):
    t = pd.DataFrame(t, columns=["d", "ret"])
    if len(t) < 30:
        return f"{lab:<40} 筆數不足（{len(t)}）"
    yr = " ".join(f"{y % 100}:{v:+.1%}" for y, v in t.groupby(t["d"].dt.year)["ret"].mean().items())
    h = t["d"].sort_values().iloc[len(t) // 2]
    return (f"{lab:<40}{len(t):>6}{(t['ret'] > 0).mean():>6.0%}{t['ret'].mean():>+8.2%}{t['ret'].median():>+8.2%}{t['ret'].quantile(.05):>+8.1%}"
            f"{t[t['d'] < h]['ret'].mean():>+8.2%}{t[t['d'] >= h]['ret'].mean():>+8.2%}  {yr} {extra}")


HEAD = f"{'':<40}{'筆數':>6}{'勝率':>6}{'平均':>8}{'中位':>8}{'最差5%':>8}{'前半':>8}{'後半':>8}  每年平均"

# 一、處置股
d = pd.read_csv(ROOT / "data/disposition.csv", dtype={"stock_id": str}, parse_dates=["announce", "start", "end"])
d = d[d["stock_id"].isin(raw.columns) & d["stock_id"].str.fullmatch(r"\d{4}")]
idx = pd.Series(np.arange(N), index=days)
d["i_s"] = idx.reindex(d["start"]).values
d["i_e"] = np.searchsorted(days.values, d["end"].values, side="right") - 1
d = d.dropna(subset=["i_s"])
d["i_s"] = d["i_s"].astype(int)
d = d[(d["i_s"] > 21) & (d["i_e"] >= d["i_s"] + 5) & (d["i_e"] < N)]       # 至少 6 個交易日，才比得了第 5 天進場
d["j"] = d["stock_id"].map(col)
d = d[[bool(L[i - 1, j]) and np.isfinite(O[i, j]) and np.isfinite(T[e, j]) for i, e, j in zip(d["i_s"], d["i_e"], d["j"])]]
d = d.sort_values(["i_s", "stock_id"]).drop_duplicates(["i_s", "stock_id"]).reset_index(drop=True)
d["g1"] = [G[i, j] for i, j in zip(d["i_s"], d["j"])]
d["day1"] = [T[i, j] / O[i, j] - 1 for i, j in zip(d["i_s"], d["j"])]
d["g2"] = [G[i + 1, j] for i, j in zip(d["i_s"], d["j"])]
lines += [f"== 一、處置股：{len(d)} 筆（處置至少 6 個交易日、流動性夠），出場都是處置最後一天收盤",
          f"處置首日（開盤到收盤）：平均 {d['day1'].mean():+.2%}、中位 {d['day1'].median():+.2%}、收黑比例 {(d['day1'] < 0).mean():.0%}；首日開盤跳空平均 {d['g1'].mean():+.2%}", HEAD]


def disp(g, when):
    t = []
    for r in g.itertuples():
        i, e, j = r.i_s, r.i_e, r.j
        k, px, gp = {"o1": (i, O[i, j], G[i, j]), "c1": (i, T[i, j], R[i, j] / R[i - 1, j] - 1), "o2": (i + 1, O[i + 1, j], G[i + 1, j]),
                     "c2": (i + 1, T[i + 1, j], R[i + 1, j] / R[i, j] - 1), "o3": (i + 2, O[i + 2, j], G[i + 2, j]), "o5": (i + 4, O[i + 4, j], G[i + 4, j])}[when]
        if not np.isfinite(px) or not np.isfinite(gp) or gp >= 0.095:      # 漲停買不到
            continue
        t.append((days[i], T[e, j] / px - 1 - COST))
    return t


NAMES = {"o1": "首日開盤買（現在）", "c1": "首日收盤買", "o2": "第 2 天開盤買", "c2": "第 2 天收盤買", "o3": "第 3 天開盤買", "o5": "第 5 天開盤買"}
for k, lab in NAMES.items():
    lines.append(row(lab, disp(d, k)))
lines += ["", "-- 依首日開盤跳空分組：首日開盤買 vs 第 2 天開盤買"]
for lab, m in [("首日開低 3% 以上", d["g1"] <= -0.03), ("首日開低 0~3%", (d["g1"] > -0.03) & (d["g1"] < 0)), ("首日開平或開高 0~3%", (d["g1"] >= 0) & (d["g1"] < 0.03)), ("首日開高 3% 以上", d["g1"] >= 0.03)]:
    lines.append(row(f"{lab}・首日開盤買", disp(d[m], "o1")))
    lines.append(row(f"{lab}・第 2 天開盤買", disp(d[m], "o2")))
lines += ["", "-- 依首日漲跌分組（第 2 天開盤才買的人看得到）：第 2 天開盤買"]
c1 = np.array([R[i, j] / R[i - 1, j] - 1 for i, j in zip(d["i_s"], d["j"])])
for lab, m in [("首日跌 5% 以上", c1 <= -0.05), ("首日跌 0~5%", (c1 > -0.05) & (c1 < 0)), ("首日漲 0~5%", (c1 >= 0) & (c1 < 0.05)), ("首日漲 5% 以上", c1 >= 0.05)]:
    lines.append(row(f"{lab}・第 2 天開盤買", disp(d[m], "o2")))
    lines.append(row(f"{lab}・（對照）首日開盤買", disp(d[m], "o1")))

# 二、一年新高
s1 = pd.read_csv(ROOT / "results" / "market_signals.csv", dtype={"stock_id": str})
s1 = s1[(s1["strategy"] == "S1_一年新高") & s1["stock_id"].isin(raw.columns)]
di, ci = days.get_indexer(pd.to_datetime(s1["signal_date"])), raw.columns.get_indexer(s1["stock_id"])
ok = (di >= 0) & (di + 64 < N)
di, ci = di[ok], ci[ok]
HOLD = 60


def s1run(mode, lim_days=1, chase=False, maxgap=None):
    t, sig, fill = [], 0, 0
    for i, j in zip(di, ci):
        c0 = T[i, j]
        if not np.isfinite(c0) or not np.isfinite(O[i + 1, j]):
            continue
        sig += 1
        k = px = None
        if mode == "open":
            if G[i + 1, j] < 0.095 and (maxgap is None or G[i + 1, j] <= maxgap):
                k, px = i + 1, O[i + 1, j]
        elif mode == "close":
            if R[i + 1, j] / R[i, j] - 1 < 0.095:
                k, px = i + 1, T[i + 1, j]
        elif mode == "open2":
            if G[i + 2, j] < 0.095:
                k, px = i + 2, O[i + 2, j]
        else:   # 掛訊號日收盤價
            for q in range(i + 1, i + 1 + lim_days):
                if np.isfinite(LO[q, j]) and LO[q, j] <= c0:
                    k, px = q, min(O[q, j], c0)
                    break
            if k is None and chase and G[i + 1 + lim_days, j] < 0.095:
                k, px = i + 1 + lim_days, O[i + 1 + lim_days, j]
        if k is None or not np.isfinite(px) or not np.isfinite(T[i + HOLD, j]):
            continue
        fill += 1
        t.append((days[i], T[i + HOLD, j] / px - 1 - COST))      # 出場日固定＝訊號後第 60 天，大家同一天賣
    tt = pd.DataFrame(t, columns=["d", "ret"])
    return t, f"成交率 {fill / sig:.0%}・每個訊號平均（沒成交算 0）{tt['ret'].sum() / sig:+.2%}"


lines += ["", f"== 二、一年新高：S1 訊號 {len(di)} 筆，都在訊號後第 60 個交易日收盤賣", HEAD]
for lab, kw in [("隔天開盤買（現在）", dict(mode="open")), ("隔天收盤買", dict(mode="close")), ("第 2 天開盤買", dict(mode="open2")),
                ("隔天開高超過 3% 不買", dict(mode="open", maxgap=0.03)), ("隔天開高超過 5% 不買", dict(mode="open", maxgap=0.05)),
                ("掛訊號日收盤價等 1 天，等不到放棄", dict(mode="limit", lim_days=1)), ("掛訊號日收盤價等 3 天，等不到放棄", dict(mode="limit", lim_days=3)),
                ("掛訊號日收盤價等 1 天，等不到隔天市價追", dict(mode="limit", lim_days=1, chase=True)),
                ("掛訊號日收盤價等 3 天，等不到市價追", dict(mode="limit", lim_days=3, chase=True))]:
    t, extra = s1run(**kw)
    lines.append(row(lab, t, extra))
text = "\n".join(lines)
print(text)
out = ROOT / "results" / "規則依據"
out.mkdir(parents=True, exist_ok=True)
(out / "entry_timing.txt").write_text(text, encoding="utf-8")
