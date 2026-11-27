"""進場點位：均線位置、乖離、壓力區、支撐區對三份策略有沒有用（2026-11-27）。

三組訊號，各用自己的出場方式，扣 0.6%：
  營收加速（pead.events）、一年新高（market_signals 的 S1）：訊號隔天開盤買，訊號後第 60 個交易日收盤賣。
  處置股：處置首日收盤判斷、第 2 天開盤買，處置最後一天收盤賣。
點位都只用判斷日收盤以前的資料（還原價）：
  均線：收盤在 5／20／60 日線上或下；離 20 日線多遠（乖離）。
  大量區：過去 120 天，把每天成交量記在當天收盤價所在的 2% 價格帶，量最大的那一帶。股價在它上面＝支撐在下，在它下面＝壓力在上。
  前高壓力：離 250 日最高收盤多遠。前低支撐：離 20 日最低收盤多遠。
一、依點位分組，看同一種訊號裡哪一組比較好。
二、不追市價，改掛在 5／10／20 日線、大量區、判斷日最低價等「支撐價」等 N 天；等不到放棄，或等不到就市價買。
    python scripts/price_level_test.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from twscreener import pead  # noqa: E402
from twscreener.market import load_market_index, load_panel  # noqa: E402

COST = 0.006
p = load_panel()
piv = lambda k: p.pivot_table(index="date", columns="stock_id", values=k)  # noqa: E731
raw, ro, vol, rl = piv("close"), piv("open"), piv("vol"), piv("low")
days = raw.index
N = len(days)
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
T, O, LO, L, G, V = tri.values, (f * ro).values, (f * rl).values, liquid.values, gap.values, vol.fillna(0).values
MA = {n: tri.rolling(n).mean().values for n in (5, 10, 20, 60)}
HI250 = tri.rolling(250, min_periods=120).max().values
LO20 = tri.rolling(20).min().values
col = {s: j for j, s in enumerate(raw.columns)}


def poc(i, j, w=120):
    """過去 w 天量最大的 2% 價格帶的中心價。"""
    c, v = T[i - w + 1:i + 1, j], V[i - w + 1:i + 1, j]
    m = np.isfinite(c) & (v > 0)
    if m.sum() < 60:
        return np.nan
    b = np.floor(np.log(c[m]) / np.log(1.02)).astype(int)
    s = pd.Series(v[m]).groupby(b).sum()
    return 1.02 ** (s.idxmax() + 0.5)


def feats(i, j):
    c = T[i, j]
    q = poc(i, j)
    return {"m5": c / MA[5][i, j] - 1, "m20": c / MA[20][i, j] - 1, "m60": c / MA[60][i, j] - 1, "poc": c / q - 1 if np.isfinite(q) else np.nan,
            "hi": c / HI250[i, j] - 1, "lo": c / LO20[i, j] - 1, "pocp": q}


# 三組訊號：每列 = (判斷日 index i, 欄 j, 進場日 k, 出場日 e)
sets = {}
ev = pead.events(raw, tri, vol, load_market_index()["taiex"])
rows = [(days.get_loc(d), col[s]) for d, ser in ev.items() for s in ser.index if s in col]
sets["營收加速"] = [(i, j, i + 1, i + 60) for i, j in rows if i + 60 < N]
s1 = pd.read_csv(ROOT / "results" / "market_signals.csv", dtype={"stock_id": str})
s1 = s1[(s1["strategy"] == "S1_一年新高") & s1["stock_id"].isin(raw.columns)]
di = days.get_indexer(pd.to_datetime(s1["signal_date"]))
sets["一年新高"] = [(i, col[s], i + 1, i + 60) for i, s in zip(di, s1["stock_id"]) if i >= 0 and i + 60 < N]
d = pd.read_csv(ROOT / "data/disposition.csv", dtype={"stock_id": str}, parse_dates=["announce", "start", "end"])
d = d[d["stock_id"].isin(raw.columns) & d["stock_id"].str.fullmatch(r"\d{4}") & d["start"].isin(days)]
d["i_s"] = days.get_indexer(d["start"])
d["i_e"] = np.searchsorted(days.values, d["end"].values, side="right") - 1
d = d[(d["i_s"] > 250) & (d["i_e"] >= d["i_s"] + 3) & (d["i_e"] < N)].drop_duplicates(["i_s", "stock_id"])
sets["處置股"] = [(r.i_s, col[r.stock_id], r.i_s + 1, r.i_e) for r in d.itertuples() if L[r.i_s - 1, col[r.stock_id]]]

lines, out_rows = [], []
HEAD = f"{'':<34}{'筆數':>6}{'勝率':>6}{'平均':>8}{'中位':>8}{'最差5%':>8}{'前半':>8}{'後半':>8}  每年平均"


def row(lab, t, extra=""):
    t = pd.DataFrame(t, columns=["d", "ret"])
    if len(t) < 30:
        return f"{lab:<34} 筆數不足（{len(t)}）"
    yr = " ".join(f"{y % 100}:{v:+.1%}" for y, v in t.groupby(t["d"].dt.year)["ret"].mean().items())
    h = t["d"].sort_values().iloc[len(t) // 2]
    return (f"{lab:<34}{len(t):>6}{(t['ret'] > 0).mean():>6.0%}{t['ret'].mean():>+8.2%}{t['ret'].median():>+8.2%}{t['ret'].quantile(.05):>+8.1%}"
            f"{t[t['d'] < h]['ret'].mean():>+8.2%}{t[t['d'] >= h]['ret'].mean():>+8.2%}  {yr} {extra}")


GROUPS = [("收盤在 5 日線上", lambda x: x.m5 >= 0), ("收盤在 5 日線下", lambda x: x.m5 < 0),
          ("收盤在 20 日線上", lambda x: x.m20 >= 0), ("收盤在 20 日線下", lambda x: x.m20 < 0),
          ("收盤在 60 日線上", lambda x: x.m60 >= 0), ("收盤在 60 日線下", lambda x: x.m60 < 0),
          ("離 20 日線 0~5%", lambda x: (x.m20 >= 0) & (x.m20 < .05)), ("離 20 日線 5~10%", lambda x: (x.m20 >= .05) & (x.m20 < .10)),
          ("離 20 日線 10~20%", lambda x: (x.m20 >= .10) & (x.m20 < .20)), ("離 20 日線 20% 以上", lambda x: x.m20 >= .20),
          ("在大量區上方 5% 以上（支撐在下）", lambda x: x.poc > .05), ("在大量區上下 5% 內", lambda x: x.poc.abs() <= .05),
          ("在大量區下方 5% 以上（壓力在上）", lambda x: x.poc < -.05),
          ("在大量區上方 5~20%", lambda x: (x.poc > .05) & (x.poc <= .20)), ("在大量區上方 20% 以上", lambda x: x.poc > .20),
          ("離一年高點 5% 內", lambda x: x.hi >= -.05), ("離一年高點 5~15%", lambda x: (x.hi < -.05) & (x.hi >= -.15)), ("離一年高點 15% 以上", lambda x: x.hi < -.15),
          ("離 20 日低點 10% 內", lambda x: x.lo < .10), ("離 20 日低點 10~25%", lambda x: (x.lo >= .10) & (x.lo < .25)), ("離 20 日低點 25% 以上", lambda x: x.lo >= .25)]
tabs = {}
for name, ev_ in sets.items():
    t = []
    for i, j, k, e in ev_:
        if not (np.isfinite(O[k, j]) and np.isfinite(T[e, j]) and np.isfinite(T[i, j])) or G[k, j] >= 0.095:
            continue
        t.append({"d": days[i], "i": i, "j": j, "k": k, "e": e, "ret": T[e, j] / O[k, j] - 1 - COST, **feats(i, j)})
    t = pd.DataFrame(t)
    tabs[name] = t
    lines += ["", f"== 一、{name}：依判斷日的點位分組（{len(t)} 筆，{t['d'].min():%Y-%m} ~ {t['d'].max():%Y-%m}）", HEAD, row("全部", t[["d", "ret"]])]
    for lab, fn in GROUPS:
        g = t[fn(t).fillna(False)]
        lines.append(row(lab, g[["d", "ret"]]))
        if len(g) >= 30:
            out_rows.append({"策略": name, "分組": lab, "筆數": len(g), "勝率": (g.ret > 0).mean(), "平均": g.ret.mean(), "中位數": g.ret.median()})


# 二、掛支撐價等 N 天
def limit_test(name, t, wait):
    lines.extend(["", f"== 二、{name}：掛支撐價買（最多等 {wait} 天；處置股等到處置結束前一天）", HEAD])
    PX = [("市價（對照）", None), ("掛 5 日線", lambda r: MA[5][r.i, r.j]), ("掛 10 日線", lambda r: MA[10][r.i, r.j]), ("掛 20 日線", lambda r: MA[20][r.i, r.j]),
          ("掛判斷日最低價", lambda r: LO[r.i, r.j]), ("掛大量區價", lambda r: r.pocp), ("掛判斷日收盤價 -3%", lambda r: T[r.i, r.j] * 0.97),
          ("掛判斷日收盤價 -5%", lambda r: T[r.i, r.j] * 0.95)]
    for lab, fn in PX:
        for chase in ((False,) if fn is None else (False, True)):
            res, sig = [], 0
            for r in t.itertuples():
                sig += 1
                if fn is None:
                    res.append((r.d, r.ret))
                    continue
                lim = fn(r)
                last = min(r.k + wait - 1, r.e - 1)
                k = px = None
                if np.isfinite(lim) and lim < T[r.i, r.j] * 1.0001:          # 支撐價要在現價以下才叫等拉回
                    for q in range(r.k, last + 1):
                        if np.isfinite(LO[q, r.j]) and LO[q, r.j] <= lim:
                            k, px = q, min(O[q, r.j], lim)
                            break
                    if k is None and chase and last + 1 <= r.e and np.isfinite(O[last + 1, r.j]) and G[last + 1, r.j] < 0.095:
                        k, px = last + 1, O[last + 1, r.j]
                elif chase or not np.isfinite(lim) or lim >= T[r.i, r.j]:    # 支撐價已在現價之上（股價在均線下）：直接市價
                    k, px = r.k, O[r.k, r.j]
                if k is not None and np.isfinite(px):
                    res.append((r.d, T[r.e, r.j] / px - 1 - COST))
            tt = pd.DataFrame(res, columns=["d", "ret"])
            nm = lab + ("" if fn is None else ("，等不到市價買" if chase else "，等不到放棄"))
            lines.append(row(nm, res, f"成交率 {len(tt) / sig:.0%}・每個訊號平均 {tt['ret'].sum() / sig:+.2%}"))


limit_test("一年新高", tabs["一年新高"], 5)
limit_test("處置股", tabs["處置股"], 99)
limit_test("營收加速", tabs["營收加速"], 5)
text = "\n".join(lines)
print(text)
out = ROOT / "results" / "規則依據"
out.mkdir(parents=True, exist_ok=True)
(out / "price_level.txt").write_text(text, encoding="utf-8")
pd.DataFrame(out_rows).to_csv(out / "price_level_groups.csv", index=False, encoding="utf-8-sig")
