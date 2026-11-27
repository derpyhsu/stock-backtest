"""處置股：除了「處置最後一天收盤賣」以外的出場方式，和處置期間的吸籌條件（2026-11-27）。

進場不變：公告隔天（處置第一天）開盤買，開盤漲停不買；處置前 20 日均量 >= 500 張、股價 >= 10。成本 0.6%，還原報酬。
一、解除處置後繼續抱，多出來的報酬（從處置最後一天收盤算起，抱 5/10/20/40 天），依處置期間看得到的條件分組：
    處置期間漲跌、法人（外資＋投信）買賣超占成交量、融資增減、量縮程度、是否收在處置期間高點附近、第一次或再次處置。
    比隨機＝同一段期間所有流動性股票的平均。
二、整筆交易換不同出場規則（固定多抱 N 天、賺錢才續抱、跌破 10 日線才賣、從高點回落才賣、條件成立才續抱）。
所有判斷只用當天收盤以前的資料。出場日跌停賣不掉的情況這裡沒有模擬（帳戶回測才有）。
    python scripts/disposition_exit_variants_test.py
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
al = lambda k: piv(k).reindex(index=days, columns=raw.columns)  # noqa: E731
inst = (al("foreign_net").fillna(0) + al("trust_net").fillna(0)).values
trust = al("trust_net").fillna(0).values
mg = al("margin_bal").values
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
T, O, L, G, V = tri.values, tro.values, liquid.values, gap.values, vol.fillna(0).values
MA10, MA20 = tri.rolling(10).mean().values, tri.rolling(20).mean().values
col = {s: j for j, s in enumerate(raw.columns)}
d = pd.read_csv(ROOT / "data/disposition.csv", dtype={"stock_id": str}, parse_dates=["announce", "start", "end"])
d = d[d["stock_id"].isin(raw.columns) & d["stock_id"].str.fullmatch(r"\d{4}")]
d["again"] = d["measure"].str.contains("再次|第二次")
idx = pd.Series(np.arange(len(days)), index=days)
d["i_s"] = idx.reindex(d["start"]).values
d["i_e"] = np.searchsorted(days.values, d["end"].values, side="right") - 1
d = d.dropna(subset=["i_s"])
d["i_s"] = d["i_s"].astype(int)
d = d[(d["i_s"] > 21) & (d["i_e"] >= d["i_s"]) & (d["i_e"] < len(days))]
d["j"] = d["stock_id"].map(col)
d = d[[L[i - 1, j] and np.isfinite(O[i, j]) and G[i, j] < 0.095 for i, j in zip(d["i_s"], d["j"])]]
d = d.sort_values(["i_s", "stock_id"]).drop_duplicates(["i_s", "stock_id"]).reset_index(drop=True)
N = len(days)

# 處置期間看得到的條件（到處置最後一天收盤為止）
f = []
for r in d.itertuples():
    i, e, j = r.i_s, r.i_e, r.j
    px = O[i, j]
    seg = T[i:e + 1, j]
    vs = V[i:e + 1, j].sum()
    f.append({"pret": T[e, j] / px - 1 if np.isfinite(T[e, j]) else np.nan,
              "inst": inst[i:e + 1, j].sum() / vs if vs > 0 else np.nan,
              "trust": trust[i:e + 1, j].sum() / vs if vs > 0 else np.nan,
              "mg": mg[e, j] / mg[i - 1, j] - 1 if np.isfinite(mg[e, j]) and mg[i - 1, j] > 0 else np.nan,
              "volr": V[i:e + 1, j].mean() / V[i - 20:i, j].mean() if V[i - 20:i, j].mean() > 0 else np.nan,
              "near_hi": bool(np.isfinite(T[e, j]) and T[e, j] >= np.nanmax(seg) * 0.97),
              "above10": bool(np.isfinite(T[e, j]) and T[e, j] > MA10[e, j]),
              "pre20": T[i - 1, j] / T[i - 21, j] - 1})
d = pd.concat([d, pd.DataFrame(f)], axis=1).dropna(subset=["pret"])
bc = {}


def base(i0, i1):          # 同期間流動性股票收盤到收盤的平均
    if (i0, i1) not in bc:
        m = L[i0]
        bc[(i0, i1)] = np.nanmean(T[i1][m] / T[i0][m] - 1)
    return bc[(i0, i1)]


def yearly(t, c):
    return " ".join(f"{y % 100}:{v:+.1%}" for y, v in t.groupby(t["d"].dt.year)[c].mean().items())


lines = [f"流動性夠、買得到的處置事件 {len(d)} 筆（{d['start'].min():%Y-%m} ~ {d['start'].max():%Y-%m}），第一次 {(~d['again']).sum()}、再次 {d['again'].sum()}",
         f"處置期間（首日開盤到最後一天收盤）平均 {d['pret'].mean() - COST:+.2%}、中位 {d['pret'].median() - COST:+.2%}", "",
         "== 一、解除處置後繼續抱多出來的報酬（從處置最後一天收盤算起；比隨機／中位數／勝率），20 天欄附每年比隨機",
         f"{'分組':<34}{'筆數':>5}  {'5 天':<20}{'10 天':<20}{'20 天':<20}{'40 天':<20}  20 天每年"]
GROUPS = [("全部", d.index == d.index), ("第一次處置", ~d["again"]), ("再次處置", d["again"]),
          ("處置期間賺錢", d["pret"] > 0), ("處置期間賺 10% 以上", d["pret"] >= 0.10), ("處置期間賠錢", d["pret"] <= 0),
          ("處置期間賠 10% 以上", d["pret"] <= -0.10),
          ("法人處置期間買超", d["inst"] > 0), ("法人買超占量 2% 以上", d["inst"] >= 0.02), ("法人賣超", d["inst"] < 0),
          ("投信買超", d["trust"] > 0), ("融資減少", d["mg"] < 0), ("融資減 10% 以上", d["mg"] <= -0.10), ("融資增加", d["mg"] > 0),
          ("量縮到處置前一半以下", d["volr"] <= 0.5), ("量沒縮（>= 處置前）", d["volr"] >= 1),
          ("收在處置期間高點 3% 內", d["near_hi"]), ("沒收在高點附近", ~d["near_hi"]),
          ("收盤在 10 日線上", d["above10"]), ("收盤在 10 日線下", ~d["above10"]),
          ("吸籌 A：量縮一半以下＋期間不跌", (d["volr"] <= 0.5) & (d["pret"] >= 0)),
          ("吸籌 B：法人買超＋融資減少", (d["inst"] > 0) & (d["mg"] < 0)),
          ("吸籌 C：法人買超＋融資減少＋期間不跌", (d["inst"] > 0) & (d["mg"] < 0) & (d["pret"] >= 0)),
          ("吸籌 D：融資減少＋收在高點附近", (d["mg"] < 0) & d["near_hi"]),
          ("再次處置＋處置期間賺錢", d["again"] & (d["pret"] > 0)), ("再次處置＋收在高點附近", d["again"] & d["near_hi"]),
          ("第一次處置＋收在高點附近", ~d["again"] & d["near_hi"]), ("再次處置＋沒收在高點附近", d["again"] & ~d["near_hi"])]
rows1 = []
for lab, m in GROUPS:
    g = d[np.asarray(m)]
    cells, y20 = [], ""
    for h in (5, 10, 20, 40):
        t = []
        for r in g.itertuples():
            if r.i_e + h < N and np.isfinite(T[r.i_e + h, r.j]):
                x = T[r.i_e + h, r.j] / T[r.i_e, r.j] - 1
                t.append((days[r.i_e], x, x - base(r.i_e, r.i_e + h)))
        t = pd.DataFrame(t, columns=["d", "ret", "x"])
        if len(t) < 30:
            cells.append("筆數不足")
            continue
        cells.append(f"{t['x'].mean():+.1%}／{t['ret'].median():+.1%}／{(t['ret'] > 0).mean():.0%}")
        rows1.append({"分組": lab, "天數": h, "筆數": len(t), "比隨機": t["x"].mean(), "平均": t["ret"].mean(), "中位數": t["ret"].median(), "勝率": (t["ret"] > 0).mean()})
        if h == 20:
            y20 = yearly(t, "x")
    lines.append(f"{lab:<34}{len(g):>5}  " + "".join(f"{c:<20}" for c in cells) + "  " + y20)


# 二、整筆交易換出場規則
def exit_fixed(n):
    return lambda r: min(r.i_e + n, N - 1)


def trail(r, kind, maxd=60, start=None):
    j = r.j
    peak = O[r.i_s, j] if start is not None else np.nanmax(T[r.i_s:r.i_e + 1, j])
    for q in range(r.i_e if start is None else start, min(r.i_e + maxd, N - 1)):
        x = T[q, j]
        if not np.isfinite(x):
            continue
        peak = max(peak, x)
        if (kind == "ma10" and x < MA10[q, j]) or (kind == "ma20" and x < MA20[q, j]) or (kind == "peak15" and x <= peak * 0.85) or (kind == "peak10" and x <= peak * 0.90):
            return q
    return min(r.i_e + maxd, N - 1)


RULES = [("現在：處置最後一天收盤賣", lambda r: r.i_e),
         ("多抱 5 天", exit_fixed(5)), ("多抱 10 天", exit_fixed(10)), ("多抱 20 天", exit_fixed(20)), ("多抱 40 天", exit_fixed(40)),
         ("只有再次處置多抱 20 天", lambda r: r.i_e + 20 if r.again else r.i_e),
         ("處置期間賺錢才多抱 20 天", lambda r: r.i_e + 20 if r.pret > 0 else r.i_e),
         ("收在高點附近才多抱 20 天", lambda r: r.i_e + 20 if r.near_hi else r.i_e),
         ("再次處置＋收在高點附近才多抱 20 天", lambda r: r.i_e + 20 if (r.again and r.near_hi) else r.i_e),
         ("吸籌 B（法人買＋融資減）才多抱 20 天", lambda r: r.i_e + 20 if (r.inst > 0 and r.mg < 0) else r.i_e),
         ("吸籌 A（量縮＋不跌）才多抱 20 天", lambda r: r.i_e + 20 if (r.volr <= 0.5 and r.pret >= 0) else r.i_e),
         ("處置期間賠 10% 以上才多抱 20 天", lambda r: r.i_e + 20 if r.pret <= -0.10 else r.i_e),
         ("融資減 10% 以上才多抱 20 天", lambda r: r.i_e + 20 if r.mg <= -0.10 else r.i_e),
         ("再次處置或賠 10% 以上才多抱 20 天", lambda r: r.i_e + 20 if (r.again or r.pret <= -0.10) else r.i_e),
         ("再次處置＋融資減少才多抱 20 天", lambda r: r.i_e + 20 if (r.again and r.mg < 0) else r.i_e),
         ("再次處置多抱 10 天", lambda r: r.i_e + 10 if r.again else r.i_e),
         ("再次處置多抱 40 天", lambda r: r.i_e + 40 if r.again else r.i_e),
         ("解除後跌破 10 日線才賣（最多 60 天）", lambda r: trail(r, "ma10")),
         ("解除後跌破 20 日線才賣（最多 60 天）", lambda r: trail(r, "ma20")),
         ("解除後從最高收盤回落 10% 才賣", lambda r: trail(r, "peak10")),
         ("解除後從最高收盤回落 15% 才賣", lambda r: trail(r, "peak15")),
         ("收在高點附近→跌破 10 日線才賣，否則最後一天賣", lambda r: trail(r, "ma10") if r.near_hi else r.i_e),
         ("收在高點附近→回落 10% 才賣，否則最後一天賣", lambda r: trail(r, "peak10") if r.near_hi else r.i_e),
         ("再次處置→跌破 10 日線才賣，第一次→最後一天賣", lambda r: trail(r, "ma10") if r.again else r.i_e),
         ("處置中就用：從最高收盤回落 10% 就賣（不等結束）", lambda r: trail(r, "peak10", start=r.i_s))]
lines += ["", "== 二、整筆交易（處置首日開盤買），不同出場規則，已扣成本",
          f"{'出場規則':<36}{'筆數':>5}{'勝率':>6}{'平均':>8}{'中位':>8}{'最差5%':>8}{'平均抱':>7}{'每天':>8}{'比隨機':>8}{'前半':>8}{'後半':>8}  每年比隨機"]
rows2 = []
for lab, fn in RULES:
    t = []
    for r in d.itertuples():
        k = int(fn(r))
        if k >= N or not np.isfinite(T[k, r.j]):
            continue
        if lab.startswith("現在") is False and r.i_e + 20 >= N:      # 還沒走完的事件不算，和其他規則同一批
            continue
        x = T[k, r.j] / O[r.i_s, r.j] - 1
        m = L[r.i_s - 1]
        b = np.nanmean(T[k][m] / O[r.i_s][m] - 1)
        t.append((days[r.i_s], x - COST, x - b, k - r.i_s + 1))
    t = pd.DataFrame(t, columns=["d", "ret", "x", "n"])
    h = t["d"].sort_values().iloc[len(t) // 2]
    rows2.append({"出場規則": lab, "筆數": len(t), "勝率": (t["ret"] > 0).mean(), "平均": t["ret"].mean(), "中位數": t["ret"].median(),
                  "最差5%": t["ret"].quantile(0.05), "平均抱": t["n"].mean(), "每天": t["ret"].sum() / t["n"].sum(), "比隨機": t["x"].mean()})
    lines.append(f"{lab:<36}{len(t):>5}{(t['ret'] > 0).mean():>6.0%}{t['ret'].mean():>+8.2%}{t['ret'].median():>+8.2%}{t['ret'].quantile(0.05):>+8.1%}"
                 f"{t['n'].mean():>7.0f}{t['ret'].sum() / t['n'].sum():>+8.2%}{t['x'].mean():>+8.2%}{t[t['d'] < h]['x'].mean():>+8.2%}{t[t['d'] >= h]['x'].mean():>+8.2%}  " + yearly(t, "x"))
text = "\n".join(lines)
print(text)
out = ROOT / "results" / "處置股出場"
out.mkdir(parents=True, exist_ok=True)
(out / "disposition_exit.txt").write_text(text, encoding="utf-8")
pd.DataFrame(rows1).to_csv(out / "after_release.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(rows2).to_csv(out / "exit_rules.csv", index=False, encoding="utf-8-sig")
