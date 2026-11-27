"""「選股」這一層單獨測（2026-11-27）。接 scripts/youtube_nocc_universe_test.py：影片老師的買點沒用，但他們的選股條件疊起來的股票池贏隨機很多。
這裡把股票池寫成正式訊號（進池子的第一天算一筆，之前 20 天沒在池子裡），照標準做法測：收盤判斷、隔天開盤買、抱 N 天、扣 0.6%，
比隨機＝每天一票減同一天全部流動性股票平均。並拆開看是哪個條件在貢獻、換參數穩不穩。
  熱門＝當天成交值排全市場前 N 名
  大盤好＝加權指數在月線上且月線上揚
  主流領頭＝近 L 日漲幅前 K 名的產業裡，近 L 日漲幅排該產業前 M 名
    python scripts/selection_layer_test.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import twscreener.portfolio as P  # noqa: E402
from twscreener.market import load_market_index, load_panel  # noqa: E402
from twscreener.simulate import prices  # noqa: E402

H, COST = (20, 60, 120), 0.006
p = load_panel()
raw, adj, opn, vol = prices(p)
days = raw.index
ro = p.pivot_table(index="date", columns="stock_id", values="open").reindex(index=days, columns=raw.columns)
c = adj
liquid = (vol.rolling(20).mean() >= 500) & (raw >= 10)
can_buy = ro.shift(-1) < raw * 1.095
fwd = {h: adj.shift(-h) / opn.shift(-1) - 1 - COST for h in H}
base = {h: fwd[h].where(liquid).mean(axis=1) for h in H}
tx = load_market_index()["taiex"].reindex(days).ffill()
txma = tx.rolling(20).mean()
mkt_s = (tx > txma) & (txma > txma.shift(5))
mk = pd.DataFrame(np.repeat(mkt_s.values[:, None], raw.shape[1], axis=1), index=days, columns=raw.columns)
val = raw * vol
vrank = val.rank(axis=1, ascending=False)
ind = pd.Series(P.load_industry()).reindex(raw.columns).fillna("其他")
print("產業數", ind.nunique(), "；每個產業檔數中位", int(ind.value_counts().median()), flush=True)
hi240 = (c >= c.rolling(240).max()).astype(float).rolling(20).max().fillna(0).astype(bool)


def lead(L=20, K=5, M=3):
    r = (c / c.shift(L) - 1).where(liquid)
    ir = r.T.groupby(ind.values).mean().T
    top = (ir.rank(axis=1, ascending=False) <= K).reindex(columns=ind.values).set_axis(raw.columns, axis=1).fillna(False).astype(bool)
    return top & (r.T.groupby(ind.values).rank(ascending=False).T <= M)


recent = lambda s, n: s.astype(float).rolling(n).max().shift(1).fillna(0).astype(bool)  # noqa: E731
first = lambda s, n=20: s & ~recent(s, n)  # noqa: E731
hot = {n: vrank <= n for n in (50, 100, 200)}
L0 = lead()
pools = {
    "熱門 100": hot[100], "大盤好": mk, "主流領頭": L0,
    "熱門 100＋大盤好": hot[100] & mk, "熱門 100＋主流領頭": hot[100] & L0, "大盤好＋主流領頭": mk & L0,
    "三個都要（熱門 100＋大盤好＋主流領頭）": hot[100] & mk & L0,
    "三個都要，熱門改前 50": hot[50] & mk & L0, "三個都要，熱門改前 200": hot[200] & mk & L0,
    "三個都要，產業前 3 名": hot[100] & mk & lead(K=3), "三個都要，產業前 8 名": hot[100] & mk & lead(K=8),
    "三個都要，每個產業取前 5 檔": hot[100] & mk & lead(M=5), "三個都要，每個產業只取第 1 檔": hot[100] & mk & lead(M=1),
    "三個都要，漲幅改看 60 日": hot[100] & mk & lead(L=60), "三個都要，漲幅改看 10 日": hot[100] & mk & lead(L=10),
    "熱門 100＋主流領頭＋大盤不好": hot[100] & L0 & ~mk,
    "三個都要，但近 20 天沒創過一年新高": hot[100] & mk & L0 & ~hi240,
    "三個都要，而且近 20 天創過一年新高": hot[100] & mk & L0 & hi240,
}
half = days[len(days) // 2]
rows, lines2 = [], []
for name, pool in pools.items():
    for tag, s in (("進池第一天", first(pool & liquid)), ("在池子裡每天", pool)):
        if tag == "在池子裡每天" and name in ("大盤好",):
            continue
        m = (s & liquid & can_buy).fillna(False)
        rec = {"股票池": name, "算法": tag}
        for h in H:
            r = fwd[h].where(m)
            st = r.stack().dropna()
            if len(st) < 30:
                continue
            x = r.sub(base[h], axis=0).mean(axis=1).dropna()
            yr = x.groupby(x.index.year).mean()
            rec[f"{h}天"] = f"{x.mean():+.1%}"
            if h == 60:
                rec.update({"筆數": len(st), "有訊號天數": int(m.any(axis=1).sum()), "勝率": f"{(st > 0).mean():.0%}", "中位數": f"{st.median():+.1%}",
                            "每筆平均(不分天)": f"{(st - base[h].reindex(st.index.get_level_values(0)).values).mean():+.1%}",
                            "贏的年數": f"{(yr > 0).sum()}/{len(yr)}", "前半/後半": f"{x[x.index < half].mean():+.1%}/{x[x.index >= half].mean():+.1%}"})
                lines2.append(f"{name}・{tag}：" + " ".join(f"{y % 100}:{v:+.1%}" for y, v in yr.items()))
        rows.append(rec)
t = pd.DataFrame(rows)
text = (f"{days[0]:%Y-%m-%d} ~ {days[-1]:%Y-%m-%d}，成本 {COST:.1%}；20/60/120 天＝比隨機（每天一票）\n" + t.to_string(index=False)
        + "\n\n== 60 天逐年超額\n" + "\n".join(lines2))
print(text)
out = ROOT / "results" / "選股層"
out.mkdir(parents=True, exist_ok=True)
(out / "selection_layer.txt").write_text(text, encoding="utf-8")
