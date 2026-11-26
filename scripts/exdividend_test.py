"""除權息前後的事件回測（2026-11-26）：除息前卡位、參加除息、除息後填息，和同期間隨機買股比。

用還原報酬（除息日用 收盤÷參考價，把股利加回來；股利所得稅、二代健保沒扣）。
息值＝(除息前收盤 − 參考價) ÷ 除息前收盤。只看流動性夠（20 日均量 >= 500 張、股價 >= 10）的股票。
假設除息日至少 10 個交易日前已公告（台股通常提前一個月以上）。成本 0.6%。
比隨機：同一進場日到同一出場日，所有流動性股票的平均。
    python scripts/exdividend_test.py
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
ex["yld"] = 1 - ex["ref"] / ex["prev_close"]
ret = raw.pct_change(fill_method=None)
for r in ex.itertuples():
    c = raw.at[r.date, r.stock_id]
    if np.isfinite(c):
        ret.at[r.date, r.stock_id] = c / r.ref - 1
ret = ret.where(ret.abs() <= 0.11, 0.0).fillna(0)
tri = (1 + ret).cumprod().where(raw.notna())          # 收盤還原指數
tro = tri / raw * ro                                   # 開盤的還原價（除息日開盤已是除息後）
for r in ex.itertuples():                              # 除息日開盤：還原因子要用當天的
    pass
liquid = (vol.rolling(20).mean() >= 500) & (raw >= 10)
pos = {d: i for i, d in enumerate(days)}
T, O, L = tri.values, tro.values, liquid.values
col = {s: j for j, s in enumerate(raw.columns)}
base_cache = {}


def base(i0, i1):
    if (i0, i1) not in base_cache:
        m = L[i0 - 1]
        base_cache[(i0, i1)] = np.nanmean(T[i1][m] / O[i0][m] - 1)
    return base_cache[(i0, i1)]


def run(name, a, b, ymin, ymax=1.0):
    """進場：除息日 + a 的開盤；出場：除息日 + b 的收盤。"""
    rows = []
    for r in ex[(ex["yld"] >= ymin) & (ex["yld"] < ymax)].itertuples():
        i, j = pos[r.date], col[r.stock_id]
        i0, i1 = i + a, i + b
        if i0 < 21 or i1 >= len(days) or not L[i0 - 1, j]:
            continue
        g = T[i1, j] / O[i0, j] - 1
        if np.isfinite(g):
            rows.append((days[i0], g - COST, g - base(i0, i1)))
    t = pd.DataFrame(rows, columns=["d", "ret", "x"])
    if len(t) < 30:
        return f"{name:<26} 筆數不足"
    yr = t.groupby(t["d"].dt.year)["x"].mean()
    return (f"{name:<26}{len(t):>6}{(t['ret'] > 0).mean():>6.0%}{t['ret'].mean():>+8.2%}{t['x'].mean():>+8.2%}  "
            + " ".join(f"{y}:{v:+.2%}" for y, v in yr.items()))


lines = [f"{days[0]:%Y-%m-%d} ~ {days[-1]:%Y-%m-%d}，除權息事件 {len(ex)} 筆；成本 {COST:.1%}",
         f"{'做法':<26}{'筆數':>6}{'勝率':>6}{'扣成本':>8}{'比隨機':>8}  每年比隨機"]
for ymin, ymax, lab in [(0.03, 1, "息值>=3%"), (0.05, 1, "息值>=5%"), (0.0, 0.03, "息值<3%")]:
    lines.append(f"-- {lab}")
    for name, a, b in [("除息前 10 天買、前 1 天賣", -10, -1), ("除息前 5 天買、前 1 天賣", -5, -1), ("除息前 5 天買、除息日賣", -5, 0),
                       ("除息前 1 天買、除息日賣", -1, 0), ("除息前 1 天買、後 5 天賣", -1, 5), ("除息前 1 天買、後 20 天賣", -1, 20),
                       ("除息日買、後 5 天賣", 0, 5), ("除息日買、後 10 天賣", 0, 10), ("除息日買、後 20 天賣", 0, 20), ("除息日買、後 40 天賣", 0, 40)]:
        lines.append(run(name, a, b, ymin, ymax))
text = "\n".join(lines)
print(text)
(ROOT / "results/大盤以外/exdividend.txt").write_text(text, encoding="utf-8")
