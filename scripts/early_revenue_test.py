"""營收提早進場的上限測試：如果在公司實際公布營收的當天（而不是等到 11 號）就買，最多能多賺多少？

我們沒有每家公司的歷史公布日，所以做「上限」：假裝月初第 K 個交易日就知道營收（會偷看未來，結果只當天花板）。
如果連這個天花板都沒比 11 號進場好多少，就不值得補抓公布日、改成每天看。
事件：營收加速（年增 >= 30%、比前 3 月平均多 10 點、創 12 個月新高；流動性夠），不加「股價已反應」條件，
因為提早進場時還看不到反應。隔天開盤買，抱 60 天，扣 0.6% 成本，和同日隨機買股比。
    python scripts/early_revenue_test.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from twscreener.backtest import COST  # noqa: E402
from twscreener.fundamentals import revenue_features  # noqa: E402
from twscreener.market import load_panel  # noqa: E402
from twscreener.simulate import prices  # noqa: E402

raw, adj, opn, vol = prices(load_panel())
days = adj.index
liquid = (vol.rolling(20).mean() >= 500) & (raw >= 10)
rev = revenue_features(pd.read_csv(ROOT / "data/revenue.csv", dtype={"stock_id": str})).sort_values(["stock_id", "ym"])
rev["prev3"] = rev.groupby("stock_id")["yoy"].transform(lambda s: s.shift(1).rolling(3).mean())
ev = rev[(rev["yoy"] >= 0.3) & (rev["yoy"] - rev["prev3"] >= 0.1) & rev["rev_high12"].astype(bool)]
H = 60
rows = []
for ym, g in ev.groupby("ym"):
    m1 = pd.Timestamp(ym + "-01") + pd.DateOffset(months=1)
    month_days = days[(days >= m1) & (days < m1 + pd.DateOffset(months=1))]
    after11 = days[days >= m1 + pd.Timedelta(days=10)]
    if len(month_days) < 8 or len(after11) == 0:
        continue
    variants = {"月初第1個交易日": month_days[0], "第3個交易日": month_days[2], "第5個交易日": month_days[4],
                "11號後（現在）": after11[0]}
    for name, d in variants.items():
        i = days.get_loc(d)
        if i + H + 1 >= len(days):
            continue
        ids = [s for s in g["stock_id"] if s in adj.columns and liquid.at[d, s]]
        ent = opn.iloc[i + 1]
        r = adj.iloc[i + H] / ent - 1 - COST
        base = r[liquid.iloc[i]].mean()
        for s in ids:
            if pd.notna(r[s]):
                rows.append({"ym": ym, "variant": name, "stock_id": s, "r": r[s], "x": r[s] - base})
t = pd.DataFrame(rows)
t["year"] = t["ym"].str[:4]


def stat(g):
    return pd.Series({"件數": len(g), "勝率": f"{(g['r'] > 0).mean():.0%}", "平均": f"{g['r'].mean():+.1%}",
                      "中位數": f"{g['r'].median():+.1%}", "每月一票超額": f"{g.groupby('ym')['x'].mean().mean():+.1%}"})


order = ["月初第1個交易日", "第3個交易日", "第5個交易日", "11號後（現在）"]
pd.set_option("display.width", 200)
out = ["== 全期（60 日）", t.groupby("variant").apply(stat).loc[order].to_string(), "", "== 每年每月一票超額"]
yy = t.groupby(["variant", "year"]).apply(lambda g: g.groupby("ym")["x"].mean().mean()).unstack().loc[order]
out.append(yy.map(lambda v: f"{v:+.1%}").to_string())
text = "\n".join(out)
print(text)
(ROOT / "results" / "重測_2022" / "early_revenue.txt").write_text(text, encoding="utf-8")
