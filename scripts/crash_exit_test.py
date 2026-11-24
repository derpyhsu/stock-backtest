"""腰斬事件（crash_hold_test 的 events.csv）之後怎麼賣：馬上賣 vs 等反彈幾 % 再賣 vs 站回月線再賣。
每種都設 60 個交易日期限；等反彈的版本另加「再跌 X% 停損」。報酬以事件日收盤為基準，同時扣同期所有股票平均（超額）。
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from twscreener.market import load_panel  # noqa: E402
from twscreener.simulate import prices  # noqa: E402
ROOT = Path(__file__).resolve().parent.parent
raw, adj, opn, vol = prices(load_panel())
adj = adj.loc["2022-01-01":]; vol = vol.reindex_like(adj); raw = raw.reindex_like(adj)
liquid = (vol.rolling(20).mean() >= 500) & (raw >= 10)
mkt = (adj.pct_change(fill_method=None).where(liquid.shift(1)).mean(axis=1).fillna(0) + 1).cumprod()
ma20 = adj.rolling(20).mean()
e = pd.read_csv(ROOT / "results/腰斬續抱/events.csv", parse_dates=["date"], dtype={"stock_id": str})
e = e[e["date"] <= adj.index[-61]]
days = adj.index
RULES = {"馬上賣": ("now",), "抱滿60天": ("hold",), "站回月線賣": ("ma",),
         "反彈10%賣(再跌15%停損)": ("up", .10, -.15), "反彈20%賣(再跌15%停損)": ("up", .20, -.15),
         "反彈10%賣(不停損)": ("up", .10, -9), "反彈20%賣(不停損)": ("up", .20, -9)}
def exit_ret(i, s, rule):
    p = adj[s].values; m = ma20[s].values; p0 = p[i]
    end = min(i + 60, len(p) - 1)
    if rule[0] == "now":
        return 0.0, i
    for j in range(i + 1, end + 1):
        if not np.isfinite(p[j]):
            continue
        r = p[j] / p0 - 1
        if rule[0] == "ma" and p[j] > m[j]:
            return r, j
        if rule[0] == "up" and (r >= rule[1] or r <= rule[2]):
            return r, j
    return p[end] / p0 - 1, end
rows = []
for _, ev in e.iterrows():
    i = days.get_loc(ev["date"])
    for k, rule in RULES.items():
        r, j = exit_ret(i, ev["stock_id"], rule)
        rows.append({"date": ev["date"], "營收": ev["營收"], "規則": k, "r": r, "x": r - (mkt.iat[j] / mkt.iat[i] - 1), "天數": j - i})
d = pd.DataFrame(rows)
d["月"] = d["date"].dt.to_period("M"); d["年"] = d["date"].dt.year
def show(df, title):
    print(f"\n== {title}")
    g = df.groupby("規則")
    mm = df.groupby(["規則", "月"])["x"].mean().groupby("規則")
    yy = df.groupby(["規則", "年"])["x"].mean().unstack().map("{:+.1%}".format)
    out = pd.DataFrame({"件數": g.size(), "平均報酬": g["r"].mean().map("{:+.1%}".format), "平均超額": g["x"].mean().map("{:+.1%}".format),
                        "每月一票超額": mm.mean().map("{:+.1%}".format), "平均持有天數": g["天數"].mean().round(0)})
    print(pd.concat([out, yy], axis=1).loc[list(RULES)].to_string())
pd.set_option("display.width", 250)
show(d, "全部腰斬事件")
show(d[d["營收"] == "年增>=20%"], "營收年增>=20%")
