"""族群短線輪動回測：某個族群剛開始集體轉強，隔天開盤買進，抱 3 或 5 天，有沒有贏同日隨機買股？

族群兩種定義（都只用訊號日以前的資料）：
- 產業：證交所產業分類（data/industry.csv），太粗，抓不到題材股。
- 連動：每月初用過去 120 天日報酬，找每檔股票相關係數最高的 10 檔當「同族群」（自動找出一起漲跌的題材）。

訊號（訊號日收盤，只看流動性夠的股票：20 日均量 >= 500 張、股價 >= 10）：
- 族群 3 日等權報酬比大盤多 HOT 以上，且族群裡 60% 以上的股票 3 日上漲（集體轉強，不是一檔拉）。
- 「剛開始」：3 天前，族群的 20 日報酬還不到 10%（已經漲一大段的不算）。
買法：
- 全買：族群裡所有股票。
- 領頭：族群裡 3 日漲最多的 3 檔。
- 補漲：族群轉強、但自己 3 日漲幅低於族群中位數的股票。
隔天開盤買（開盤已漲 9.5% 以上買不到，略過），第 3、5 天收盤賣，扣 0.6% 成本。
基準：同一天所有流動性夠的股票（隨機買股），看超額報酬；也分年、前後段看是不是只贏一段。
    python scripts/sector_momentum_test.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from twscreener.backtest import COST  # noqa: E402
from twscreener.fundamentals import load_industry  # noqa: E402
from twscreener.market import load_panel  # noqa: E402
from twscreener.simulate import prices  # noqa: E402

HOT = 0.05
HOLDS = (3, 5)
OUT = ROOT / "results" / "族群輪動"
OUT.mkdir(parents=True, exist_ok=True)

raw, adj, opn, vol = prices(load_panel())
days = adj.index
liquid = (vol.rolling(20).mean() >= 500) & (raw >= 10)
ret3 = adj / adj.shift(3) - 1
up3 = ret3 > 0
mkt3 = ret3.where(liquid).mean(axis=1)
# 隔天開盤買、第 h 天收盤賣
entry = opn.shift(-1)
limit_up = opn.shift(-1) >= adj * 1.095  # 隔天開盤比今天收盤漲 9.5% 以上（買不到）
fwd = {h: adj.shift(-h) / entry - 1 - COST for h in HOLDS}
base = {h: fwd[h].where(liquid).mean(axis=1) for h in HOLDS}  # 同日隨機買股


cols = list(adj.columns)
col_idx = {c: k for k, c in enumerate(cols)}
A = {"liq": liquid.to_numpy(), "r3": ret3.to_numpy(), "u3": up3.to_numpy(), "lu": limit_up.fillna(False).to_numpy(bool),
     "r20b": (adj.shift(3) / adj.shift(23) - 1).to_numpy(), "mkt3": mkt3.to_numpy()}
for h in HOLDS:
    A[f"f{h}"] = fwd[h].to_numpy()
    A[f"b{h}"] = base[h].to_numpy()


def run(groups_by_day, name):
    """groups_by_day(i) -> (keys, 成員矩陣 K x N)。族群報酬用矩陣一次算完，只對轉強的族群逐檔展開。"""
    rows = []
    for i in range(140, len(days) - max(HOLDS) - 1):
        keys, G = groups_by_day(i)
        liq = A["liq"][i] & ~np.isnan(A["r3"][i]) & ~np.isnan(A["r20b"][i])
        r3 = np.nan_to_num(A["r3"][i]); u3 = A["u3"][i].astype(float); r20 = np.nan_to_num(A["r20b"][i])
        Gl = G & liq
        cnt = Gl.sum(1)
        ok = cnt >= 5
        if not ok.any():
            continue
        c = np.maximum(cnt, 1)
        gr, br, g20 = (Gl @ r3) / c, (Gl @ u3) / c, (Gl @ r20) / c
        hot = np.where(ok & (gr - A["mkt3"][i] >= HOT) & (br >= 0.6) & (g20 < 0.10))[0]
        for k in hot:
            m = np.where(Gl[k])[0]
            med = np.median(r3[m])
            picks = {"全買": m, "領頭": m[np.argsort(-r3[m])[:3]], "補漲": m[r3[m] < med]}
            for how, ss in picks.items():
                for j in ss:
                    if A["lu"][i, j]:
                        continue
                    row = {"date": days[i], "group": keys[k], "how": how, "stock_id": cols[j]}
                    for h in HOLDS:
                        row[f"r{h}"] = A[f"f{h}"][i, j]
                        row[f"x{h}"] = row[f"r{h}"] - A[f"b{h}"][i]
                    rows.append(row)
    t = pd.DataFrame(rows).dropna(subset=[f"r{h}" for h in HOLDS])
    # 同一天同一檔被多個族群選到只算一次（連動族群會重疊）
    t = t.drop_duplicates(["date", "how", "stock_id"])
    t["kind"] = name
    return t


# 產業族群（固定）
ind = load_industry()
ind_groups = {}
for s in adj.columns:
    g = ind.get(s)
    if g and g not in ("其他", "存託憑證"):
        ind_groups.setdefault(g, []).append(s)
ind_keys = list(ind_groups)
ind_G = np.zeros((len(ind_keys), len(cols)), bool)
for k, g in enumerate(ind_keys):
    ind_G[k, [col_idx[x] for x in ind_groups[g]]] = True
industry = run(lambda i: (ind_keys, ind_G), "產業")
print("產業族群完成", len(industry))

# 連動族群：每月第一個交易日重算，用過去 120 天
daily_ret = adj.pct_change(fill_method=None)
peer_cache = {}


def peers(i):
    d = days[i]
    key = (d.year, d.month)
    if key not in peer_cache:
        win = daily_ret.iloc[max(0, i - 120):i]
        ok = [k for k, s in enumerate(cols) if A["liq"][i - 1, k] and win[s].notna().sum() >= 100]
        x = win.iloc[:, ok].fillna(0).to_numpy()
        x = (x - x.mean(0)) / (x.std(0) + 1e-12)
        c = (x.T @ x) / len(x)
        np.fill_diagonal(c, -1)
        top = np.argsort(-c, axis=1)[:, :10]
        G = np.zeros((len(ok), len(cols)), bool)
        for r, k in enumerate(ok):
            G[r, k] = True
            G[r, [ok[j] for j in top[r] if c[r, j] >= 0.4]] = True
        peer_cache[key] = ([cols[k] for k in ok], G)
    return peer_cache[key]


corr = run(peers, "連動")

t = pd.concat([industry, corr])
t["year"] = t["date"].dt.year
cut = t["date"].quantile(0.5)
t["half"] = np.where(t["date"] <= cut, "前段", "後段")


def stat(g):
    out = {"件數": len(g), "訊號天數": g["date"].nunique()}
    for h in HOLDS:
        out[f"{h}日勝率"] = f"{(g[f'r{h}'] > 0).mean():.0%}"
        out[f"{h}日平均"] = f"{g[f'r{h}'].mean():+.2%}"
        out[f"{h}日超額"] = f"{g[f'x{h}'].mean():+.2%}"
        # 每天一票：避免某幾天訊號特別多主導結果
        out[f"{h}日每天一票超額"] = f"{g.groupby('date')[f'x{h}'].mean().mean():+.2%}"
    return pd.Series(out)


pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
lines = []
for (kind, how), g in t.groupby(["kind", "how"]):
    lines.append(f"\n== {kind}・{how}（{g['date'].min():%Y-%m-%d} ~ {g['date'].max():%Y-%m-%d}）")
    tab = pd.concat({"全期": stat(g)}, axis=1).T
    tab = pd.concat([tab, g.groupby("year").apply(stat), g.groupby("half").apply(stat)])
    lines.append(tab.to_string())
text = "\n".join(lines)
print(text)
(OUT / "sector_momentum.txt").write_text(text, encoding="utf-8")
t.to_csv(OUT / "trades.csv", index=False, encoding="utf-8-sig")
