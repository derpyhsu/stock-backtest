"""族群輪動＋三大法人回測：法人在族群裡的買賣，能不能提早抓到進場點、告訴我們何時出場？

族群定義同 scripts/sector_momentum_test.py（產業分類、股價連動的同族群）。
法人＝外資＋投信買賣超（張），用族群成員 20 日均量加總正規化：族群 5 日法人買超 ÷ 族群 20 日均量。
訊號（訊號日收盤，只用當天以前的資料）：
- 確認：族群 3 日比大盤多漲 5% 以上、60% 成員上漲、3 天前 20 日漲幅 < 10%（同前一支），且族群 5 日法人買超 >= FLOW。
- 提早：族群 5 日法人買超 >= FLOW_EARLY、60% 以上成員 5 日法人買超，但族群 3 日比大盤多漲不到 2%（股價還沒動）。
- 對照：族群轉強但法人 5 日賣超。
出場：
- 固定：第 3、5、10 天收盤。
- 法人轉賣：進場後族群法人連 2 天賣超就在第 2 天收盤賣，最多抱 10 天。
隔天開盤買（開盤已漲 9.5% 以上買不到，略過），扣 0.6% 成本，和同日隨機買股比超額。
    python scripts/sector_flow_test.py
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

HOT, FLOW, FLOW_EARLY, QUIET = 0.05, 0.2, 0.3, 0.02
HOLDS = (3, 5, 10)
MAXH = 10
OUT = ROOT / "results" / "族群輪動"
OUT.mkdir(parents=True, exist_ok=True)

panel = load_panel()
raw, adj, opn, vol = prices(panel)
inst = panel.assign(n=panel["foreign_net"].fillna(0) + panel["trust_net"].fillna(0)) \
    .pivot_table(index="date", columns="stock_id", values="n").reindex(index=adj.index, columns=adj.columns)
days, cols = adj.index, list(adj.columns)
col_idx = {c: k for k, c in enumerate(cols)}
vol20 = vol.rolling(20).mean()
liquid = (vol20 >= 500) & (raw >= 10)
ret3 = adj / adj.shift(3) - 1
mkt3 = ret3.where(liquid).mean(axis=1)
entry = opn.shift(-1)
fwd = {h: adj.shift(-h) / entry - 1 - COST for h in HOLDS}
base = {h: fwd[h].where(liquid).mean(axis=1) for h in HOLDS}
A = {"liq": liquid.to_numpy(), "r3": ret3.to_numpy(), "u3": (ret3 > 0).to_numpy(float),
     "lu": (opn.shift(-1) >= adj * 1.095).to_numpy(bool), "r20b": (adj.shift(3) / adj.shift(23) - 1).to_numpy(),
     "mkt3": mkt3.to_numpy(), "flow5": inst.rolling(5).sum().to_numpy(), "inst": inst.fillna(0).to_numpy(),
     "v20": vol20.to_numpy(), "adj": adj.to_numpy(), "entry": entry.to_numpy()}
for h in HOLDS:
    A[f"f{h}"], A[f"b{h}"] = fwd[h].to_numpy(), base[h].to_numpy()
# 法人轉賣出場的基準：同日隨機買股抱一樣天數，近似用 5 日基準
A["bexit"] = base[5].to_numpy()


def exit_on_flow(i, members):
    """進場（i+1 開盤）後族群法人連 2 天賣超就賣；回傳報酬與天數。"""
    neg = 0
    for t in range(i + 1, i + 1 + MAXH):
        neg = neg + 1 if A["inst"][t, members].sum() < 0 else 0
        if neg >= 2 or t == i + MAXH:
            return t - i
    return MAXH


def run(groups_by_day, name):
    rows = []
    for i in range(140, len(days) - MAXH - 1):
        keys, G = groups_by_day(i)
        liq = A["liq"][i] & ~np.isnan(A["r3"][i]) & ~np.isnan(A["r20b"][i]) & ~np.isnan(A["v20"][i])
        r3, u3 = np.nan_to_num(A["r3"][i]), A["u3"][i]
        r20, f5, v20 = np.nan_to_num(A["r20b"][i]), np.nan_to_num(A["flow5"][i]), np.nan_to_num(A["v20"][i])
        Gl = G & liq
        cnt = Gl.sum(1)
        ok = cnt >= 5
        if not ok.any():
            continue
        c = np.maximum(cnt, 1)
        gx = (Gl @ r3) / c - A["mkt3"][i]
        br, g20 = (Gl @ u3) / c, (Gl @ r20) / c
        gflow = (Gl @ f5) / np.maximum(Gl @ v20, 1)
        fbr = (Gl @ (f5 > 0).astype(float)) / c
        hot = ok & (gx >= HOT) & (br >= 0.6) & (g20 < 0.10)
        sig = {"確認（轉強＋法人買）": hot & (gflow >= FLOW),
               "對照（轉強＋法人賣）": hot & (gflow < 0),
               "提早（法人買、股價未動）": ok & (gflow >= FLOW_EARLY) & (fbr >= 0.6) & (gx < QUIET)}
        for how, mask in sig.items():
            for k in np.where(mask)[0]:
                m = np.where(Gl[k])[0]
                for j in m:
                    if A["lu"][i, j] or np.isnan(A["entry"][i, j]):
                        continue
                    row = {"date": days[i], "group": keys[k], "how": how, "stock_id": cols[j]}
                    for h in HOLDS:
                        row[f"r{h}"] = A[f"f{h}"][i, j]
                        row[f"x{h}"] = row[f"r{h}"] - A[f"b{h}"][i]
                    n = exit_on_flow(i, m)
                    row["r法人出"] = A["adj"][i + n, j] / A["entry"][i, j] - 1 - COST
                    row["x法人出"] = row["r法人出"] - A["bexit"][i]
                    row["天數法人出"] = n
                    rows.append(row)
    t = pd.DataFrame(rows).dropna(subset=[f"r{h}" for h in HOLDS])
    t = t.drop_duplicates(["date", "how", "stock_id"])
    t["kind"] = name
    print(name, "完成", len(t), flush=True)
    return t


ind = load_industry()
groups = {}
for s in cols:
    g = ind.get(s)
    if g and g not in ("其他", "存託憑證"):
        groups.setdefault(g, []).append(s)
ind_keys = list(groups)
ind_G = np.zeros((len(ind_keys), len(cols)), bool)
for k, g in enumerate(ind_keys):
    ind_G[k, [col_idx[x] for x in groups[g]]] = True
industry = run(lambda i: (ind_keys, ind_G), "產業")

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
t["half"] = np.where(t["date"] <= t["date"].quantile(0.5), "前段", "後段")
EXITS = [str(h) for h in HOLDS] + ["法人出"]


def stat(g):
    out = {"件數": len(g), "天數": g["date"].nunique()}
    for h in EXITS:
        out[f"{h}勝率"] = f"{(g[f'r{h}'] > 0).mean():.0%}"
        out[f"{h}平均"] = f"{g[f'r{h}'].mean():+.2%}"
        out[f"{h}每天一票超額"] = f"{g.groupby('date')[f'x{h}'].mean().mean():+.2%}"
    out["法人出平均天數"] = f"{g['天數法人出'].mean():.1f}"
    return pd.Series(out)


pd.set_option("display.width", 300)
pd.set_option("display.max_columns", 40)
lines = []
for (kind, how), g in t.groupby(["kind", "how"]):
    lines.append(f"\n== {kind}・{how}（{g['date'].min():%Y-%m-%d} ~ {g['date'].max():%Y-%m-%d}）")
    tab = pd.concat([pd.concat({"全期": stat(g)}, axis=1).T, g.groupby("year").apply(stat), g.groupby("half").apply(stat)])
    lines.append(tab.to_string())
text = "\n".join(lines)
print(text)
(OUT / "sector_flow.txt").write_text(text, encoding="utf-8")
