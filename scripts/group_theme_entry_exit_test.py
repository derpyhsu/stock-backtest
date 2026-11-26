"""跟風族群怎麼上下車（2026-11-26）。延續 group_limitup_test.py：隔天直接追沒有優勢，改測別的上車點和下車規則。

真族群漲停日＝同產業當天漲停 3 檔以上，而且占該產業有流動性股票的 10% 以上（排除大產業湊數）。
上車（都是訊號日收盤判斷、隔天開盤買，開盤漲停買不到）：
  E0 族群漲停的漲停股，隔天追
  E1 拉回不破：漲停後 10 日內，收盤曾跌回漲停日收盤的 97% 以下、但沒跌破漲停日最低，之後第一次收盤站回漲停日收盤
  E2 整理後再過高：漲停後第 3~20 日，第一次收盤創漲停日以來新高（中間至少 2 天沒創高）
  E3 題材延續：該產業近 10 日有 3 天以上是真族群漲停日，買當天漲停的
  E4 題材延續＋拉回不破（E1 且產業近 20 日有 3 天以上真族群漲停）
  E5 題材延續時買族群裡近 20 日最強的 3 檔（不必漲停）
下車：抱 5／10／20 天；收盤跌破 5 日線；收盤跌破 10 日線；跌破進場後最高收盤的 10%。最多抱 40 天。隔天開盤賣。
比大盤＝減掉同一段期間全體有流動性股票等權平均（兩邊都扣 0.6%）。只用 2022 年以後。
    python scripts/group_theme_entry_exit_test.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from twscreener.market import load_panel  # noqa: E402
from twscreener.simulate import prices  # noqa: E402

COST, MAXH = 0.006, 40
p = load_panel()
p = p[p["date"] >= "2022-01-01"]
raw, adj, opn, vol = prices(p)
piv = lambda col: p.pivot_table(index="date", columns="stock_id", values=col).reindex(index=raw.index, columns=raw.columns)  # noqa: E731
ro, rl = piv("open"), piv("low")
days = raw.index
liquid = (vol.rolling(20).mean() >= 500) & (raw >= 10)
can_buy = (ro.shift(-1) < raw * 1.095).values
chg = raw / raw.shift(1) - 1
lim = (chg >= 0.095) & liquid
ind = pd.read_csv(ROOT / "data" / "industry.csv", dtype={"stock_id": str}).set_index("stock_id")["industry"].reindex(raw.columns).fillna("其他")
cnt = lim.T.groupby(ind.values).sum().T
mem = liquid.T.groupby(ind.values).sum().T
grp = (cnt >= 3) & (cnt / mem.clip(lower=1) >= 0.10)
if "其他" in grp.columns:
    grp["其他"] = False
to_stock = lambda g: pd.DataFrame({s: g[ind[s]] for s in raw.columns}, index=days)  # noqa: E731
hot = to_stock(grp)
persist10 = to_stock(grp.astype(float).rolling(10).sum() >= 3)
persist20 = to_stock(grp.astype(float).rolling(20).sum() >= 3)

A, O, C = adj.values, opn.values, adj.values
ma5, ma10 = adj.rolling(5).mean().values, adj.rolling(10).mean().values
adj_low = (rl * adj / raw).values
n, m = A.shape
mkt = (adj.pct_change().where(liquid.shift(1).fillna(False))).mean(axis=1).fillna(0)
mkt_idx = (1 + mkt).cumprod().values
L, HOT = lim.values, hot.values

# 事件：真族群漲停日的漲停股
ev = np.argwhere(L & HOT)
E = {k: [] for k in ("E0 隔天追", "E1 拉回不破再站回", "E2 整理後再過高", "E3 題材延續才追漲停", "E4 題材延續＋拉回不破", "E5 題材延續買族群最強 3 檔")}
P10, P20 = persist10.values, persist20.values
for i, j in ev:
    E["E0 隔天追"].append((i, j))
    if P10[i, j]:
        E["E3 題材延續才追漲停"].append((i, j))
    c0, l0 = A[i, j], adj_low[i, j]
    dipped, done1, done2, hi, flat = False, False, False, c0, 0
    for t in range(i + 1, min(i + 21, n)):
        c = A[t, j]
        if not np.isfinite(c):
            continue
        if not done1 and t <= i + 10:
            if c < l0:
                done1 = True               # 跌破漲停日最低：放棄
            elif c <= c0 * 0.97:
                dipped = True
            elif dipped and c >= c0:
                E["E1 拉回不破再站回"].append((t, j))
                if P20[t, j]:
                    E["E4 題材延續＋拉回不破"].append((t, j))
                done1 = True
        if not done2:
            if c > hi:
                if flat >= 2 and t >= i + 3:
                    E["E2 整理後再過高"].append((t, j))
                    done2 = True
                hi, flat = c, 0
            else:
                flat += 1
r20 = (adj / adj.shift(20) - 1).where(liquid)
for i in np.where(grp.any(axis=1).values)[0]:
    for g in grp.columns[grp.iloc[i].values]:
        if (grp[g].iloc[max(0, i - 9):i + 1].sum()) < 3:
            continue
        s = r20.iloc[i][ind.values == g].dropna().nlargest(3)
        E["E5 題材延續買族群最強 3 檔"] += [(i, raw.columns.get_loc(x)) for x in s.index]

EXITS = ("抱 5 天", "抱 10 天", "抱 20 天", "跌破 5 日線", "跌破 10 日線", "從高點回 10%")


def trade(i, j, rule):
    """i 日收盤訊號，i+1 開盤買。回傳 (報酬, 抱幾天, 同期全體報酬)。"""
    if i + 1 >= n or not can_buy[i, j] or not np.isfinite(O[i + 1, j]):
        return None
    e = O[i + 1, j]
    fixed = {"抱 5 天": 5, "抱 10 天": 10, "抱 20 天": 20}.get(rule)
    peak = -np.inf
    for h in range(1, MAXH + 1):
        t = i + h
        if t >= n:
            return None
        c = C[t, j]
        if not np.isfinite(c):
            continue
        peak = max(peak, c)
        if fixed:
            if h == fixed:
                return c / e - 1 - COST, h, mkt_idx[t] / mkt_idx[i] - 1 - COST
            continue
        out = (rule == "跌破 5 日線" and c < ma5[t, j]) or (rule == "跌破 10 日線" and c < ma10[t, j]) or (rule == "從高點回 10%" and c <= peak * 0.9)
        if out or h == MAXH:
            if t + 1 >= n or not np.isfinite(O[t + 1, j]):
                return None
            return O[t + 1, j] / e - 1 - COST, h + 1, mkt_idx[t + 1] / mkt_idx[i] - 1 - COST
    return None


rows = []
for name, lst in E.items():
    lst = sorted(set(lst))
    for rule in EXITS:
        res = [(days[i], *r) for i, j in lst if (r := trade(i, j, rule))]
        if len(res) < 30:
            continue
        d = pd.DataFrame(res, columns=["date", "ret", "hold", "mkt"])
        d["x"] = d["ret"] - d["mkt"]
        day = d.groupby("date")["x"].mean()               # 每天一票
        yr = day.groupby(day.index.year).mean()
        rows.append({"上車": name, "下車": rule, "筆數": len(d), "訊號天數": len(day), "勝率": (d["ret"] > 0).mean(), "平均": d["ret"].mean(),
                     "中位數": d["ret"].median(), "平均抱幾天": d["hold"].mean(), "比大盤": day.mean(),
                     "贏的年數": f"{(yr > 0).sum()}/{len(yr)}", "每年": " ".join(f"{y % 100}:{v:+.1%}" for y, v in yr.items())})
t = pd.DataFrame(rows)
out = ROOT / "results" / "消息面族群"
out.mkdir(parents=True, exist_ok=True)
t.to_csv(out / "theme_entry_exit.csv", index=False, encoding="utf-8-sig")
lines = [f"{days[0]:%Y-%m-%d} ~ {days[-1]:%Y-%m-%d}；真族群漲停日 {int(grp.sum().sum())} 個（產業×日）；成本 {COST:.1%}；比大盤＝每天一票"]
for title, col, fmt in (("比大盤", "比大盤", lambda v: f"{v:+.1%}"), ("勝率", "勝率", lambda v: f"{v:.0%}"), ("平均報酬", "平均", lambda v: f"{v:+.1%}"),
                        ("平均抱幾天", "平均抱幾天", lambda v: f"{v:.0f}"), ("贏的年數", "贏的年數", str), ("筆數", "筆數", lambda v: f"{v:.0f}")):
    pv = t.pivot(index="上車", columns="下車", values=col).reindex(index=list(E), columns=list(EXITS))
    lines += ["", f"== {title}", pv.map(lambda v: fmt(v) if pd.notna(v) else "").to_string()]
lines += ["", "== 抱 10 天逐年"]
for _, r in t[t["下車"] == "抱 10 天"].iterrows():
    lines.append(f"{r['上車']}：{r['每年']}（訊號 {r['訊號天數']} 天）")
text = "\n".join(lines)
print(text)
(out / "theme_entry_exit.txt").write_text(text, encoding="utf-8")
