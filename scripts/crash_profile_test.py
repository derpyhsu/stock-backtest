"""腰斬事件加入籌碼與估值：外資、投信近 5／20 日買賣超（佔 20 日均量）、融資 20 日增減、本益比，
看哪些條件下事後 60 日表現較好，並找出和指定股票（預設 2449）條件相近的事件。
籌碼 2024-06 起完整（2022～2024-05 補資料中），本益比 2024-11 起。
    python scripts/crash_profile_test.py 2449
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from twscreener.market import load_panel  # noqa: E402
from twscreener.simulate import prices  # noqa: E402
ROOT = Path(__file__).resolve().parent.parent
sid = sys.argv[1] if len(sys.argv) > 1 else "2449"
panel = load_panel()
raw, adj, opn, vol = prices(panel)
piv = lambda k: panel.pivot_table(index="date", columns="stock_id", values=k).reindex_like(adj)
fn, tn, mb = piv("foreign_net"), piv("trust_net"), piv("margin_bal")
v20 = vol.rolling(20).mean()
F = {"外資5日": fn.rolling(5).sum() / v20, "外資20日": fn.rolling(20).sum() / v20,
     "投信20日": tn.rolling(20).sum() / v20, "融資20日增減": mb / mb.shift(20) - 1}
val = []
for f in sorted((ROOT / "data/valuation").glob("*.csv")):
    v = pd.read_csv(f, dtype={"stock_id": str}); v["date"] = pd.to_datetime(f.stem); val.append(v)
val = pd.concat(val)
F["本益比"] = val.pivot_table(index="date", columns="stock_id", values="pe").reindex(adj.index).ffill(limit=5)
e = pd.read_csv(ROOT / "results/腰斬續抱/events.csv", parse_dates=["date"], dtype={"stock_id": str})
for k, df in F.items():
    e[k] = [df.at[d, s] if s in df.columns else np.nan for d, s in zip(e["date"], e["stock_id"])]
cur = {k: df[sid].dropna().iat[-1] if sid in df.columns and df[sid].notna().any() else np.nan for k, df in F.items()}
print(f"{sid} 現況：", {k: round(v, 3) for k, v in cur.items()})
h = e.dropna(subset=["x60", "外資20日"]).copy()
h["月"] = h["date"].dt.to_period("M")
print(f"\n有籌碼、且已有 60 日結果的事件：{len(h)} 件（{h['date'].min().date()} ~ {h['date'].max().date()}，{h['月'].nunique()} 個月）")
def grp(mask, name):
    g = h[mask]
    if len(g) == 0:
        return
    m = g.groupby("月")["x60"].mean()
    print(f"{name:28s} 件數 {len(g):4d}  60日平均 {g['r60'].mean():+.1%}  勝率 {(g['r60'] > 0).mean():.0%}  超額 {g['x60'].mean():+.1%}  每月一票超額 {m.mean():+.1%}（{(m > 0).mean():.0%} 月贏，{len(m)} 月）")
print("\n== 單一條件")
grp(h.index == h.index, "全部")
for k in ["外資5日", "外資20日", "投信20日"]:
    grp(h[k] > 0, f"{k}買超"); grp(h[k] <= 0, f"{k}賣超")
grp(h["融資20日增減"] < 0, "融資減少"); grp(h["融資20日增減"] >= 0, "融資增加")
grp(h["yoy"] >= 0.2, "營收年增>=20%"); grp(h["yoy"] < 0.2, "營收年增<20%")
pe = h.dropna(subset=["本益比"])
if len(pe):
    q = pe["本益比"].median()
    grp(h["本益比"] <= q, f"本益比<={q:.0f}"); grp(h["本益比"] > q, f"本益比>{q:.0f}")
print("\n== 和京元電條件相近（營收年增>=20%、外資近 5 日買超、融資減少）")
like = (h["yoy"] >= 0.2) & (h["外資5日"] > 0) & (h["融資20日增減"] < 0)
grp(like, "三條件都符合")
grp(like & (h["外資20日"] > 0), "＋外資20日也買超")
grp(like & (h["外資20日"] <= 0), "＋外資20日賣超")
print(h[like][["date", "stock_id", "yoy", "外資5日", "外資20日", "融資20日增減", "r60", "x60"]].round(3).to_string())
