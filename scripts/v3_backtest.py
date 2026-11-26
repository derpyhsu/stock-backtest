"""v4＝v3 再加處置股（三份資金各 1/3）；v3（資金一半 v2 營收加速、一半 S1 一年新高，同一個帳戶）用 portfolio.run 實際回測，和 v2、加權比（2026-11-26）。
    python scripts/v3_backtest.py
"""
import sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import twscreener.portfolio as P  # noqa: E402
from twscreener.market import load_market_index  # noqa: E402
inputs = P._inputs()
days = inputs[0].index
taiex = load_market_index()["taiex"].reindex(days).ffill()
navs, tr = {}, {}
for v in ("v2", "v3", "v4"):
    a = P.run(days[days >= "2023-01-01"], P.Account(), inputs, version=v)
    n = pd.DataFrame(a.nav)
    navs[v] = n.set_index(pd.to_datetime(n["date"]))["淨值"]
    tr[v] = pd.DataFrame(a.trades)
navs = pd.DataFrame(navs)
navs["加權指數"] = taiex.reindex(navs.index)
navs.to_csv(ROOT / "results" / "持有天數" / "v3_v4_navs.csv")
tr["v4"].to_csv(ROOT / "results" / "持有天數" / "v4_trades.csv", index=False, encoding="utf-8-sig")
def summ(s):
    s = s.dropna() / s.dropna().iat[0]
    return f"{s.iat[-1] - 1:+.1%}／{(s / s.cummax() - 1).min():.1%}"
lines = []
for st in ("2023-01-01", "2023-04-01"):
    seg = navs[navs.index >= st]
    h = seg.index[len(seg) // 2]
    lines.append(f"\n期間 {seg.index[0].date()} ~ {seg.index[-1].date()}（報酬／最大回撤）")
    for k in seg.columns:
        yr = "  ".join(f"{y} {summ(g[k])}" for y, g in seg.groupby(seg.index.year))
        lines.append(f"{k:6s} 全期 {summ(seg[k])}  前段 {summ(seg[k][seg.index < h])}  後段 {summ(seg[k][seg.index >= h])}  {yr}")
for v in ("v3", "v4"):
  t = tr[v]
  lines.append(f"\n{v} 交易 {len(t)} 筆（" + "、".join(f"{k} {n} 筆" for k, n in t["strategy"].value_counts().items()) + f"），勝率 {(t['報酬'] > 0).mean():.0%}")
  b = pd.to_datetime(t["買進日"])
  lines.append(f"有買進的週數 {b.dt.to_period('W').nunique()}，有買進的月份 {b.dt.to_period('M').nunique()}")
text = "\n".join(lines)
print(text)
(ROOT / "results" / "持有天數" / "v3_v4_backtest.txt").write_text(text, encoding="utf-8")
