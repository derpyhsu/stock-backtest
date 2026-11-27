"""閒置現金怎麼用（2026-11-27）。v6（30／30／40）平均仍有三成多現金閒置，因為每份很少用滿。
做法：把各份的名目比例加大（加起來超過 100%），實際買進仍受手上現金限制，不借錢。
  A v6（30／30／40，處置股每筆佔那份 5%）
  B v6，處置股每筆 7.5%
  C 三份都放大 1.2 倍（36／36／48）
  D 三份都放大 1.4 倍（42／42／56）
  E 放大 1.2 倍，處置股每筆 6.25%
  F 只放大中長線（45／45／40）
  G 只放大處置股（30／30／60）
    python scripts/idle_cash_account_test.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import twscreener.portfolio as P  # noqa: E402
from twscreener.market import load_market_index  # noqa: E402

inputs = P._inputs()
raw = inputs[0]
days = raw.index
MODES = {"A": (0.3, 0.3, 0.4, 0.05), "B": (0.3, 0.3, 0.4, 0.075), "C": (0.36, 0.36, 0.48, 0.05), "D": (0.42, 0.42, 0.56, 0.05),
         "E": (0.36, 0.36, 0.48, 0.0625), "F": (0.45, 0.45, 0.4, 0.05), "G": (0.3, 0.3, 0.6, 0.05)}
NAME = {"v2": "營收加速", "s1": "一年新高", "disp": "處置股"}
orig_plan = P.Account.plan
EXPO = []


def plan(self, day, close, cands, prev=None, trend=None):
    nav = self.value(close)
    row = {"date": day}
    for k, sts in P.SLEEVES.items():
        h = self.pos[self.pos["strategy"].isin(sts)]
        row[k] = float((h["stock_id"].map(close).fillna(h["entry"]) * h["shares"]).sum()) / nav if len(h) else 0.0
    row["cash"] = self.cash / nav
    EXPO.append(row)
    return orig_plan(self, day, close, cands, prev, trend=trend)


P.Account.plan = plan
tx = load_market_index()["taiex"].reindex(days).ffill()
navs, info = {}, {}
for m, (a, b, c, w) in MODES.items():
    P.SLEEVE_W["v5"] = {"v2": a, "s1": b, "disp": c}
    P.DISP_W = w
    EXPO.clear()
    acct = P.run(days[days >= "2023-01-01"], P.Account(), inputs, version="v5")
    n = pd.DataFrame(acct.nav)
    navs[m] = n.set_index(pd.to_datetime(n["date"]))["淨值"]
    e = pd.DataFrame(EXPO).set_index("date")
    e = e[e.index >= "2023-04-01"]
    t = pd.DataFrame(acct.trades)
    t["k"] = t["strategy"].map(lambda s: next((k for k, v in P.SLEEVES.items() if s in v), "v2"))
    t["hold"] = (pd.to_datetime(t["賣出日"]) - pd.to_datetime(t["買進日"])).dt.days
    parts = []
    for k in ("v2", "s1", "disp"):
        s = t[t["k"] == k]
        parts.append(f"{NAME[k]} 平均用掉總資金 {e[k].mean():.0%}（那份的 {e[k].mean() / P.SLEEVE_W['v5'][k]:.0%}）、{len(s)} 筆、平均抱 {s['hold'].mean():.0f} 天、平均 {s['報酬'].mean():+.1%}")
    info[m] = "；".join(parts) + f"；現金平均 {e['cash'].mean():.0%}"
    print(m, "done", flush=True)
navs = pd.DataFrame(navs)
navs["加權指數"] = tx.reindex(navs.index)
seg = navs[navs.index >= "2023-04-01"]


def summ(s):
    s = s.dropna() / s.dropna().iat[0]
    return f"{s.iat[-1] - 1:+.1%}／{(s / s.cummax() - 1).min():.1%}"


half = seg.index[len(seg) // 2]
lines = [f"期間 {seg.index[0].date()} ~ {seg.index[-1].date()}（報酬／最大回撤）"]
for k in seg.columns:
    yr = "  ".join(f"{y} {summ(x[k])}" for y, x in seg.groupby(seg.index.year))
    lines.append(f"{k:5s} 全期 {summ(seg[k])}  前半 {summ(seg.loc[seg.index < half, k])}  後半 {summ(seg.loc[seg.index >= half, k])}  {yr}\n      {info.get(k, '')}")
text = "\n".join(lines)
print(text)
(ROOT / "results" / "資金分配" / "idle_cash.txt").write_text(text, encoding="utf-8")
