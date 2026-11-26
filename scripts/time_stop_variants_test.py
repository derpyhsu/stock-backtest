"""模擬帳戶 v4（含續抱）加「盤著不動就換股」的時間停損，和現在的規則比（2026-11-26）。
只動營收加速、一年新高兩份；處置股不動。買進後第 N 個交易日收盤，報酬（對買進價，未扣成本）不到門檻就隔天開盤賣。
  A 現在的規則
  N=20/30/40 × 門檻 0%／+5%
另外 S＝只對一年新高那份套用（N=30、門檻 0%）。
    python scripts/time_stop_variants_test.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import twscreener.portfolio as P  # noqa: E402
from twscreener.market import load_market_index  # noqa: E402

inputs = P._inputs()
raw = inputs[0]
days = raw.index
MODE = {"n": None, "th": 0.0, "only": None}
orig_plan = P.Account.plan


def plan(self, day, close, cands, prev=None, trend=None):
    n = MODE["n"]
    if n and len(self.pos):
        for idx, h in self.pos.iterrows():
            if pd.notna(h["exit_date"]) or (MODE["only"] and h["strategy"] != MODE["only"]):
                continue
            c = close.get(h["stock_id"], np.nan)
            if h["days"] + 1 == n and np.isfinite(c) and c / h["entry"] - 1 < MODE["th"]:
                self.pos.at[idx, "stop"] = 1e9   # 照停損流程隔天開盤賣
    return orig_plan(self, day, close, cands, prev, trend)


P.Account.plan = plan
taiex = load_market_index()["taiex"].reindex(days).ffill()
cases = [("A 現在", None, 0, None)] + [(f"{n}天<{int(th * 100)}%", n, th, None) for n in (20, 30, 40) for th in (0.0, 0.05)]
cases.append(("S 只一年新高 30天<0%", 30, 0.0, "一年新高"))
navs, info = {}, {}
for name, n, th, only in cases:
    MODE.update(n=n, th=th, only=only)
    a = P.run(days[days >= "2023-01-01"], P.Account(), inputs, version="v4")
    nv = pd.DataFrame(a.nav)
    navs[name] = nv.set_index(pd.to_datetime(nv["date"]))["淨值"]
    t = pd.DataFrame(a.trades)
    s = t[t["strategy"] != "處置股"]
    info[name] = f"非處置股 {len(s)} 筆、勝率 {(s['報酬'] > 0).mean():.0%}、平均 {s['報酬'].mean():+.1%}"
navs = pd.DataFrame(navs)
navs["加權指數"] = taiex.reindex(navs.index)
seg = navs[navs.index >= "2023-04-01"]


def summ(s):
    s = s.dropna() / s.dropna().iat[0]
    return f"{s.iat[-1] - 1:+.1%}／{(s / s.cummax() - 1).min():.1%}"


lines = [f"期間 {seg.index[0].date()} ~ {seg.index[-1].date()}（報酬／最大回撤）"]
for k in seg.columns:
    yr = "  ".join(f"{y} {summ(g[k])}" for y, g in seg.groupby(seg.index.year))
    lines.append(f"{k}  全期 {summ(seg[k])}  {yr}  {info.get(k, '')}")
text = "\n".join(lines)
print(text)
(ROOT / "results" / "手冊未測型態" / "time_stop_variants.txt").write_text(text, encoding="utf-8")
