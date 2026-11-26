"""v2 模擬帳戶用台指期（以加權指數近似）放空避險，看扣掉大盤後剩多少純超額（2026-11-26）。

每天收盤後，依持股市值 × beta 放空等值期貨（beta：持股過去 60 天對加權日報酬的迴歸，或固定 1）。
期貨正常是逆價差（除息造成），空方每年約損失大盤殖利率，用 3.5%／年扣；手續費、保證金利息忽略。
保證金需求約名目 15%，用帳戶現金支應，不足時照算（只看報酬，不模擬追繳）。
    python scripts/hedged_v2_test.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import twscreener.portfolio as P  # noqa: E402
from twscreener.market import load_market_index  # noqa: E402
DIV = 0.035
inputs = P._inputs()
days = inputs[0].index
tx = load_market_index()["taiex"].reindex(days).ffill()
a = P.run(days[days >= "2022-08-01"], P.Account(), inputs, version="v2")
n = pd.DataFrame(a.nav)
n.index = pd.to_datetime(n["date"])
nav, cash = n["淨值"], n["現金"]
inv = ((nav - cash) / nav).clip(0, 1)
r = nav.pct_change().fillna(0)
rm = tx.reindex(nav.index).pct_change().fillna(0) + DIV / 252
beta = (r.rolling(60).cov(rm) / rm.rolling(60).var() / inv.shift(1).replace(0, np.nan)).clip(0.3, 2).fillna(1.0)
out = {"v2（不避險）": nav / nav.iat[0], "加權指數": tx.reindex(nav.index) / tx.reindex(nav.index).iat[0]}
for lab, b in [("v2＋避險 beta=1", 1.0), ("v2＋避險 滾動beta", beta.shift(1))]:
    h = inv.shift(1).fillna(0) * b
    out[lab] = (1 + r - h * rm).cumprod()
navs = pd.DataFrame(out)
navs.to_csv(ROOT / "results/大盤以外/hedged_v2_navs.csv")
def summ(s):
    s = s.dropna() / s.dropna().iat[0]
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    m = s.resample("ME").last().pct_change().dropna()
    return f"{s.iat[-1] - 1:+.1%}（年化 {s.iat[-1] ** (1 / yrs) - 1:+.1%}）／回撤 {(s / s.cummax() - 1).min():.1%}／賺錢月 {(m > 0).mean():.0%}"
lines = [f"平均持股比例 {inv.mean():.0%}，滾動 beta 中位數 {beta.median():.2f}；避險成本 {DIV:.1%}／年"]
for st in ("2022-08-01", "2023-04-01"):
    seg = navs[navs.index >= st]
    lines.append(f"\n期間 {seg.index[0].date()} ~ {seg.index[-1].date()}")
    for k in seg.columns:
        yr = "  ".join(f"{y}:{g[k].iat[-1] / g[k].iat[0] - 1:+.1%}" for y, g in seg.groupby(seg.index.year))
        lines.append(f"{k:16s} {summ(seg[k])}  {yr}")
corr = navs.pct_change().resample("ME").sum().corr()
lines.append(f"\n月報酬和加權指數的相關：不避險 {corr.loc['v2（不避險）', '加權指數']:.2f}，避險 beta=1 {corr.loc['v2＋避險 beta=1', '加權指數']:.2f}，滾動 {corr.loc['v2＋避險 滾動beta', '加權指數']:.2f}")
text = "\n".join(lines)
print(text)
(ROOT / "results/大盤以外/hedged_v2.txt").write_text(text, encoding="utf-8")
