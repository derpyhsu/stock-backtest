"""「錢滾錢」：v5 帳戶每筆投入比例放大或分段調整（2026-11-27）。
帳戶本來就用「當下淨值」算每筆金額（賺了之後每筆自動變大），但回測期間平均有 39% 是現金。
營收加速、一年新高每檔的大小＝打到停損（-15%）虧那一份淨值的 RISK；RISK 1% → 每檔佔那一份 6.7%，10 檔滿也只用到 67%。
  A 現在的 v5（RISK 1%、處置股每筆 5%）
  B RISK 1.25%          C RISK 1.5%（10 檔剛好用滿那一份）          D RISK 2%（約 7 檔就滿）
  E 處置股每筆 7.5%      F C＋E
  G 分段：淨值到 120 萬後 RISK 1.25%，到 150 萬後 1.5%（只升不降）
  H 回撤時縮小：平常 RISK 1.5%，帳戶從高點回落 10% 以上時 0.75%
從 2023-04-06 全部現金開始。
    python scripts/sizing_variants_test.py
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
MODE = {"m": "A", "peak": 0.0}
orig_plan = P.Account.plan


def plan(self, day, close, cands, prev=None, trend=None):
    m = MODE["m"]
    if m in "GH":
        nav = self.value(close)
        MODE["peak"] = max(MODE["peak"], nav)
        if m == "G":
            P.RISK = 0.015 if MODE["peak"] >= 1.5e6 else 0.0125 if MODE["peak"] >= 1.2e6 else 0.01
        else:
            P.RISK = 0.0075 if nav / MODE["peak"] - 1 <= -0.10 else 0.015
    return orig_plan(self, day, close, cands, prev, trend=trend)


P.Account.plan = plan
CFG = {"A": (0.01, 0.05), "B": (0.0125, 0.05), "C": (0.015, 0.05), "D": (0.02, 0.05), "E": (0.01, 0.075), "F": (0.015, 0.075), "G": (0.01, 0.05), "H": (0.015, 0.05)}
taiex = load_market_index()["taiex"].reindex(days).ffill()
navs, info = {}, {}
for m, (risk, dw) in CFG.items():
    MODE.update(m=m, peak=0.0)
    P.RISK, P.DISP_W = risk, dw
    a = P.run(days[days >= "2023-04-06"], P.Account(), inputs, version="v5")
    n = pd.DataFrame(a.nav).set_index(pd.to_datetime(pd.DataFrame(a.nav)["date"]))
    navs[m] = n["淨值"]
    mth = n["淨值"].resample("ME").last().pct_change().dropna()
    info[m] = f"平均現金 {(n['現金'] / n['淨值']).mean():.0%}・最差月 {mth.min():+.1%}・交易 {len(a.trades)} 筆"
P.RISK, P.DISP_W = 0.01, 0.05
navs = pd.DataFrame(navs)
navs["加權指數"] = taiex.reindex(navs.index) / taiex.reindex(navs.index).iat[0] * 1e6


def summ(s, base=None):
    s = s.dropna()
    b = base if base is not None else s.iat[0]
    return f"{s.iat[-1] / b - 1:+.1%}／{(s / s.cummax() - 1).min():.1%}"


half = navs.index[len(navs) // 2]
lines = [f"期間 {navs.index[0].date()} ~ {navs.index[-1].date()}，從 100 萬現金開始（報酬／最大回撤）"]
for k in navs.columns:
    s = navs[k]
    prev, yr = 1e6, []
    for y, x in s.groupby(s.index.year):
        yr.append(f"{y} {x.iat[-1] / prev - 1:+.1%}")
        prev = x.iat[-1]
    lines.append(f"{k:5s} 全期 {summ(s, 1e6)}  前半 {s[s.index < half].iat[-1] / 1e6 - 1:+.1%}  後半 {s.iat[-1] / s[s.index < half].iat[-1] - 1:+.1%}  {'  '.join(yr)}  {info.get(k, '')}")
text = "\n".join(lines)
print(text)
out = ROOT / "results" / "規則依據"
(out / "sizing_variants.txt").write_text(text, encoding="utf-8")
