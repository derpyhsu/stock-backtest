"""強勢股隔天開盤放空、收盤回補（當沖）做成帳戶回測，看能不能當大盤以外的現金流（2026-11-26）。

訊號（前一晚就知道）：C 漲停收盤＋量 >= 20 日均量 2 倍；A 漲 5% 以上、收在最高 1% 內、量 2 倍。
隔天開盤沒有鎖漲停（開盤 < 昨收 +9.5%）才空，收盤回補。每檔最多帳戶 10%，一天最多 10 檔（成交金額大的優先）。
成本：當沖稅 0.15% + 手續費 0.1425%×2；另外測手續費 2.8 折、每筆滑價 0.1%／0.2%。
風險情境：收盤漲停鎖死買不回來 → 隔天開盤補，另付借券等費用 1%。
    python scripts/short_fade_account_test.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from twscreener.market import load_market_index, load_panel  # noqa: E402
p = load_panel()
piv = lambda k: p.pivot_table(index="date", columns="stock_id", values=k)
o, h, c, v = piv("open"), piv("high"), piv("close"), piv("vol")
days = c.index
v20 = v.rolling(20).mean()
liquid = (v20 >= 1000) & (c >= 10)
chg = (c / c.shift(1) - 1).where(lambda x: x.abs() <= 0.11)
SIG = {"C 漲停＋量2倍": (chg >= 0.095) & (v >= 2 * v20),
       "A 帶量長紅收最高": (chg >= 0.05) & (c >= h * 0.99) & (v >= 2 * v20)}
amt = (c * v).shift(1)
gap = o / c.shift(1) - 1
day_ret = c / o - 1
locked = chg >= 0.095                      # 當天收漲停：空單買不回來
next_open = o.shift(-1)
tx = load_market_index()["taiex"].reindex(days).ffill()
half = days[len(days) // 2]


def run(sig, cost, slip, max_n=10, w=0.10, stop=None):
    m = (sig.shift(1) & liquid.shift(1) & (gap < 0.095)).fillna(False)
    rows = []
    for d in days[21:-1]:
        ids = m.columns[m.loc[d].values]
        if len(ids) == 0:
            rows.append((d, 0.0, 0))
            continue
        ids = amt.loc[d, ids].nlargest(max_n).index
        r = -day_ret.loc[d, ids] - cost - slip
        lk = locked.loc[d, ids].fillna(False)
        if stop:  # 盤中漲到開盤價 +stop 就停損回補（停損價要低於漲停價才補得到）
            sp = o.loc[d, ids] * (1 + stop)
            hit = (h.loc[d, ids] >= sp) & (sp < c.shift(1).loc[d, ids] * 1.095)
            r[hit] = -stop - cost - slip * 2
            lk = lk & ~hit
        if lk.any():
            r[lk] = -(next_open.loc[d, ids][lk] / o.loc[d, ids][lk] - 1) - cost - slip - 0.01
        r = r.dropna()
        rows.append((d, float(r.sum() * w), len(r)))
    t = pd.DataFrame(rows, columns=["d", "ret", "n"]).set_index("d")
    return t


def summ(t):
    nav = (1 + t["ret"]).cumprod()
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    mth = nav.resample("ME").last().pct_change().dropna()
    yr = nav.groupby(nav.index.year).apply(lambda s: s.iat[-1] / s.iat[0] - 1)
    return (f"年化 {nav.iat[-1] ** (1 / yrs) - 1:+.1%}／回撤 {(nav / nav.cummax() - 1).min():.1%}／賺錢月 {(mth > 0).mean():.0%}／最差月 {mth.min():+.1%}／"
            f"最差日 {t['ret'].min():+.1%}／平均每天 {t['n'].mean():.1f} 檔  " + " ".join(f"{y}:{x:+.0%}" for y, x in yr.items())), nav


lines = [f"{days[21]:%Y-%m-%d} ~ {days[-2]:%Y-%m-%d}；每檔 10%、一天最多 10 檔；收盤鎖漲停隔天開盤補＋1% 費用"]
navs = {}
FULL, DISC = 0.0015 + 0.001425 * 2, 0.0015 + 0.001425 * 2 * 0.28
for name, s in SIG.items():
    lines.append(f"\n== {name}")
    for lab, cost, slip in [("牌告手續費、無滑價", FULL, 0), ("牌告手續費、滑價 0.1%", FULL, 0.001), ("牌告手續費、滑價 0.2%", FULL, 0.002),
                            ("手續費 2.8 折、滑價 0.1%", DISC, 0.001), ("手續費 2.8 折、滑價 0.2%", DISC, 0.002), ("手續費 2.8 折、滑價 0.3%", DISC, 0.003)]:
        t = run(s, cost, slip)
        txt, nav = summ(t)
        navs[f"{name}・{lab}"] = nav
        lines.append(f"{lab:<16} {txt}")
    for st in (0.02, 0.03, 0.05):
        for lab, cost in (("牌告", FULL), ("2.8 折", DISC)):
            t = run(s, cost, 0.001, stop=st)
            txt, nav = summ(t)
            navs[f"{name}・停損{st:.0%}・{lab}"] = nav
            lines.append(f"停損 +{st:.0%}・{lab}・滑價0.1% {txt}")
    t = run(s, DISC, 0.002)
    mm = t["ret"].resample("ME").sum()
    tm = tx.resample("ME").last().pct_change()
    lines.append(f"月報酬和加權指數相關 {mm.corr(tm):.2f}；鎖漲停買不回來的比例見下")
    m = (s.shift(1) & liquid.shift(1) & (gap < 0.095)).fillna(False)
    lines.append(f"訊號 {int(m.sum().sum())} 筆，其中當天收漲停 {(locked & m).sum().sum() / m.sum().sum():.1%}，這些平均虧 {(-(next_open / o - 1)).where(locked & m).stack().mean():+.1%}")
text = "\n".join(lines)
print(text)
pd.DataFrame(navs).to_csv(ROOT / "results/大盤以外/short_fade_navs.csv")
(ROOT / "results/大盤以外/short_fade_account.txt").write_text(text, encoding="utf-8")
