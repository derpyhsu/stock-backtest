"""同樣的影片買點，只套在「老師會挑的那種股票」上會不會變好（2026-11-27）。接 scripts/youtube_nocc_rules_test.py。
老師們都先選股再看買點（熱門、主流類股的領頭股、大盤好的時候），之前是套在全部流動性股票上。這裡把股票池縮小再測：
  全部＝20 日均量 500 張以上、股價 10 元以上
  熱門＝當天成交值排全市場前 100
  大盤好＝加權指數在月線上且月線上揚的日子
  主流領頭＝近 20 日漲幅排前 5 名的產業裡，近 20 日漲幅排該產業前 3 名的股票
  已經強＝近 20 天內創過 240 日新高
  全部條件＝熱門＋大盤好＋主流領頭
每個買點列兩個數字：「比全部隨機」是減同一天全部流動性股票平均；「比同池隨機」是減同一天同一個股票池的平均。
後者才是買點本身的貢獻，前者會把股票池本身的好壞算進去。60 個交易日、扣 0.6%。
    python scripts/youtube_nocc_universe_test.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import twscreener.portfolio as P  # noqa: E402
from twscreener.market import load_market_index, load_panel  # noqa: E402
from twscreener.simulate import prices  # noqa: E402

HD, COST = 60, 0.006
p = load_panel()
raw, adj, opn, vol = prices(p)
days = raw.index


def piv(col):
    return p.pivot_table(index="date", columns="stock_id", values=col).reindex(index=days, columns=raw.columns)


ro, rh, rl = piv("open"), piv("high"), piv("low")
f = adj / raw
c, ao, ah, al = adj, ro * f, rh * f, rl * f
liquid = (vol.rolling(20).mean() >= 500) & (raw >= 10)
can_buy = ro.shift(-1) < raw * 1.095
fwd = adj.shift(-HD) / opn.shift(-1) - 1 - COST
ma = {n: c.rolling(n).mean() for n in (5, 10, 20, 60)}
tx = load_market_index()["taiex"].reindex(days).ffill()
v5 = vol.shift(1).rolling(5).mean()

# 買點（定義同 youtube_nocc_rules_test.py）
body = c / ao - 1
key = (c > ah.shift(1)) & (vol > vol.shift(1) * 1.3) & (body > 0.02)
bull = (c > ma[20]) & (ma[20] > ma[20].shift(5))
stack = (ma[5] > ma[10]) & (ma[10] > ma[20])
back = bull & (c.shift(1) <= c.shift(1).rolling(10).max() * 0.97) & (al.shift(1).rolling(5).min() > al.shift(6).rolling(15).min())
rng20 = ah.shift(1).rolling(20).max() / al.shift(1).rolling(20).min() - 1
big = (body >= 0.05) & (vol >= v5 * 2) & (c > ah.shift(1))
mid = (ao + c) / 2
sig = {
    "（不看買點，池子裡每天全買）": pd.DataFrame(True, index=days, columns=raw.columns),
    "月線上揚＋關鍵紅 K": key & bull,
    "回後買上漲＋關鍵紅 K": back & key,
    "回後買上漲＋關鍵紅 K＋三線多排": back & key & stack,
    "盤整突破（20 天 15% 內）＋關鍵紅 K": (c > ah.shift(1).rolling(20).max()) & (rng20 <= 0.15) & key & (ma[20] > ma[20].shift(5)),
    "爆量長紅隔天守住 1/2": big.shift(1).fillna(False).astype(bool) & (c >= mid.shift(1)),
}

# 股票池
val = raw * vol
hot = val.rank(axis=1, ascending=False) <= 100
txma = tx.rolling(20).mean()
mkt = pd.Series((tx > txma) & (txma > txma.shift(5)), index=days)
ind = pd.Series(P.load_industry()).reindex(raw.columns).fillna("其他")
r20 = (c / c.shift(20) - 1).where(liquid)
ind_ret = r20.T.groupby(ind.values).mean().T
top_ind = ind_ret.rank(axis=1, ascending=False) <= 5
in_top = top_ind.reindex(columns=ind.values).set_axis(raw.columns, axis=1).fillna(False).astype(bool)
rk = r20.T.groupby(ind.values).rank(ascending=False).T
lead = in_top & (rk <= 3)
strong = (c >= c.rolling(240).max()).astype(float).rolling(20).max().fillna(0).astype(bool)
mk = pd.DataFrame(np.repeat(mkt.values[:, None], raw.shape[1], axis=1), index=days, columns=raw.columns)
pools = {"全部": liquid, "熱門（成交值前 100）": liquid & hot, "大盤好的日子": liquid & mk, "主流類股領頭股": liquid & lead,
         "近 20 天創過一年新高": liquid & strong, "熱門＋大盤好＋主流領頭": liquid & hot & mk & lead}
base_all = fwd.where(liquid).mean(axis=1)
rows = []
for pn, pool in pools.items():
    base_pool = fwd.where(pool).mean(axis=1)
    for sn, s in sig.items():
        m = (s & pool & can_buy).fillna(False)
        r = fwd.where(m)
        st = r.stack().dropna()
        if len(st) < 30:
            continue
        xa = r.sub(base_all, axis=0).mean(axis=1).dropna()
        xp = r.sub(base_pool, axis=0).mean(axis=1).dropna()
        ya, yp = xa.groupby(xa.index.year).mean(), xp.groupby(xp.index.year).mean()
        rows.append({"股票池": pn, "買點": sn, "筆數": len(st), "勝率": f"{(st > 0).mean():.0%}", "比全部隨機": f"{xa.mean():+.1%}", "贏的年數(全部)": f"{(ya > 0).sum()}/{len(ya)}",
                     "比同池隨機": f"{xp.mean():+.1%}", "贏的年數(同池)": f"{(yp > 0).sum()}/{len(yp)}"})
t = pd.DataFrame(rows)
text = f"{days[0]:%Y-%m-%d} ~ {days[-1]:%Y-%m-%d}，抱 {HD} 個交易日，成本 {COST:.1%}\n" + t.to_string(index=False)
print(text)
(ROOT / "results" / "選股影片規則" / "nocc_universe.txt").write_text(text, encoding="utf-8")
