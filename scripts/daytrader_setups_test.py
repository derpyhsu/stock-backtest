"""day-trader-skill（jakeschincariol，美股）的日線型態，放到台股回測。

型態（只用收盤，和原工具一樣；訊號日收盤判斷，隔天開盤買）：
- breakout_20d：收盤高於前 20 天每一天的收盤。
- bull_flag：最後 5 天之前的 10 天漲超過 10%，最後 5 天收盤都在 5% 區間內。
- reclaim_ma20：前一天收盤在 20 日均線下，今天站上。
- oversold：RSI14（Wilder）< 30。
- oversold_bounce：昨天 RSI14 < 30，今天收紅。
流動性：20 日均量 >= 500 張、股價 >= 10；隔天開盤已漲 9.5% 以上買不到。
抱 1、5、20 天收盤賣，扣 0.6% 成本，和同日隨機買股比（每天一票）。
    python scripts/daytrader_setups_test.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from twscreener.backtest import COST  # noqa: E402
from twscreener.market import load_panel  # noqa: E402
from twscreener.simulate import prices  # noqa: E402

HOLDS = (1, 5, 20)
raw, adj, opn, vol = prices(load_panel())
liquid = (vol.rolling(20).mean() >= 500) & (raw >= 10)
c = adj
d = c.diff()
up, dn = d.clip(lower=0), -d.clip(upper=0)
rsi = 100 - 100 / (1 + up.ewm(alpha=1 / 14, adjust=False).mean() / dn.ewm(alpha=1 / 14, adjust=False).mean())
ma20 = c.rolling(20).mean()
last5 = c.rolling(5)
setups = {
    "breakout_20d": c > c.shift(1).rolling(20).max(),
    "bull_flag": (c.shift(5) / c.shift(15) - 1 > 0.10) & (last5.max() / last5.min() - 1 <= 0.05),
    "reclaim_ma20": (c.shift(1) < ma20.shift(1)) & (c > ma20),
    "oversold": rsi < 30,
    "oversold_bounce": (rsi.shift(1) < 30) & (c > c.shift(1)),
}
entry = opn.shift(-1)
can_buy = (opn.shift(-1) < c * 1.095)
fwd = {h: c.shift(-h) / entry - 1 - COST for h in HOLDS}
base = {h: fwd[h].where(liquid).mean(axis=1) for h in HOLDS}
lines = []
for name, sig in setups.items():
    s = (sig & liquid & can_buy).fillna(False)
    rows = {}
    for h in HOLDS:
        r = fwd[h].where(s)
        x = r.sub(base[h], axis=0)
        rows[h] = (r, x)
    r5, x5 = rows[5]
    n = int(r5.notna().sum().sum())
    out = [f"\n== {name}（{n} 筆）", "期間  " + "  ".join(f"{h}日勝率 {h}日平均 {h}日每天一票超額" for h in HOLDS)]
    years = sorted(set(c.index.year))
    for label, mask in [("全期", slice(None))] + [(str(y), c.index.year == y) for y in years]:
        cells = []
        for h in HOLDS:
            r, x = rows[h]
            rr, xx = r[mask], x[mask]
            v = rr.stack().dropna()
            cells.append(f"{(v > 0).mean():.0%} {v.mean():+.2%} {xx.mean(axis=1).dropna().mean():+.2%}")
        out.append(f"{label}  " + "  ".join(cells))
    lines += out
text = "\n".join(lines)
print(text)
OUT = ROOT / "results" / "當沖工具"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "daily_setups.txt").write_text(text, encoding="utf-8")
