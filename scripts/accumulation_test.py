"""吸籌階段回測：股價橫盤、量縮，法人卻一直買、融資減少（籌碼從散戶流到大戶），之後會不會漲？

吸籌（訊號日收盤，只用當天以前的資料）：
- 橫盤：近 60 日收盤最高 ÷ 最低 - 1 <= RANGE。
- 量縮：20 日均量 <= 120 日均量 × VOLR。
- 法人買：外資＋投信 20 日買超合計 >= 20 日均量 × INST（等於 INST 天的平均成交量）。
- 融資減：融資餘額比 20 天前少。
- 流動性：20 日均量 >= 500 張、股價 >= 10。
同一檔 20 個交易日內只算第一次。
買點：
- 吸籌中：訊號隔天開盤買。
- 突破：吸籌訊號後 20 天內，第一次收盤創 60 日新高且量 >= 2 倍 20 日均量，隔天開盤買。
抱 20、60 天，扣 0.6% 成本，和同日隨機買股比；也試拿掉個別條件，看哪個條件有用。
    python scripts/accumulation_test.py
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

RANGE, VOLR, INST = 0.20, 0.8, 1.0
HOLDS = (20, 60)
OUT = ROOT / "results" / "吸籌"
OUT.mkdir(parents=True, exist_ok=True)

panel = load_panel()
raw, adj, opn, vol = prices(panel)
piv = lambda col: panel.pivot_table(index="date", columns="stock_id", values=col).reindex(index=adj.index, columns=adj.columns)
inst = (piv("foreign_net").fillna(0) + piv("trust_net").fillna(0))
margin = piv("margin_bal")
vol20, vol120 = vol.rolling(20).mean(), vol.rolling(120).mean()
liquid = (vol20 >= 500) & (raw >= 10)
hi60, lo60 = adj.rolling(60).max(), adj.rolling(60).min()
cond = {
    "橫盤": (hi60 / lo60 - 1) <= RANGE,
    "量縮": vol20 <= vol120 * VOLR,
    "法人買": inst.rolling(20).sum() >= vol20 * INST,
    "融資減": margin < margin.shift(20),
}
entry = opn.shift(-1)
fwd = {h: adj.shift(-h) / entry - 1 - COST for h in HOLDS}
base = {h: fwd[h].where(liquid).mean(axis=1) for h in HOLDS}
breakout = (adj >= hi60) & (vol >= 2 * vol20)
can_buy = opn.shift(-1) < adj * 1.095  # 隔天開盤沒有漲停附近


def first_in(sig, gap=20):
    """同一檔 gap 天內只留第一次。"""
    prev = sig.shift(1).rolling(gap, min_periods=1).max().fillna(0).astype(bool)
    return sig & ~prev


def collect(sig, name):
    rows = []
    s = sig.stack()
    s = s[s]
    for (d, sid) in s.index:
        i = adj.index.get_loc(d)
        if not can_buy.iat[i, adj.columns.get_loc(sid)]:
            continue
        row = {"date": d, "stock_id": sid, "variant": name}
        for h in HOLDS:
            row[f"r{h}"] = fwd[h].at[d, sid]
            row[f"x{h}"] = row[f"r{h}"] - base[h].at[d]
        rows.append(row)
    return pd.DataFrame(rows).dropna(subset=[f"r{h}" for h in HOLDS])


full = liquid & cond["橫盤"] & cond["量縮"] & cond["法人買"] & cond["融資減"]
variants = {"吸籌（全部條件）": full}
for drop in cond:
    variants[f"拿掉{drop}"] = liquid & np.logical_and.reduce([cond[k] for k in cond if k != drop])
variants["只有法人買＋融資減（不看股價）"] = liquid & cond["法人買"] & cond["融資減"]
parts = []
for name, sig in variants.items():
    parts.append(collect(first_in(sig.fillna(False)), name))
    print(name, len(parts[-1]), flush=True)

# 吸籌後 20 天內第一次帶量突破
acc = first_in(full.fillna(False))
recent = acc.shift(1).rolling(20, min_periods=1).max().fillna(0).astype(bool)
brk = first_in((breakout & recent & liquid).fillna(False))
parts.append(collect(brk, "吸籌後帶量突破"))
parts.append(collect(first_in((breakout & liquid).fillna(False)), "對照：任何帶量突破"))

t = pd.concat(parts)
t["year"] = t["date"].dt.year
t["half"] = np.where(t["date"] <= t["date"].quantile(0.5), "前段", "後段")


def stat(g):
    out = {"件數": len(g)}
    for h in HOLDS:
        out[f"{h}日勝率"] = f"{(g[f'r{h}'] > 0).mean():.0%}"
        out[f"{h}日中位數"] = f"{g[f'r{h}'].median():+.1%}"
        out[f"{h}日平均"] = f"{g[f'r{h}'].mean():+.1%}"
        out[f"{h}日每月一票超額"] = f"{g.groupby(g['date'].dt.to_period('M'))[f'x{h}'].mean().mean():+.1%}"
    out["60日超額>0的年份"] = ""
    return pd.Series(out)


pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 20)
lines = []
for name in list(variants) + ["吸籌後帶量突破", "對照：任何帶量突破"]:
    g = t[t["variant"] == name]
    if g.empty:
        lines.append(f"\n== {name}：沒有訊號")
        continue
    tab = pd.concat([pd.concat({"全期": stat(g)}, axis=1).T, g.groupby("year").apply(stat), g.groupby("half").apply(stat)])
    tab = tab.drop(columns="60日超額>0的年份")
    lines.append(f"\n== {name}（{g['date'].min():%Y-%m-%d} ~ {g['date'].max():%Y-%m-%d}）\n" + tab.to_string())
text = "\n".join(lines)
print(text)
(OUT / "accumulation.txt").write_text(text, encoding="utf-8")
t.to_csv(OUT / "events.csv", index=False, encoding="utf-8-sig")
