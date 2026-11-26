"""當沖選股條件，用日線能驗的部分（2026-11-26）。

網路文章常見的當沖選股條件：量大（3000 張以上）、振幅大、昨天漲停或強勢、開盤跳空、突破昨天高點。
我們只有日線（開高低收），所以只能驗「開盤進、收盤出」這種整天的結果，盤中怎麼走（第一次拉回、
5 分 K 突破）驗不了。三種做法：
  開盤買收盤賣：當天開盤價買、收盤價賣
  同上＋停損 2%：盤中最低碰到開盤價 -2% 就當作 -2% 出場（實際會滑價，這是樂觀的）
  過昨高才買：開盤在昨高以下、盤中碰到昨高，用昨高買、收盤賣（實際追價會更貴，這是樂觀的）
「先賣後買」是同一筆反過來做（開盤賣、收盤買回），只是參考；不是每檔都能先賣，平盤下也有限制。
成本 0.4%（手續費 0.1425% 兩次打約 9 折＋當沖證交稅 0.15%）。每天一票：先算當天訊號平均，再對天數平均。
價格都用原始價；昨收、昨高用還原因子調到今天，避免除權息日被當成跳空開低。只用 2022 年以後。
    python scripts/daytrade_daily_proxy_test.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from twscreener.market import load_panel  # noqa: E402
from twscreener.simulate import prices  # noqa: E402

COST, STOP = 0.004, 0.02
p = load_panel()
p = p[p["date"] >= "2022-01-01"]
raw, adj, _, vol = prices(p)


def piv(c):
    return p.pivot_table(index="date", columns="stock_id", values=c).reindex(index=raw.index, columns=raw.columns)


O, Hi, Lo, C = piv("open"), piv("high"), piv("low"), raw
f = raw / adj                                   # 今天的還原因子
k = f / f.shift(1)                              # 昨天的價格換成今天的基準
pc, ph, pl, po = C.shift(1) * k, Hi.shift(1) * k, Lo.shift(1) * k, O.shift(1) * k
ppc = C.shift(2) * k * k.shift(1)
ok = (vol.rolling(20).mean().shift(1) >= 3000) & (pc >= 10) & O.notna() & (O > 0)
gap = O / pc - 1
buyable = gap < 0.095                           # 開盤就漲停買不到
ychg = pc / ppc - 1                             # 昨天漲跌
yamp = (ph - pl) / ppc                          # 昨天振幅
amp3 = ((Hi - Lo) / C.shift(1)).rolling(3).mean().shift(1)
vr = (vol / vol.rolling(20).mean().shift(1)).shift(1)   # 昨天量 / 之前 20 日均量
hi20 = (C >= C.rolling(20).max()).shift(1).fillna(False).astype(bool)
yred = pc > po

oc = C / O - 1
ret = {
    "開盤買收盤賣": oc - COST,
    "＋停損 2%": oc.where(Lo > O * (1 - STOP), -STOP) - COST,
    "過昨高才買": (C / ph - 1 - COST).where((O < ph) & (Hi > ph)),
    "先賣後買": -oc - COST,
}

lim = ychg >= 0.095
sig = {
    "全部（20 日均量 3000 張以上）": ok,
    "昨天漲停": lim,
    "　昨天漲停＋今天開高不到 2%": lim & (gap >= 0) & (gap < 0.02),
    "　昨天漲停＋今天開高 2~5%": lim & (gap >= 0.02) & (gap < 0.05),
    "　昨天漲停＋今天開高 5% 以上": lim & (gap >= 0.05),
    "　昨天漲停＋今天開低": lim & (gap < 0),
    "昨天漲 5%~9.5%": (ychg >= 0.05) & (ychg < 0.095),
    "昨天振幅 7% 以上且收紅": (yamp >= 0.07) & yred,
    "近 3 天平均振幅 6% 以上": amp3 >= 0.06,
    "昨天量是均量 2 倍以上且收紅": (vr >= 2) & yred,
    "昨天創 20 日新高": hi20,
    "昨天跌 5% 以上": ychg <= -0.05,
    "昨天跌停": ychg <= -0.095,
    "今天開高 2~5%": (gap >= 0.02) & (gap < 0.05),
    "今天開高 5% 以上": gap >= 0.05,
    "今天開低 2~5%": (gap <= -0.02) & (gap > -0.05),
    "今天開低 5% 以上": gap <= -0.05,
}
rows = []
for name, s in sig.items():
    m = (s & ok & buyable).fillna(False)
    row = {"條件": name, "筆數": int(m.sum().sum()), "有訊號的天數": int(m.any(axis=1).sum())}
    for rn, r in ret.items():
        x = r.where(m)
        st = x.stack().dropna()
        if len(st) < 30:
            continue
        d = x.mean(axis=1).dropna()
        yr = d.groupby(d.index.year).mean()
        row[f"{rn}_平均"] = d.mean()
        row[f"{rn}_勝率"] = (st > 0).mean()
        row[f"{rn}_賺的年數"] = f"{(yr > 0).sum()}/{len(yr)}"
        row[f"{rn}_每年"] = " ".join(f"{y % 100}:{v:+.2%}" for y, v in yr.items())
    rows.append(row)
t = pd.DataFrame(rows)
out = ROOT / "results" / "當沖"
out.mkdir(parents=True, exist_ok=True)
t.to_csv(out / "daytrade_daily_proxy.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250, "display.unicode.east_asian_width", True, "display.max_columns", 30)
print(f"{raw.index[0].date()} ~ {raw.index[-1].date()}；成本 {COST:.1%}；每天一票（扣成本後每筆平均）")
for rn in ret:
    cols = [c for c in (f"{rn}_平均", f"{rn}_勝率", f"{rn}_賺的年數") if c in t]
    v = t[["條件", "筆數"] + cols].copy()
    for c in cols[:2]:
        v[c] = v[c].map(lambda z: "" if pd.isna(z) else (f"{z:+.2%}" if c.endswith("平均") else f"{z:.0%}"))
    print(f"\n== {rn}")
    print(v.to_string(index=False))
print("\n== 開盤買收盤賣逐年")
for _, r in t.iterrows():
    print(f"{r['條件']}：{r.get('開盤買收盤賣_每年', '')}")
