"""沒字幕影片裡兩套講得出完整規則的當沖，用 5 分 K 測（2026-11-27）。規則是 AI Studio 聽影片整理的，筆記在專案外的 strategy/。
資料：data/intraday/ 約 300 檔、60 個交易日。成本 0.535%（手續費、當沖稅、滑價）。同一根碰到停損和停利算停損。

一、朱家泓「開盤 4 法」：看第一根 5 分 K（開高或開低、收紅或收黑）＋昨天盤中趨勢；開盤漲跌超過 5% 不做；10:30 前進場；
    停損＝進場訊號那根的低點（做空是高點）、目標 2%，沒到就收盤出（另列 10:30 出）。
    昨天趨勢：昨天後半場的高低點都比前半場高＝多，都比較低＝空，其他＝盤整。
    做多：強強 K（開高收紅）昨多或盤整 → 第二根開盤買；強弱 K（開高收黑）昨多 → 等有紅 K 收盤過第一根高點；弱強 K（開低收紅）昨多 → 等收盤過第一根高點；弱弱 K 不做。
    做空（只當參考，我們不做空）：弱弱 K 昨空或盤整、強弱 K 昨空 → 第二根開盤空；其餘等黑 K 收盤破第一根低點。
二、Mr.Price 做多：5 分 K 成交量 5MA > 60MA（昨天最後一根或今天第一根）、5 分 K 均線 10 > 20 > 60、價在均價線上；
    09:10（另列 09:45）前突破第一根（爆量 5 分 K）高點就買在那個價位；停損＝第一根低點；11:30 還在就出（另列抱到收盤）。
    python scripts/youtube_nocc_intraday_test.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
COST = 0.00535
rows = []


def run_exit(o, h, l, c, hm, k, e, stop, target, side, until):
    """k＝進場那根的位置（以該根開盤或觸價 e 進場）。回傳報酬（未扣成本）。"""
    for j in range(k, len(o)):
        if side > 0:
            if l[j] <= stop:
                return min(stop, o[j]) / e - 1 if j > k else stop / e - 1
            if target and h[j] >= e * (1 + target):
                return target
        else:
            if h[j] >= stop:
                return -(max(stop, o[j]) / e - 1) if j > k else -(stop / e - 1)
            if target and l[j] <= e * (1 - target):
                return target
        if hm[j] >= until:
            return side * (c[j] / e - 1)
    return side * (c[-1] / e - 1)


for f in sorted((ROOT / "data" / "intraday").glob("*.csv")):
    d = pd.read_csv(f, parse_dates=["time"]).sort_values("time")
    d["day"], d["hm"] = d["time"].dt.normalize(), d["time"].dt.strftime("%H:%M")
    d = d[d["hm"] <= "13:30"].drop_duplicates(["day", "hm"]).reset_index(drop=True)
    d["v5"], d["v60"] = d["volume"].rolling(5).mean(), d["volume"].rolling(60).mean()
    for n in (10, 20, 60):
        d[f"m{n}"] = d["close"].rolling(n).mean()
    prev = None
    for day, g in d.groupby("day"):
        g = g.reset_index(drop=True)
        if len(g) < 40 or g["volume"].sum() <= 0:
            prev = None
            continue
        o, h, l, c, hm, v = (g[x].values for x in ("open", "high", "low", "close", "hm", "volume"))
        cur = {"close": c[-1], "h1": h[:len(g) // 2].max(), "h2": h[len(g) // 2:].max(), "l1": l[:len(g) // 2].min(), "l2": l[len(g) // 2:].min(),
               "vx": g["v5"].iat[-1] > g["v60"].iat[-1]}
        if prev is None:
            prev = cur
            continue
        pc = prev["close"]
        yt = 1 if prev["h2"] > prev["h1"] and prev["l2"] > prev["l1"] else -1 if prev["h2"] < prev["h1"] and prev["l2"] < prev["l1"] else 0
        gap = o[0] / pc - 1
        base = {"stock_id": f.stem, "day": day}
        rows.append({**base, "rule": "對照：第二根開盤買到收盤", "ret": c[-1] / o[1] - 1 - COST})
        rows.append({**base, "rule": "對照：第二根開盤空到收盤（參考）", "ret": -(c[-1] / o[1] - 1) - COST})
        # 一、開盤 4 法
        if abs(gap) <= 0.05 and o[0] < pc * 1.09:
            hi, red = gap > 0, c[0] > o[0]
            kind = ("強強" if red else "強弱") if hi else ("弱強" if red else "弱弱")
            last = int(np.searchsorted(hm, "10:30", side="right")) - 1
            ent = None  # (side, k, stop)
            if kind == "強強" and yt >= 0:
                ent = (1, 1, l[0])
            elif (kind == "強弱" and yt > 0) or (kind == "弱強" and yt > 0):
                for j in range(1, last):
                    if c[j] > h[0] and c[j] > o[j]:
                        ent = (1, j + 1, l[j])
                        break
            elif kind == "弱弱" and yt <= 0:
                ent = (-1, 1, h[0])
            elif kind == "強弱" and yt < 0:
                ent = (-1, 1, h[0])
            elif (kind == "弱強" and yt < 0) or (kind == "強強" and yt < 0):
                for j in range(1, last):
                    if c[j] < l[0] and c[j] < o[j]:
                        ent = (-1, j + 1, h[j])
                        break
            if ent and ent[1] < len(o) - 1:
                side, k, stop = ent
                e = o[k]
                if (side > 0 and e > stop) or (side < 0 and e < stop):
                    tag = f"開盤 4 法 {'做多' if side > 0 else '做空（參考）'}"
                    for nm, until in (("收盤出", "13:25"), ("10:30 出", "10:30")):
                        r = run_exit(o, h, l, c, hm, k, e, stop, 0.02, side, until) - COST
                        rows.append({**base, "rule": f"{tag}・{nm}", "ret": r})
                        rows.append({**base, "rule": f"{tag}・{kind}K・{nm}", "ret": r})
        # 二、Mr.Price 做多
        volx = prev["vx"] or (g["v5"].iat[0] > g["v60"].iat[0])
        for nm, lim in (("09:10 前突破", "09:05"), ("09:45 前突破", "09:40")):
            lastk = int(np.searchsorted(hm, lim, side="right")) - 1
            for j in range(1, lastk + 1):
                if h[j] > h[0]:
                    e = max(h[0], o[j])
                    cum = (g["close"] * g["volume"]).iloc[:j].sum() / max(g["volume"].iloc[:j].sum(), 1)
                    align = g["m10"].iat[j - 1] > g["m20"].iat[j - 1] > g["m60"].iat[j - 1]
                    if e < pc * 1.09 and e > l[0]:
                        for full, okc in (("只看突破", True), ("加量能交叉、均線多排、均價線上", volx and align and e > cum)):
                            if okc:
                                for xn, until in (("11:30 出", "11:30"), ("收盤出", "13:25")):
                                    r = run_exit(o, h, l, c, hm, j, e, l[0], None, 1, until) - COST
                                    rows.append({**base, "rule": f"Mr.Price {nm}・{full}・{xn}", "ret": r})
                    break
        prev = cur
x = pd.DataFrame(rows)
alld = sorted(x["day"].unique())
half = alld[len(alld) // 2]
lines = [f"5 分 K：{x['stock_id'].nunique()} 檔、{alld[0]:%Y-%m-%d} ~ {alld[-1]:%Y-%m-%d}（{len(alld)} 天），扣成本 {COST:.3%}"]
for r in sorted(x["rule"].unique()):
    d = x[x["rule"] == r]
    a, b = d[d["day"] < half], d[d["day"] >= half]
    lines.append(f"{r:40s} {len(d):6d} 筆（{d['day'].nunique()} 天） 勝率 {(d['ret'] > 0).mean():.0%} 平均 {d['ret'].mean():+.2%} 中位 {d['ret'].median():+.2%} 前半 {a['ret'].mean():+.2%} 後半 {b['ret'].mean():+.2%}")
text = "\n".join(lines)
print(text)
(ROOT / "results" / "當沖影片規則" / "nocc_intraday.txt").write_text(text, encoding="utf-8")
