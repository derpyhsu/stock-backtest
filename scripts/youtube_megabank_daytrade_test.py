"""兆豐證券「短線當沖升級法」和權證小哥「放空攻略」裡做多的那套，用 5 分 K 測（2026-11-27）。
影片沒有字幕，規則是請 NotebookLM 讀轉錄稿整理的。原版用每 5 秒的連次量找進場點，我們只有 5 分 K，改用「那根 5 分 K 收紅」代替。
  做多：盤中預估量 ÷ 20 日均量 >= 5 倍、前一天漲跌 3% 以內（小紅小黑）、現價創 20 日收盤新高、當天已漲 3% 以上、那根收紅 → 下一根開盤買
  出場：停損 -3%、停利 +6%，都沒到就 13:25 收盤賣（另列不設停損停利）
  搶反彈：跌破 20 日線 15% 以上、當天再跌 5% 以上、預估量 >= 3 倍 → 下一根開盤買，收盤賣
預估量＝累計量 ÷ 全市場平均在那個時間點已成交的比例。成本 0.535%。對照：同一時間點隨便一檔買到收盤。
    python scripts/youtube_megabank_daytrade_test.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
import sys  # noqa: E402
sys.path.insert(0, str(ROOT))
from twscreener.market import load_panel  # noqa: E402

COST = 0.00535
p = load_panel()
p = p[p["date"] >= "2026-06-01"]
cl = p.pivot_table(index="date", columns="stock_id", values="close")
vo = p.pivot_table(index="date", columns="stock_id", values="vol")
prev, prev2 = cl.shift(1), cl.shift(2)
hi20 = cl.shift(1).rolling(20).max()
ma20 = cl.shift(1).rolling(20).mean()
v20 = vo.shift(1).rolling(20).mean()

days = {}
for f in sorted((ROOT / "data" / "intraday").glob("*.csv")):
    d = pd.read_csv(f, parse_dates=["time"])
    d["day"], d["hm"] = d["time"].dt.normalize(), d["time"].dt.strftime("%H:%M")
    d = d[d["hm"] <= "13:30"].drop_duplicates(["day", "hm"])
    for day, g in d.groupby("day"):
        if len(g) >= 40 and g["volume"].sum() > 0:
            days[(f.stem, day)] = g.reset_index(drop=True)
# 全市場平均：各時間點累計量佔全天的比例
frac = pd.concat([(g.set_index("hm")["volume"].cumsum() / g["volume"].sum()) for g in list(days.values())[::7]], axis=1).mean(axis=1)
unit = {}
rows = []
for (sid, day), g in days.items():
    if sid not in cl.columns or day not in cl.index:
        continue
    pc, pc2, h20, m20, av = prev.at[day, sid], prev2.at[day, sid], hi20.at[day, sid], ma20.at[day, sid], v20.at[day, sid]
    dv = vo.at[day, sid]
    if not np.isfinite([pc, pc2, h20, m20, av, dv]).all() or av <= 0 or dv <= 0:
        continue
    k = g["volume"].sum() / dv  # 5 分 K 的量和日線的量單位換算
    cum = g["volume"].cumsum().values / k
    o, h, l, c, hm = g["open"].values, g["high"].values, g["low"].values, g["close"].values, g["hm"].values
    est = cum / frac.reindex(hm).values
    ratio = est / av
    done = set()
    for i in range(1, len(g) - 2):
        if hm[i] > "11:00":
            break
        e = o[i + 1]
        if e >= pc * 1.095:
            continue
        rest_h, rest_l, last = h[i + 1:], l[i + 1:], c[-1]
        base = {"stock_id": sid, "day": day, "hm": hm[i], "close_ret": last / e - 1 - COST}
        if "rand" not in done and hm[i] == "09:30":
            rows.append({**base, "rule": "對照：09:30 隨便買到收盤"})
        for name, thr in (("做多 均量比>=5", 5), ("做多 均量比>=3", 3)):
            if name in done:
                continue
            if ratio[i] >= thr and abs(pc / pc2 - 1) <= 0.03 and c[i] > h20 and c[i] >= pc * 1.03 and c[i] > o[i]:
                done.add(name)
                ret = last / e - 1
                for j in range(len(rest_h)):
                    if rest_l[j] <= e * 0.97:
                        ret = -0.03
                        break
                    if rest_h[j] >= e * 1.06:
                        ret = 0.06
                        break
                rows.append({**base, "rule": name, "ret": ret - COST})
        if "reb" not in done and ratio[i] >= 3 and pc <= m20 * 0.85 and c[i] <= pc * 0.95:
            done.add("reb")
            rows.append({**base, "rule": "搶反彈", "ret": last / e - 1 - COST})
x = pd.DataFrame(rows)
x["ret"] = x["ret"].fillna(x["close_ret"])
alld = sorted(x["day"].unique())
half = alld[len(alld) // 2]
lines = [f"5 分 K：{x['stock_id'].nunique()} 檔、{alld[0]:%Y-%m-%d} ~ {alld[-1]:%Y-%m-%d}（{len(alld)} 天），扣成本 {COST:.3%}"]
for r, d in x.groupby("rule"):
    a, b = d[d["day"] < half], d[d["day"] >= half]
    lines.append(f"{r:16s} {len(d):5d} 筆（{d['day'].nunique()} 天） 勝率 {(d['ret'] > 0).mean():.0%} 平均 {d['ret'].mean():+.2%} 中位 {d['ret'].median():+.2%} "
                 f"前半 {a['ret'].mean():+.2%} 後半 {b['ret'].mean():+.2%}｜不設停損停利抱到收盤 {d['close_ret'].mean():+.2%}")
text = "\n".join(lines)
print(text)
(ROOT / "results" / "當沖影片規則" / "megabank_rules.txt").write_text(text, encoding="utf-8")
