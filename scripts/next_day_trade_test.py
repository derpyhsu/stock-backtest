"""前一晚就能選出的當沖候選：隔天開盤買（或放空）、收盤平倉，扣當沖成本 0.435%，2022 起日線回測。

只用訊號日收盤以前知道的資料。流動性：20 日均量 >= 1000 張、股價 >= 10。
隔天開盤漲停（比昨收 +9.5% 以上）買不到、跌停放空不到，都略過。
對照：同一天所有流動性夠的股票開盤買、收盤賣（未扣成本）。
訊號：
 A 帶量長紅收最高：漲 5% 以上、收在最高價 1% 內、量 >= 20 日均量 2 倍
 B 漲停收盤：收盤漲 9.5% 以上
 C 漲停收盤＋量 >= 2 倍
 D 連 3 天漲、每天量放大
 E 長黑跌深（跌 7% 以上）→ 隔天反彈
 F 法人大買：外資＋投信當天買超 >= 20 日均量 30%
 G 突破 60 日新高＋量 2 倍
 H 法人大買＋帶量長紅（A 且 F）
 各訊號也測「隔天放空」（開盤空、收盤補）。
    python scripts/next_day_trade_test.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from twscreener.market import load_panel  # noqa: E402

COST = 0.0015 + 0.001425 * 2
p = load_panel()
piv = lambda c: p.pivot_table(index="date", columns="stock_id", values=c)
o, h, l, c, v = (piv(x) for x in ["open", "high", "low", "close", "vol"])
inst = (piv("foreign_net").fillna(0) + piv("trust_net").fillna(0)).reindex(index=c.index, columns=c.columns).fillna(0)
v20 = v.rolling(20).mean()
liquid = (v20 >= 1000) & (c >= 10)
chg = c / c.shift(1) - 1
# 拆股、減資造成的大跳動不算
chg = chg.where(chg.abs() <= 0.11)
sig = {
    "A 帶量長紅收最高": (chg >= 0.05) & (c >= h * 0.99) & (v >= 2 * v20),
    "B 漲停收盤": chg >= 0.095,
    "C 漲停＋量2倍": (chg >= 0.095) & (v >= 2 * v20),
    "D 連3漲＋量增": (chg > 0) & (chg.shift(1) > 0) & (chg.shift(2) > 0) & (v > v.shift(1)) & (v.shift(1) > v.shift(2)),
    "E 長黑跌7%": chg <= -0.07,
    "F 法人大買": inst >= 0.3 * v20,
    "G 60日新高＋量2倍": (c >= c.rolling(60).max()) & (v >= 2 * v20),
}
sig["H 法人大買＋帶量長紅"] = sig["A 帶量長紅收最高"] & sig["F 法人大買"]
no = o.shift(-1)
oc = c.shift(-1) / no - 1                    # 隔天開盤買、收盤賣（未扣成本）
gap = no / c - 1
can_long = gap < 0.095
can_short = gap > -0.095
base = oc.where(liquid).mean(axis=1)          # 同日隨機
days = c.index
half = days[len(days) // 2]
lines = [f"{days[0]:%Y-%m-%d} ~ {days[-2]:%Y-%m-%d}；成本 {COST:.3%}；隨機開盤買收盤賣每筆未扣成本 {base.mean():+.3%}", ""]
lines.append(f"{'訊號':<14}{'方向':<4}{'筆數':>7}{'勝率':>6}{'每筆扣成本':>11}{'比隨機':>9}{'前半':>9}{'後半':>9}  每年每筆扣成本")
for name, s in sig.items():
    for side, ok in (("多", can_long), ("空", can_short)):
        m = (s & liquid & ok).fillna(False)
        sgn = 1 if side == "多" else -1
        r = (sgn * oc - COST).where(m)
        x = (sgn * oc - sgn * pd.DataFrame(np.repeat(base.values[:, None], oc.shape[1], 1), index=oc.index, columns=oc.columns)).where(m)
        st = r.stack().dropna()
        if st.empty:
            continue
        dd = st.index.get_level_values(0)
        yr = st.groupby(dd.year).mean()
        lines.append(f"{name:<14}{side:<4}{len(st):>7}{(st > 0).mean():>6.0%}{st.mean():>+11.3%}{x.stack().dropna().mean():>+9.3%}"
                     f"{st[dd < half].mean():>+9.3%}{st[dd >= half].mean():>+9.3%}  "
                     + " ".join(f"{y}:{val:+.2%}" for y, val in yr.items()))
# 隔日沖：訊號日收盤買（不能當沖，成本＝證交稅 0.3%＋手續費 0.285%），隔天開盤賣或收盤賣
COST2 = 0.003 + 0.001425 * 2
lim_up = chg >= 0.095  # 收漲停：收盤排隊不一定買得到，另外標出
co = no / c - 1
cc = c.shift(-1) / c - 1
bco, bcc = co.where(liquid).mean(axis=1), cc.where(liquid).mean(axis=1)
lines += ["", f"隔日沖（訊號日收盤買）成本 {COST2:.3%}；隨機收盤買、隔天開盤賣每筆未扣成本 {bco.mean():+.3%}，隔天收盤賣 {bcc.mean():+.3%}",
          f"{'訊號':<14}{'賣點':<6}{'筆數':>7}{'勝率':>6}{'每筆扣成本':>11}{'前半':>9}{'後半':>9}  每年每筆扣成本（收漲停的另計）"]
for name, s in sig.items():
    for exit_name, rr in (("隔天開盤", co), ("隔天收盤", cc)):
        for tag, mm in (("", ~lim_up), ("漲停", lim_up)):
            m = (s & liquid & mm).fillna(False)
            st = (rr - COST2).where(m).stack().dropna()
            if len(st) < 100:
                continue
            dd = st.index.get_level_values(0)
            yr = st.groupby(dd.year).mean()
            lines.append(f"{name + ('・' + tag if tag else ''):<14}{exit_name:<6}{len(st):>7}{(st > 0).mean():>6.0%}{st.mean():>+11.3%}"
                         f"{st[dd < half].mean():>+9.3%}{st[dd >= half].mean():>+9.3%}  "
                         + " ".join(f"{y}:{val:+.2%}" for y, val in yr.items()))
text = "\n".join(lines)
print(text)
(ROOT / "results" / "當沖工具" / "next_day.txt").write_text(text, encoding="utf-8")
