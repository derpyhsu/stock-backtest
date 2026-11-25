"""隔日沖「收漲停買、隔天開盤賣」的成交檢查：收盤鎖漲停時買單要排隊，真的買得到嗎？

1. 日線（2022 起）：接近漲停但沒鎖（收盤漲 8%～9.5%、收在最高 1% 內）—— 一定買得到的版本。
2. 5 分 K（近 58 天）：收漲停的股票分兩種
   - 尾盤打開過：12:55 之後有 5 分 K 最低價低於漲停價（有人賣，排隊的買單有機會成交）
   - 一路鎖死：尾盤 5 分 K 最低價都等於漲停價（排隊買單多半買不到）
   比較兩種隔天開盤賣的報酬。成本 0.585%（非當沖）。
    python scripts/limit_up_fill_test.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from twscreener.market import load_panel  # noqa: E402

COST2 = 0.003 + 0.001425 * 2
p = load_panel()
piv = lambda c: p.pivot_table(index="date", columns="stock_id", values=c)
o, h, c, v = (piv(x) for x in ["open", "high", "close", "vol"])
liquid = (v.rolling(20).mean() >= 1000) & (c >= 10)
chg = c / c.shift(1) - 1
co = o.shift(-1) / c - 1 - COST2
near = (chg >= 0.08) & (chg < 0.095) & (c >= h * 0.99) & liquid
lines = []
for name, m in [("接近漲停沒鎖（8%～9.5%，買得到）", near), ("收漲停（假設都買到）", (chg >= 0.095) & liquid)]:
    st = co.where(m).stack().dropna()
    yr = st.groupby(st.index.get_level_values(0).year).mean()
    lines.append(f"{name}：{len(st)} 筆，勝率 {(st > 0).mean():.0%}，每筆 {st.mean():+.2%}　" + " ".join(f"{y}:{x:+.2%}" for y, x in yr.items()))


def tick(x):
    return 0.01 if x < 10 else 0.05 if x < 50 else 0.1 if x < 100 else 0.5 if x < 500 else 1 if x < 1000 else 5


rows = []
lu = ((chg >= 0.095) & liquid).stack()
lu = lu[lu]
for f in sorted((ROOT / "data" / "intraday").glob("*.csv")):
    sid = f.stem
    days = [d for d, s in lu.index if s == sid]
    if not days:
        continue
    b = pd.read_csv(f, parse_dates=["time"])
    b["d"] = b["time"].dt.normalize()
    for d in days:
        g = b[(b["d"] == d) & (b["time"].dt.strftime("%H:%M") >= "12:55")]
        if g.empty or pd.isna(co.at[d, sid]):
            continue
        limit = c.at[d, sid]
        opened = (g["low"] < limit - tick(limit) / 2).any()
        rows.append({"date": d, "stock_id": sid, "尾盤打開過": opened, "ret": co.at[d, sid]})
t = pd.DataFrame(rows)
lines.append(f"\n5 分 K 期間收漲停 {len(t)} 筆（{t['date'].min():%Y-%m-%d} ~ {t['date'].max():%Y-%m-%d}）")
for k, g in t.groupby("尾盤打開過"):
    lines.append(f"{'尾盤打開過（較可能買到）' if k else '一路鎖死（多半買不到）'}：{len(g)} 筆，勝率 {(g['ret'] > 0).mean():.0%}，隔天開盤賣每筆 {g['ret'].mean():+.2%}，中位數 {g['ret'].median():+.2%}")
text = "\n".join(lines)
print(text)
(ROOT / "results" / "當沖工具" / "limit_up_fill.txt").write_text(text, encoding="utf-8")
