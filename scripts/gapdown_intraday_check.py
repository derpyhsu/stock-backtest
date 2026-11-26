"""開盤跳空大跌買（web_strategies D2）用 5 分 K 檢查買點：開盤價、9:00 那根收盤、9:05 收盤買，13:20 收盤或收盤價賣。
    python scripts/gapdown_intraday_check.py
"""
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
COST = 0.0015 + 0.001425 * 2
files = sorted((ROOT / "data" / "daily").glob("*.csv"))[-100:]
d = pd.concat(pd.read_csv(f, dtype={"stock_id": str}, usecols=["date", "stock_id", "open", "close"]) for f in files)
d["date"] = pd.to_datetime(d["date"])
d = d.sort_values(["stock_id", "date"])
d["pc"] = d.groupby("stock_id")["close"].shift(1)
d["gap"] = d["open"] / d["pc"] - 1
ev = d[(d["gap"] <= -0.04) & (d["gap"] > -0.095)].set_index(["stock_id", "date"])
rows = []
for f in sorted((ROOT / "data" / "intraday").glob("*.csv")):
    b = pd.read_csv(f, parse_dates=["time"])
    b["day"] = b["time"].dt.normalize()
    b["hm"] = b["time"].dt.strftime("%H:%M")
    for day, g in b.groupby("day"):
        if (f.stem, day) not in ev.index:
            continue
        g = g.set_index("hm")
        if "09:00" not in g.index or "09:05" not in g.index or "13:20" not in g.index:
            continue
        e = ev.loc[(f.stem, day)]
        rows.append({"sid": f.stem, "day": day, "gap": e["gap"], "daily_open_close": e["close"] / e["open"] - 1,
                     "bar_open": e["close"] / g.at["09:00", "open"] - 1, "c0900": e["close"] / g.at["09:00", "close"] - 1,
                     "c0905": e["close"] / g.at["09:05", "close"] - 1, "c0905_1320": g.at["13:20", "close"] / g.at["09:05", "close"] - 1})
t = pd.DataFrame(rows)
lines = [f"5 分 K 涵蓋期間 {t['day'].min():%Y-%m-%d} ~ {t['day'].max():%Y-%m-%d}，跳空 <= -4% 的股票日 {len(t)} 筆，扣成本 {COST:.3%}",
         "買點 → 收盤賣：每筆平均（扣成本）／勝率"]
for c, lab in [("daily_open_close", "日線開盤價"), ("bar_open", "5分K 9:00 開盤"), ("c0900", "9:00 那根收盤"),
               ("c0905", "9:05 那根收盤"), ("c0905_1320", "9:05 收盤買、13:20 收盤賣")]:
    r = t[c] - COST
    lines.append(f"{lab:<20}{r.mean():+.2%}／{(r > 0).mean():.0%}")
lines.append(f"跳空 <= -6%：{len(t[t['gap'] <= -0.06])} 筆，9:05 收盤買 {(t[t['gap'] <= -0.06]['c0905'] - COST).mean():+.2%}")
text = "\n".join(lines)
print(text)
(ROOT / "results" / "網路策略" / "gapdown_intraday.txt").write_text(text, encoding="utf-8")
