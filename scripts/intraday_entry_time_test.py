"""進場那天盤中什麼時候買比較便宜（2026-11-27）。資料：data/intraday/ 的 5 分 K（300 檔、最近約 60 個交易日）。
現在的規則是開盤市價買。這裡看同一天其他時間點的價格相對開盤價是高還是低：
  一、全部股票每一天（當背景）  二、一年新高訊號的進場日  三、處置股的進場日（處置第 2 天）
另外測幾個只用到當下資訊的盤中規則：開盤後先看 30 分鐘再決定、掛開盤價 -1%／-2% 等到收盤前沒成交就收盤買。
數字＝買到的價格比開盤價貴（+）或便宜（−）多少，越負越好。只有約 60 天，當參考。
    python scripts/intraday_entry_time_test.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
T = ["09:05", "09:15", "09:30", "10:00", "10:30", "11:00", "12:00", "13:00", "13:25"]
rows = []
for f in sorted((ROOT / "data" / "intraday").glob("*.csv")):
    d = pd.read_csv(f, parse_dates=["time"])
    d["day"], d["hm"] = d["time"].dt.normalize(), d["time"].dt.strftime("%H:%M")
    for day, g in d.groupby("day"):
        g = g[g["hm"] <= "13:30"]
        if len(g) < 40 or g["volume"].sum() <= 0:
            continue
        o = g["open"].iat[0]
        hm, cl = g["hm"].values, g["close"].values
        r = {"stock_id": f.stem, "day": day, "open": o, "close": g["close"].iat[-1], "low": g["low"].min(), "high": g["high"].max(),
             "vwap": (g["close"] * g["volume"]).sum() / g["volume"].sum()}
        for t in T:
            k = np.searchsorted(hm, t, side="right") - 1
            r[t] = cl[k] if k >= 0 else np.nan
        am = g[g["hm"] <= "09:30"]
        r["lo30"], r["hi30"] = am["low"].min(), am["high"].max()
        later = g[g["hm"] > "09:30"]
        r["lo_after"] = later["low"].min() if len(later) else np.nan
        rows.append(r)
x = pd.DataFrame(rows)
x = x.sort_values(["stock_id", "day"]).reset_index(drop=True)
x["prev"] = x.groupby("stock_id")["close"].shift(1)
x["gap"] = x["open"] / x["prev"] - 1
x = x[(x["open"] < x["prev"] * 1.095) | x["prev"].isna()]  # 開盤漲停買不到


def table(d, title):
    out = [f"== {title}：{len(d)} 筆（{d['day'].nunique()} 天）"]
    cols = T + ["vwap", "close"]
    m = pd.DataFrame({"平均": [(d[c] / d["open"] - 1).mean() for c in cols], "中位數": [(d[c] / d["open"] - 1).median() for c in cols],
                      "比開盤便宜的比例": [(d[c] < d["open"]).mean() for c in cols]}, index=[c.replace("vwap", "全天均價").replace("close", "收盤") for c in cols])
    out.append(m.map(lambda v: f"{v:+.2%}").to_string())
    out.append(f"當天最低價比開盤低：平均 {(d['low'] / d['open'] - 1).mean():+.2%}、中位 {(d['low'] / d['open'] - 1).median():+.2%}；最高價比開盤高：平均 {(d['high'] / d['open'] - 1).mean():+.2%}")
    for k in (0.01, 0.02, 0.03):
        hit = d["low"] <= d["open"] * (1 - k)
        cost = np.where(hit, -k, d["close"] / d["open"] - 1)
        out.append(f"掛開盤價 -{k:.0%}，沒成交收盤買：成交率 {hit.mean():.0%}，平均買價比開盤 {cost.mean():+.2%}（沒成交的那些收盤比開盤 {(d['close'] / d['open'] - 1)[~hit].mean():+.2%}）")
    up = d["09:30"] > d["open"]
    out.append(f"開盤後 30 分鐘是漲的（{up.mean():.0%}）：之後到收盤再 {(d['close'] / d['09:30'] - 1)[up].mean():+.2%}；是跌的：之後到收盤再 {(d['close'] / d['09:30'] - 1)[~up].mean():+.2%}")
    w = np.where(up, d["09:30"], d["close"]) / d["open"] - 1
    out.append(f"規則：30 分鐘漲就 09:30 買、跌就等收盤買 → 平均買價比開盤 {w.mean():+.2%}")
    return out


lines = [f"5 分 K：{x['stock_id'].nunique()} 檔、{x['day'].min():%Y-%m-%d} ~ {x['day'].max():%Y-%m-%d}。數字＝該時間點價格比開盤價高（+）低（−）多少"]
lines += table(x, "全部股票每一天（背景）")
for name, f in (("開高 3% 以上", x["gap"] >= 0.03), ("開高 0~3%", (x["gap"] >= 0) & (x["gap"] < 0.03)), ("開低", x["gap"] < 0)):
    lines += [""] + table(x[f], f"全部股票・{name}")
s1 = pd.read_csv(ROOT / "results" / "market_signals.csv", dtype={"stock_id": str}, parse_dates=["signal_date"])
s1 = s1[s1["strategy"] == "S1_一年新高"]
days = sorted(pd.to_datetime([f.stem for f in (ROOT / "data" / "daily").glob("*.csv")]))
nxt = {a: b for a, b in zip(days[:-1], days[1:])}
s1["day"] = s1["signal_date"].map(nxt)
lines += [""] + table(x.merge(s1[["stock_id", "day"]].drop_duplicates(), on=["stock_id", "day"]), "一年新高訊號的進場日（訊號隔天）")
dd = pd.read_csv(ROOT / "data" / "disposition.csv", dtype={"stock_id": str}, parse_dates=["start"])
dd["day"] = dd["start"].map(nxt)
lines += [""] + table(x.merge(dd[["stock_id", "day"]].drop_duplicates(), on=["stock_id", "day"]), "處置股進場日（處置第 2 天；處置股是分盤撮合，5 分 K 只當參考）")
text = "\n".join(lines)
print(text)
(ROOT / "results" / "規則依據" / "intraday_entry_time.txt").write_text(text, encoding="utf-8")
