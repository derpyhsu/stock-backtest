"""網路上找到、有研究或明確規則的當沖做多依據，用台股 5 分 K（data/intraday）回測。成本 0.435%，當天平倉。

1 熱門股 ORB（Zarattini, Barbon & Aziz 2024「Stocks in Play」）：
  前 15 分鐘成交量 ÷ 過去 14 天同時段平均 = 相對量（原文用第一根 5 分 K；Yahoo 台股 9:00 那根量常是 0）；相對量 >= 1，且每天只取相對量前 20 檔。
  前 15 分鐘收紅才做多：之後價格突破區間高點就買在高點（跳空就用開盤價），停損區間低點，持有到收盤（13:20）。
2 同上，但相對量前 20 檔不限第一根紅黑、突破第一根高點（給對照）。
3 日內動能（Gao, Han, Li & Zhou 2018 JFE；個股版見後續研究）：
  昨收到 9:30 的漲幅 >= 2%，12:50 收盤買、13:20 收盤賣（最後半小時）。
4 早盤量能＋連紅創高（XQ 範例）：9:30 前累計量 >= 昨天全天量 20%、開高 < 6%，連 3 根紅 K 且收盤創當天新高 → 買，停損當天最低、收盤平倉。
對照：同一天同一批股票 9:15 買、13:20 賣。看扣成本後每筆、前後半段。
    python scripts/intraday_research_test.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BARS = ROOT / "data" / "intraday"
COST = 0.0015 + 0.001425 * 2


def load():
    frames = []
    for f in sorted(BARS.glob("*.csv")):
        b = pd.read_csv(f, parse_dates=["time"])
        b["stock_id"] = f.stem
        frames.append(b)
    b = pd.concat(frames)
    b["day"] = b["time"].dt.normalize()
    b["hm"] = b["time"].dt.strftime("%H:%M")
    b = b[(b["hm"] >= "09:00") & (b["hm"] <= "13:20")]
    b = b[b["day"] < pd.Timestamp.now().normalize()]
    return b


def daily_prev():
    files = sorted((ROOT / "data" / "daily").glob("*.csv"))[-90:]
    d = pd.concat(pd.read_csv(f, dtype={"stock_id": str}, usecols=["date", "stock_id", "close", "vol"]) for f in files)
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values(["stock_id", "date"])
    d["pc"] = d.groupby("stock_id")["close"].shift(1)
    d["pv"] = d.groupby("stock_id")["vol"].shift(1) * 1000  # 張 → 股（Yahoo 成交量是股）
    return d.set_index(["stock_id", "date"])[["pc", "pv"]]


def exit_long(rest, stop):
    for _, r in rest.iterrows():
        if r["low"] <= stop:
            return min(stop, r["open"]), "停損"
    return (rest["close"].iat[-1] if len(rest) else np.nan), "收盤"


def main():
    b = load()
    prev = daily_prev()
    # Yahoo 台股 9:00 那根量常是 0（開盤集合競價量沒算進去），改用前 15 分鐘（9:00～9:10 三根）合計量
    first = b[b["hm"] <= "09:10"].groupby(["stock_id", "day"])["volume"].sum()
    rv = first.unstack(0)
    rel = (rv / rv.rolling(14).mean().shift(1)).stack().rename("rel")
    rows, base = [], []
    for (sid, day), g in b.groupby(["stock_id", "day"]):
        g = g.set_index("hm")
        if len(g) < 50 or "09:00" not in g.index or "09:10" not in g.index:
            continue
        try:
            pc, pv = prev.loc[(sid, day)]
        except KeyError:
            continue
        if not pc or np.isnan(pc):
            continue
        base.append({"day": day, "g": g["close"].iat[-1] / g.at["09:10", "close"] - 1})
        r = rel.get((day, sid), np.nan)
        r15 = g[g.index <= "09:10"]
        f1 = pd.Series({"high": r15["high"].max(), "low": r15["low"].min(), "open": r15["open"].iat[0], "close": r15["close"].iat[-1]})
        # 1、2 熱門股 ORB
        if r >= 1:
            br = g[(g.index >= "09:15") & (g["high"] > f1["high"])]
            if len(br):
                hm0 = br.index[0]
                e = max(f1["high"], br["open"].iat[0])
                ex, how = exit_long(g[g.index > hm0], f1["low"])
                for name, ok in (("1 熱門股ORB（第一根紅）", f1["close"] > f1["open"]), ("2 熱門股ORB（不看紅黑）", True)):
                    if ok and e > f1["low"]:
                        rows.append({"s": name, "day": day, "sid": sid, "rel": r, "g": ex / e - 1, "how": how})
        # 3 日內動能
        if "09:25" in g.index and "12:50" in g.index:
            m30 = g.at["09:25", "close"] / pc - 1
            if m30 >= 0.02:
                rows.append({"s": "3 日內動能（最後半小時）", "day": day, "sid": sid, "rel": r,
                             "g": g["close"].iat[-1] / g.at["12:50", "close"] - 1, "how": "收盤"})
        # 4 早盤量能＋連紅創高
        gap = g["open"].iat[0] / pc - 1
        early = g[g.index <= "09:25"]
        if gap < 0.06 and pv > 0 and early["volume"].sum() >= 0.2 * pv:
            red = g["close"] > g["open"]
            newhi = g["close"] >= g["high"].cummax()
            trig = red & red.shift(1, fill_value=False) & red.shift(2, fill_value=False) & newhi & (g.index >= "09:25") & (g.index <= "12:00")
            if trig.any():
                hm0 = trig[trig].index[0]
                e = g.at[hm0, "close"]
                stop = g.loc[:hm0, "low"].min()
                ex, how = exit_long(g[g.index > hm0], stop)
                rows.append({"s": "4 早盤量能＋連紅創高", "day": day, "sid": sid, "rel": r, "g": ex / e - 1, "how": how})
    t, bs = pd.DataFrame(rows).dropna(subset=["g"]), pd.DataFrame(base)
    # 1、2 每天只取相對量前 20
    keep = []
    for s, g in t.groupby("s"):
        if s.startswith(("1", "2")):
            g = g.sort_values("rel", ascending=False).groupby("day").head(20)
        keep.append(g)
    t = pd.concat(keep)
    t["ret"] = t["g"] - COST
    t["x"] = t["g"] - t["day"].map(bs.groupby("day")["g"].mean())
    days = sorted(t["day"].unique())
    half = days[len(days) // 2]
    lines = [f"{bs['day'].min():%Y-%m-%d} ~ {bs['day'].max():%Y-%m-%d}（{bs['day'].nunique()} 天，大盤偏空）；成本 {COST:.3%}；"
             f"隨機當沖每筆未扣成本 {bs['g'].mean():+.3%}", ""]
    for s, g in t.groupby("s"):
        hw = g["how"].value_counts(normalize=True).map("{:.0%}".format).to_dict()
        daily = g.groupby("day")["ret"].mean()
        lines.append(f"{s}：{len(g)} 筆，勝率 {(g['ret'] > 0).mean():.0%}，每筆扣成本 {g['ret'].mean():+.3%}（未扣 {g['g'].mean():+.3%}），"
                     f"比同日隨機 {g['x'].mean():+.3%}，前半 {g[g['day'] < half]['ret'].mean():+.3%}／後半 {g[g['day'] >= half]['ret'].mean():+.3%}，"
                     f"每天平均為正 {(daily > 0).mean():.0%}，出場 {hw}")
    text = "\n".join(lines)
    print(text)
    (ROOT / "results" / "當沖工具" / "intraday_research.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
