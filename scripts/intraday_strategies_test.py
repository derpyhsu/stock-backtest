"""台股當沖策略總回測：用 data/intraday 的 5 分 K（scripts/orb_test.py fetch 抓的，約 60 個交易日、成交最大 300 檔）。

每個策略都在當天平倉（13:20 那根收盤），扣當沖成本 0.435%（證交稅 0.15%＋手續費 0.1425%×2，未折扣），
和「同一天同一批股票隨機當沖（9:15 收盤買、13:20 收盤賣）」比較；做空的對照是隨機放空。
同一根 K 同時碰到停損和停利算停損；跳空越過停損用開盤價成交。

策略（訊號只用當下以前的資料）：
 1 ORB 多：9:00～9:15 區間，9:15～12:30 第一根收盤站上區間高點買；停損區間中點、停利 1R。
 2 ORB 空：第一根收盤跌破區間低點放空；停損區間中點、停利 1R。
 3 跳空續強：開盤比昨收高 2%～7%，第一根 5 分 K 收紅 → 9:05 收盤買；停損第一根低點、停利 2R。
 4 跳空回補（空）：開盤比昨收高 3% 以上，第一根收黑 → 9:05 收盤放空；停損第一根高點、停利 2R。
 5 VWAP 站回：9:30 後，前一根在當日均價線下、這根收盤站上，且當天比昨收漲；停損當日目前最低、停利 2R，12:00 前進場。
 6 突破昨高：9:15 後第一根收盤站上昨天最高價；停損 -1%、停利 +2%。
 7 隔天候選＋開盤買：昨天漲 5% 以上、收在當天最高 1% 內、量 >= 20 日均量 2 倍（前一晚就能選）→ 開盤買、收盤賣。
 8 隔天候選＋ORB：同 7 的股票，只在 ORB 多訊號出現才買。
    python scripts/intraday_strategies_test.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BARS = ROOT / "data" / "intraday"
OUT = ROOT / "results" / "當沖工具"
COST = 0.0015 + 0.001425 * 2


def daily_info():
    files = sorted((ROOT / "data" / "daily").glob("*.csv"))[-110:]
    d = pd.concat(pd.read_csv(f, dtype={"stock_id": str}, usecols=["date", "stock_id", "high", "close", "vol"]) for f in files)
    d["date"] = pd.to_datetime(d["date"]).dt.date
    d = d.sort_values(["stock_id", "date"])
    g = d.groupby("stock_id")
    d["prev_close"] = g["close"].shift(1)
    d["chg"] = d["close"] / d["prev_close"] - 1
    d["vol20"] = g["vol"].transform(lambda s: s.shift(1).rolling(20).mean())
    d["cand"] = (d["chg"] >= 0.05) & (d["close"] >= d["high"] * 0.99) & (d["vol"] >= 2 * d["vol20"])
    # 用「前一個交易日」的資料：shift 到下一天
    nxt = d[["stock_id", "date", "close", "high", "cand"]].copy()
    nxt[["pc", "ph", "pcand"]] = g[["close", "high", "cand"]].shift(0)
    nxt["date"] = g["date"].shift(-1)
    return nxt.dropna(subset=["date"]).set_index(["stock_id", "date"])[["pc", "ph", "pcand"]]


def simulate(rest, entry, stop, target, side):
    """從 rest 的第一根開始看，回傳出場價。side=1 多、-1 空。"""
    for _, b in rest.iterrows():
        if side == 1:
            if b["low"] <= stop:
                return min(stop, b["open"]), "停損"
            if b["high"] >= target:
                return max(target, b["open"]), "停利"
        else:
            if b["high"] >= stop:
                return max(stop, b["open"]), "停損"
            if b["low"] <= target:
                return min(target, b["open"]), "停利"
    return (rest["close"].iat[-1] if len(rest) else entry), "收盤"


def trade(name, day, sid, entry, stop, target, side, rest):
    if side * (entry - stop) <= entry * 0.001:
        return None
    exit_p, how = simulate(rest, entry, stop, target, side)
    g = side * (exit_p / entry - 1)
    return {"strategy": name, "day": day, "stock_id": sid, "side": side, "gross": g, "ret": g - COST, "how": how}


def run_stock(sid, bars, info):
    rows, base = [], []
    for day, g in bars.groupby(bars["time"].dt.date):
        g = g.set_index("time").sort_index()
        hm = g.index.strftime("%H:%M")
        g = g[(hm >= "09:00") & (hm <= "13:20")]
        hm = g.index.strftime("%H:%M")
        if len(g) < 50 or hm[0] != "09:00":
            continue
        try:
            pc, ph, pcand = info.loc[(sid, day)]
        except KeyError:
            continue
        t915 = g[hm == "09:10"]
        if t915.empty:
            continue
        b_gross = g["close"].iat[-1] / t915["close"].iat[0] - 1
        base.append({"day": day, "stock_id": sid, "gross": b_gross})
        rng = g.iloc[:3]
        hi, lo = rng["high"].max(), rng["low"].min()
        mid = (hi + lo) / 2
        win = g[(hm >= "09:15") & (hm <= "12:30")]
        # 1、2、8 ORB
        up = win[win["close"] > hi]
        if len(up):
            t0, e = up.index[0], up["close"].iat[0]
            r = trade("1 ORB 多", day, sid, e, mid, e + (e - mid), 1, g[g.index > t0])
            rows.append(r)
            if pcand:
                rows.append(trade("8 隔天候選＋ORB", day, sid, e, mid, e + (e - mid), 1, g[g.index > t0]))
        dn = win[win["close"] < lo]
        if len(dn):
            t0, e = dn.index[0], dn["close"].iat[0]
            rows.append(trade("2 ORB 空", day, sid, e, mid, e - (mid - e), -1, g[g.index > t0]))
        # 3、4 跳空
        o, b1 = g["open"].iat[0], g.iloc[0]
        gap = o / pc - 1
        rest = g.iloc[1:]
        if 0.02 <= gap <= 0.07 and b1["close"] > b1["open"]:
            e, s = b1["close"], b1["low"]
            rows.append(trade("3 跳空續強", day, sid, e, s, e + 2 * (e - s), 1, rest))
        if gap >= 0.03 and b1["close"] < b1["open"]:
            e, s = b1["close"], b1["high"]
            rows.append(trade("4 跳空回補（空）", day, sid, e, s, e - 2 * (s - e), -1, rest))
        # 5 VWAP 站回
        tp = (g["high"] + g["low"] + g["close"]) / 3
        vwap = (tp * g["volume"]).cumsum() / g["volume"].cumsum().replace(0, np.nan)
        cond = (g["close"] > vwap) & (g["close"].shift(1) < vwap.shift(1)) & (g["close"] > pc) & (hm >= "09:30") & (hm <= "12:00")
        if cond.any():
            t0 = cond[cond].index[0]
            e, s = g.at[t0, "close"], g.loc[:t0, "low"].min()
            rows.append(trade("5 VWAP 站回", day, sid, e, s, e + 2 * (e - s), 1, g[g.index > t0]))
        # 6 突破昨高
        br = g[(hm >= "09:15") & (hm <= "12:30") & (g["close"] > ph)]
        if len(br):
            t0, e = br.index[0], br["close"].iat[0]
            rows.append(trade("6 突破昨高", day, sid, e, e * 0.99, e * 1.02, 1, g[g.index > t0]))
        # 7 隔天候選開盤買
        if pcand:
            g7 = g["close"].iat[-1] / o - 1
            rows.append({"strategy": "7 隔天候選＋開盤買", "day": day, "stock_id": sid, "side": 1, "gross": g7,
                         "ret": g7 - COST, "how": "收盤"})
    return [r for r in rows if r], base


def main():
    info = daily_info()
    rows, base = [], []
    for f in sorted(BARS.glob("*.csv")):
        b = pd.read_csv(f, parse_dates=["time"])
        r, bs = run_stock(f.stem, b, info)
        rows += r
        base += bs
    t, b = pd.DataFrame(rows), pd.DataFrame(base)
    day_base = b.groupby("day")["gross"].mean()
    t["x"] = t["gross"] - t["side"] * t["day"].map(day_base)  # 比同日隨機同方向
    days = sorted(t["day"].unique())
    half = days[len(days) // 2]
    lines = [f"{b['day'].min()} ~ {b['day'].max()}，{b['day'].nunique()} 天，{b['stock_id'].nunique()} 檔；成本 {COST:.3%}",
             f"隨機當沖（9:15 買、13:20 賣）每筆未扣成本 {b['gross'].mean():+.3%}；這段期間大盤偏空，做多吃虧、做空占便宜，要看「比隨機」。", ""]
    hdr = f"{'策略':<16}{'筆數':>6}{'勝率':>6}{'每筆(扣成本)':>12}{'未扣成本':>10}{'比同日隨機':>10}{'前半':>9}{'後半':>9}{'停利/停損/收盤':>16}"
    lines.append(hdr)
    for name, g in t.groupby("strategy"):
        hw = g["how"].value_counts(normalize=True)
        lines.append(f"{name:<16}{len(g):>6}{(g['ret'] > 0).mean():>6.0%}{g['ret'].mean():>+12.3%}{g['gross'].mean():>+10.3%}"
                     f"{g['x'].mean():>+10.3%}{g[g['day'] < half]['ret'].mean():>+9.3%}{g[g['day'] >= half]['ret'].mean():>+9.3%}"
                     f"{hw.get('停利', 0):>8.0%}/{hw.get('停損', 0):.0%}/{hw.get('收盤', 0):.0%}")
    text = "\n".join(lines)
    print(text)
    (OUT / "intraday_strategies.txt").write_text(text, encoding="utf-8")
    t.to_csv(OUT / "intraday_trades.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
