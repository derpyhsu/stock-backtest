"""當沖盤中進出場，用 5 分 K 驗（2026-11-26）。資料只有最近 60 個交易日（scripts/fetch_intraday.py），樣本短，只能當初步線索。

只做多，當天進當天出。成本 0.4%（手續費約 9 折兩次＋當沖證交稅 0.15%）。
進場：
  E0 開盤買            第一根 5 分 K 開盤價
  E1 突破開盤 15 分高   09:15 之後第一次高過前三根的最高價，用那個價買（開盤就跳過用該根開盤價）
  E2 強開後回均價線     前 15 分鐘漲 1% 以上，之後第一次回到當日均價線（成交量加權），用均價買
  E3 過昨高            開盤在昨高以下，盤中第一次高過昨高，用昨高買
  E4 十點站在均價線上   10:00 時收在均價線上而且比昨收漲 2% 以上，用那根收盤價買
出場：
  X0 收盤賣  X1 停損 2%  X2 停損 2%／停利 4%（同一根都碰到算停損）  X3 收盤跌破均價線就賣  X4 11:00 賣
  停損停利用觸價價位算，沒有算滑價，所以是樂觀的。
選股（前一天收盤就知道，或開盤就知道）：全部、昨天漲 5% 以上、昨天漲停、今天開高 2% 以上。
每天一票：先算當天所有訊號的平均，再對天數平均；「前半／後半」是把期間對半切，看是不是兩段都一樣。
    python scripts/daytrade_intraday_test.py            # 5 分 K
    python scripts/daytrade_intraday_test.py 1m         # 1 分 K（只有最近 7 天）
    python scripts/daytrade_intraday_test.py since=2026-11-18   # 只算某天之後，拿來和 1 分 K 比同一段
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IV = "1m" if "1m" in sys.argv[1:] else "5m"
SINCE = next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("since=")), None)
SRC = ROOT / "data" / f"intraday{IV}"
K = 5 if IV == "1m" else 1        # 一根 5 分 K 等於幾根
COST, STOP, TAKE = 0.004, 0.02, 0.04
ENT = {"E0": "開盤買", "E1": "突破開盤 15 分高", "E2": "強開後回均價線", "E3": "過昨高", "E4": "十點站在均價線上"}
EXI = {"X0": "收盤賣", "X1": "停損 2%", "X2": "停損 2%／停利 4%", "X3": "跌破均價線賣", "X4": "11:00 賣"}


def exits(i, px, h, l, c, vw, t11):
    """從第 i 根（進場那根）之後開始看；回傳各出場方式的報酬（未扣成本）。"""
    n = len(c)
    out = {"X0": c[-1] / px - 1}
    r1 = r2 = r3 = None
    for j in range(i + 1, n):
        if r1 is None and l[j] <= px * (1 - STOP):
            r1 = -STOP
        if r2 is None:
            if l[j] <= px * (1 - STOP):
                r2 = -STOP
            elif h[j] >= px * (1 + TAKE):
                r2 = TAKE
        if r3 is None and c[j] < vw[j]:
            r3 = c[j] / px - 1
    out["X1"] = r1 if r1 is not None else out["X0"]
    out["X2"] = r2 if r2 is not None else out["X0"]
    out["X3"] = r3 if r3 is not None else out["X0"]
    k = min(max(t11, i), n - 1)
    out["X4"] = c[k] / px - 1
    return out


# Yahoo 的最後一根（收盤集合競價）常常不準，收盤價改用我們自己的日線
dc = pd.concat(pd.read_csv(x, dtype={"stock_id": str}, usecols=["date", "stock_id", "close"])
               for x in sorted((ROOT / "data" / "daily").glob("*.csv"))[-70:])
dc = dc.set_index([pd.to_datetime(dc["date"]), "stock_id"])["close"].to_dict()
rows = []
files = sorted(SRC.glob("*.csv"))
for f in files:
    d = pd.read_csv(f, parse_dates=["time"])
    d = d[(d["time"].dt.time >= pd.Timestamp("09:00").time()) & (d["time"].dt.time <= pd.Timestamp("13:30").time())]
    d["day"] = d["time"].dt.normalize()
    pc = ph = ppc = None
    for day, g in d.groupby("day"):
        o, h, l, c, v = (g[x].to_numpy(float).copy() for x in ("open", "high", "low", "close", "vol"))
        n = len(c)
        if (day, f.stem) in dc:
            c[-1] = dc[(day, f.stem)]
        if n < 30 * K:
            pc = ph = None
            continue
        if pc is not None and ppc is not None:
            cv = np.cumsum(v)
            vw = np.where(cv > 0, np.cumsum((h + l + c) / 3 * v) / np.where(cv > 0, cv, 1), c)
            mins = (g["time"].dt.hour * 60 + g["time"].dt.minute).to_numpy()
            t11 = int(np.searchsorted(mins, 11 * 60))
            gap, ychg = o[0] / pc - 1, pc / ppc - 1
            if gap < 0.095:
                e = {"E0": (0, o[0])}
                orh = h[:3 * K].max()
                for j in range(3 * K, n):
                    if h[j] > orh:
                        e["E1"] = (j, max(orh, o[j]))
                        break
                if c[3 * K - 1] / o[0] - 1 >= 0.01:
                    for j in range(3 * K, n):
                        if l[j] <= vw[j - 1]:
                            e["E2"] = (j, min(vw[j - 1], o[j]))
                            break
                if o[0] < ph:
                    for j in range(n):
                        if h[j] > ph:
                            e["E3"] = (j, max(ph, o[j]))
                            break
                j10 = int(np.searchsorted(mins, 10 * 60))
                if j10 < n - 1 and c[j10] > vw[j10] and c[j10] / pc - 1 >= 0.02:
                    e["E4"] = (j10, c[j10])
                for en, (i, px) in e.items():
                    if i >= n - 1 or px <= 0:
                        continue
                    for xn, r in exits(i, px, h, l, c, vw, t11).items():
                        rows.append((day, f.stem, en, xn, r - COST, ychg, gap))
        ppc, pc, ph = pc, c[-1], h.max()

t = pd.DataFrame(rows, columns=["day", "sid", "entry", "exit", "ret", "ychg", "gap"])
if SINCE:
    t = t[t["day"] >= SINCE]
if "dump" in sys.argv[1:]:      # 留每一筆，給 1 分 K 和 5 分 K 逐筆比對用
    t.to_csv(ROOT / "results" / "當沖" / f"trades_{IV}.csv", index=False)
days = sorted(t["day"].unique())
half = days[len(days) // 2]
sel = {"全部": t["ret"].notna(), "昨天漲 5% 以上": t["ychg"] >= 0.05, "昨天漲停": t["ychg"] >= 0.095, "今天開高 2% 以上": t["gap"] >= 0.02}
res = []
for sn, m in sel.items():
    for (en, xn), g in t[m].groupby(["entry", "exit"]):
        dm = g.groupby("day")["ret"].mean()
        res.append({"選股": sn, "進場": ENT[en], "出場": EXI[xn], "筆數": len(g), "天數": len(dm), "勝率": (g["ret"] > 0).mean(),
                    "每筆平均": dm.mean(), "前半": dm[dm.index < half].mean(), "後半": dm[dm.index >= half].mean(), "賺錢天數比": (dm > 0).mean()})
r = pd.DataFrame(res)
out = ROOT / "results" / "當沖"
out.mkdir(parents=True, exist_ok=True)
r.to_csv(out / f"daytrade_intraday_{IV}{'_' + SINCE if SINCE else ''}.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250, "display.unicode.east_asian_width", True, "display.max_rows", 200)
print(f"{IV} K；{pd.Timestamp(days[0]).date()} ~ {pd.Timestamp(days[-1]).date()}，{len(days)} 天，{len(files)} 檔；成本 {COST:.1%}")
for sn in sel:
    v = r[r["選股"] == sn].drop(columns="選股").copy()
    for c_ in ("勝率", "賺錢天數比"):
        v[c_] = v[c_].map("{:.0%}".format)
    for c_ in ("每筆平均", "前半", "後半"):
        v[c_] = v[c_].map("{:+.2%}".format)
    print(f"\n== {sn}")
    print(v.to_string(index=False))
