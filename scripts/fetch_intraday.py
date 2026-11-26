"""抓 5 分鐘 K（Yahoo Finance，只給最近 60 個交易日），存 data/intraday5m/<代號>.csv，重跑會接續合併。
加 1m 改抓 1 分鐘 K（只給最近 7 天），存 data/intraday1m/。

注意：每晚例行累積的是 scripts/orb_test.py fetch（data/intraday/，300 檔）。這支是 2026-11-26 比較 1 分 K 和 5 分 K 時
一次性用的，抓下來約 180MB，比完就刪了，不要留在 repo 裡（同步會整包推上 GitHub）。
Yahoo 不給更久以前的分鐘資料，所以要定期跑（至少每兩個月一次）才能累積歷史。
範圍：最近 20 日均量 1000 張以上、股價 10 元以上的股票（當沖要有量）。
    python scripts/fetch_intraday.py [1m]
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IV = "1m" if "1m" in sys.argv[1:] else "5m"
RNG = {"5m": "60d", "1m": "7d"}[IV]
OUT = ROOT / "data" / f"intraday{IV}"
MIN_VOL, MIN_PRICE = 1000, 10


def universe():
    files = sorted((ROOT / "data" / "daily").glob("*.csv"))[-20:]
    d = pd.concat(pd.read_csv(f, dtype={"stock_id": str}) for f in files)
    g = d.groupby("stock_id").agg(vol=("vol", "mean"), close=("close", "last"), market=("market", "last"))
    g = g[(g["vol"] >= MIN_VOL) & (g["close"] >= MIN_PRICE) & g.index.str.fullmatch(r"\d{4}")]
    return g["market"].map(lambda m: ".TW" if m == "上市" else ".TWO")


def fetch(sym):
    url = f"https://query2.finance.yahoo.com/v8/finance/chart/{sym}?interval={IV}&range={RNG}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    r = json.load(urllib.request.urlopen(req, timeout=30))["chart"]["result"][0]
    q = r["indicators"]["quote"][0]
    t = pd.to_datetime(r["timestamp"], unit="s", utc=True).tz_convert("Asia/Taipei").tz_localize(None)
    return pd.DataFrame({"time": t, "open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"], "vol": q["volume"]}).dropna()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    u = universe()
    fail = 0
    for i, (sid, suf) in enumerate(u.items(), 1):
        f = OUT / f"{sid}.csv"
        for attempt in range(3):
            try:
                d = fetch(sid + suf)
                break
            except Exception as e:  # noqa: BLE001
                d, err = None, type(e).__name__
                time.sleep(5 * (attempt + 1))
        if d is None or d.empty:
            fail += 1
            print(f"{sid} 失敗（{err if d is None else '沒有資料'}）")
            continue
        if f.exists():
            d = pd.concat([pd.read_csv(f, parse_dates=["time"]), d]).drop_duplicates("time", keep="last").sort_values("time")
        d.to_csv(f, index=False)
        if i % 100 == 0:
            print(f"{i}/{len(u)}")
        time.sleep(0.4)
    print(f"完成，{len(u) - fail}/{len(u)} 檔")


if __name__ == "__main__":
    sys.exit(main())
