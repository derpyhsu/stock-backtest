"""補月營收的實際公布日（FinMind TaiwanStockMonthRevenue 的 create_time，2026-05 起才有）。
只抓需要的股票（預設：2026-04 起營收加速事件的股票），免費帳號每小時 600 次，每次間隔 6 秒。
結果：data/revenue_dates.csv（stock_id, ym, announced）。
    python scripts/fetch_revenue_dates.py
    python scripts/fetch_revenue_dates.py --refresh   # 每月 15 號後跑一次，補最新月份
"""
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from twscreener.fundamentals import revenue_features  # noqa: E402

OUT = ROOT / "data" / "revenue_dates.csv"
tok = next(l.split("=", 1)[1].strip() for l in open(ROOT / ".env") if l.startswith("FINMIND"))
rev = revenue_features(pd.read_csv(ROOT / "data/revenue.csv", dtype={"stock_id": str})).sort_values(["stock_id", "ym"])
rev["prev3"] = rev.groupby("stock_id")["yoy"].transform(lambda s: s.shift(1).rolling(3).mean())
ev = rev[(rev["yoy"] >= 0.3) & (rev["yoy"] - rev["prev3"] >= 0.1) & rev["rev_high12"].astype(bool) & (rev["ym"] >= "2026-04")]
old = pd.read_csv(OUT, dtype=str) if OUT.exists() else pd.DataFrame(columns=["stock_id", "ym", "announced"])
if "--refresh" in sys.argv:  # 每月 15 號後重抓：只抓最新營收月份的事件股票
    ev = ev[ev["ym"] == ev["ym"].max()]
    done = set(old[old["ym"] == ev["ym"].max()]["stock_id"])
else:
    done = set(old["stock_id"])
todo = sorted(set(ev["stock_id"]) - done)
print(f"要抓 {len(todo)} 檔", flush=True)
for n, sid in enumerate(todo, 1):
    for attempt in range(3):
        try:
            r = requests.get("https://api.finmindtrade.com/api/v4/data", timeout=60,
                             params={"dataset": "TaiwanStockMonthRevenue", "data_id": sid, "start_date": "2026-04-01", "token": tok}).json()
            break
        except Exception:
            time.sleep(30)
    else:
        continue
    if r.get("status") == 402 or "upper limit" in str(r.get("msg", "")):
        print("達到次數上限，等 10 分鐘", flush=True)
        time.sleep(600)
        continue
    rows = [{"stock_id": sid, "ym": f"{x['revenue_year']}-{int(x['revenue_month']):02d}", "announced": x["create_time"]}
            for x in r.get("data", [])]
    if not rows:
        rows = [{"stock_id": sid, "ym": "", "announced": ""}]
    pd.DataFrame(rows).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
    if n % 50 == 0:
        print(f"{n}/{len(todo)}", flush=True)
    time.sleep(6)
d = pd.read_csv(OUT, dtype=str).drop_duplicates(["stock_id", "ym"], keep="last")
d.to_csv(OUT, index=False)
print("完成", flush=True)
