"""抓季報損益表（FinMind TaiwanStockFinancialStatements，免費額度逐檔抓，可中斷續抓）。
FinMind 原始資料不能轉散布，所以存在專案外的 ../private-data/financials/，不進 repo。
只抓曾經達到流動性門檻（20 日均量 >= 500 張、股價 >= 10）的股票。
    python scripts/fetch_financials.py
"""
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from twscreener.market import load_panel  # noqa: E402
from twscreener.simulate import prices  # noqa: E402

OUT = ROOT.parent / "private-data" / "financials"
OUT.mkdir(parents=True, exist_ok=True)
KEEP = {"Revenue", "GrossProfit", "OperatingIncome", "IncomeAfterTaxes", "EPS", "PreTaxIncome", "EquityAttributableToOwnersOfParent"}
tok = next((l.split("=", 1)[1].strip() for l in open(ROOT / ".env") if l.startswith("FINMIND_TOKEN")), "")
raw, adj, opn, vol = prices(load_panel())
liq = ((vol.rolling(20).mean() >= 500) & (raw >= 10)).any()
ids = [s for s in raw.columns[liq] if not (OUT / f"{s}.csv").exists()]
print(f"還有 {len(ids)} 檔", flush=True)
for n, s in enumerate(ids):
    while True:
        try:
            r = requests.get("https://api.finmindtrade.com/api/v4/data", timeout=60, headers={"Authorization": f"Bearer {tok}"} if tok else {},
                             params={"dataset": "TaiwanStockFinancialStatements", "data_id": s, "start_date": "2019-01-01"})
            js = r.json()
        except Exception as e:  # noqa: BLE001
            print(f"  ! {type(e).__name__}，等 30 秒", flush=True)
            time.sleep(30)
            continue
        if r.status_code != 200 or js.get("status") != 200:
            print(f"  ! FinMind {r.status_code} {str(js.get('msg', ''))[:40]}，等 10 分鐘", flush=True)
            time.sleep(600)
            continue
        break
    d = pd.DataFrame(js.get("data", []))
    if len(d):
        d = d[d["type"].isin(KEEP)][["date", "stock_id", "type", "value"]]
    d.to_csv(OUT / f"{s}.csv", index=False)
    if n % 50 == 0:
        print(f"{n}/{len(ids)} {s} {len(d)} 列", flush=True)
    time.sleep(6.2)
print("完成", flush=True)
