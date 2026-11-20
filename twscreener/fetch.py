"""從 FinMind 下載台股資料，存成本機 CSV 快取（可中斷續抓）。

用法：
    python -m twscreener.fetch --universe cases          # 只抓案例表裡的股票（快，先驗證用）
    python -m twscreener.fetch --universe all            # 全上市櫃普通股（免費額度下要跑很久，可中斷續抓）
    python -m twscreener.fetch --stocks 2330 1785

環境變數 FINMIND_TOKEN：到 https://finmindtrade.com 註冊後取得，有 token 額度較高。
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

API = "https://api.finmindtrade.com/api/v4/data"
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CASES = ROOT / "個股案例表.csv"

# 每檔股票要抓的資料集 -> 本機資料夾名
STOCK_DATASETS = {
    "TaiwanStockPrice": "price",
    "TaiwanStockInstitutionalInvestorsBuySell": "inst",
    "TaiwanStockShareholding": "foreign_hold",
    "TaiwanStockMarginPurchaseShortSale": "margin",
}
# 可選：集保股權分散表（大戶、股東人數），可能需要付費方案
HOLDING_DATASET = ("TaiwanStockHoldingSharesPer", "holding")

MARKET_DATASETS = {
    # 加權指數
    ("TaiwanStockPrice", "TAIEX"): "taiex",
    # 全市場三大法人、融資
    ("TaiwanStockTotalInstitutionalInvestors", None): "market_inst",
    ("TaiwanStockTotalMarginPurchaseShortSale", None): "market_margin",
}


class RateLimiter:
    """免費版約每小時 600 次（有 token）/ 300 次（無 token），保守一點平均分配。"""

    def __init__(self, per_hour):
        self.interval = 3600.0 / per_hour
        self.last = 0.0

    def wait(self):
        dt = time.time() - self.last
        if dt < self.interval:
            time.sleep(self.interval - dt)
        self.last = time.time()


def get(dataset, data_id, start, end, token, limiter, retries=5):
    params = {"dataset": dataset, "start_date": start, "end_date": end}
    if data_id:
        params["data_id"] = data_id
    if token:
        params["token"] = token
    for attempt in range(retries):
        limiter.wait()
        try:
            r = requests.get(API, params=params, timeout=60)
            js = r.json()
        except Exception as e:  # 網路錯誤就重試
            print(f"  ! {dataset} {data_id}: {e}，重試", file=sys.stderr)
            time.sleep(2 ** attempt * 5)
            continue
        if js.get("status") == 200:
            return pd.DataFrame(js.get("data", []))
        msg = js.get("msg", "")
        if r.status_code == 402 or "limit" in msg.lower():
            print(f"  ! 超過額度，等 10 分鐘：{msg}", file=sys.stderr)
            time.sleep(600)
            continue
        print(f"  ! {dataset} {data_id}: {r.status_code} {msg}", file=sys.stderr)
        return None
    return None


def save(df, folder, name):
    path = DATA / folder / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def exists(folder, name):
    return (DATA / folder / f"{name}.csv").exists()


def case_stocks():
    ids = []
    with open(CASES, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["代號"]:
                ids.append(row["代號"])
    return sorted(set(ids))


def all_stocks(token, limiter):
    info = get("TaiwanStockInfo", None, "2000-01-01", "2100-01-01", token, limiter)
    save(info, "meta", "stock_info")
    # 只留 4 碼普通股（排除 ETF、權證、特別股）
    info = info[info["type"].isin(["twse", "tpex"])]
    ids = info["stock_id"].astype(str)
    ids = ids[ids.str.fullmatch(r"[1-9]\d{3}")]
    return sorted(set(ids))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", choices=["cases", "all"], default="cases")
    ap.add_argument("--stocks", nargs="*")
    ap.add_argument("--start", default="2017-01-01")
    ap.add_argument("--end", default=time.strftime("%Y-%m-%d"))
    ap.add_argument("--holding", action="store_true", help="也抓集保股權分散表（可能需要付費方案）")
    ap.add_argument("--refresh", action="store_true", help="已存在的檔案也重抓")
    ap.add_argument("--per-hour", type=int, default=None)
    args = ap.parse_args()

    token = os.environ.get("FINMIND_TOKEN", "")
    env_file = ROOT / ".env"
    if not token and env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("FINMIND_TOKEN="):
                token = line.split("=", 1)[1].strip()
    limiter = RateLimiter(args.per_hour or (550 if token else 280))
    if not token:
        print("提醒：沒有設定 FINMIND_TOKEN，額度較低。")

    for (ds, data_id), name in MARKET_DATASETS.items():
        if args.refresh or not exists("market", name):
            df = get(ds, data_id, args.start, args.end, token, limiter)
            if df is not None:
                save(df, "market", name)
                print(f"市場資料 {name}: {len(df)} 筆")

    if args.stocks:
        ids = args.stocks
    elif args.universe == "all":
        ids = all_stocks(token, limiter)
    else:
        ids = case_stocks()

    datasets = dict(STOCK_DATASETS)
    if args.holding:
        datasets[HOLDING_DATASET[0]] = HOLDING_DATASET[1]

    total = len(ids)
    for i, sid in enumerate(ids, 1):
        for ds, folder in datasets.items():
            if not args.refresh and exists(folder, sid):
                continue
            df = get(ds, sid, args.start, args.end, token, limiter)
            if df is not None:
                save(df, folder, sid)
        print(f"[{i}/{total}] {sid} 完成")


if __name__ == "__main__":
    main()
