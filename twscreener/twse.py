"""從證交所、櫃買中心抓「全市場一天」的資料，一天只要 6 次請求。

每天存成 data/daily/YYYYMMDD.csv（休市日存 YYYYMMDD.none），可中斷續抓。

用法：
    python -m twscreener.twse                       # 補到今天（已抓過的日子會跳過）
    python -m twscreener.twse --start 2023-10-01    # 回補歷史

欄位：stock_id, name, market, open, high, low, close, vol(張),
      foreign_net(張，外資不含自營), trust_net(張), margin_bal(融資餘額，張)
另存 data/daily_index/YYYYMMDD.csv：加權指數收盤、全市場外資買賣超（張）
"""
import argparse
import sys
import threading
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
DAILY = ROOT / "data" / "daily"
DAILY_INDEX = ROOT / "data" / "daily_index"
UA = {"User-Agent": "Mozilla/5.0"}
GAP = {"twse": 3.0, "tpex": 2.0}  # 兩次請求之間至少間隔幾秒，太快會被證交所暫時封鎖

PRICES_ONLY = False
FILL_CHIPS = False  # 回補舊資料時只抓價格（跳過法人、融資），請求少三分之二；之後用 --fill-chips 補
_last = {"twse": 0.0, "tpex": 0.0}
_lock = {"twse": threading.Lock(), "tpex": threading.Lock()}


def _get(site, url, retries=12):
    for attempt in range(retries):
        # 只控制「開始」的間隔，下載本身可以同時進行
        with _lock[site]:
            dt = time.time() - _last[site]
            if dt < GAP[site]:
                time.sleep(GAP[site] - dt)
            _last[site] = time.time()
        try:
            r = requests.get(url, headers=UA, timeout=90, allow_redirects=False)
            if r.status_code == 200:
                return r.json()
            err = f"HTTP {r.status_code}（被限流）"
        except Exception as e:
            err = e.__class__.__name__
        wait = min(10 * (attempt + 1), 120)
        print(f"  ! {site} {err}，等 {wait} 秒重試", file=sys.stderr, flush=True)
        with _lock[site]:  # 被限流時，同一個交易所的其他請求也一起暫停
            _last[site] = time.time() + wait - GAP[site]
        time.sleep(wait)
    raise RuntimeError(f"{site} 連續失敗：{url}")


def _num(s):
    s = str(s).replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _is_stock(code):
    return len(code) == 4 and code.isdigit() and code[0] != "0"


def twse_day(d):
    ds = d.strftime("%Y%m%d")
    base = "https://www.twse.com.tw/rwd/zh"
    px = _get("twse", f"{base}/afterTrading/MI_INDEX?date={ds}&type=ALLBUT0999&response=json")
    if px.get("stat") != "OK":
        return None, None
    taiex = None
    quotes = None
    for t in px.get("tables", []):
        title = t.get("title") or ""
        if "價格指數(臺灣證券交易所)" in title:
            for row in t["data"]:
                if row[0] == "發行量加權股價指數":
                    taiex = _num(row[1])
        if "每日收盤行情" in title:
            quotes = t
    rows = {}
    for r in quotes["data"] if quotes else []:
        code = r[0].strip()
        if not _is_stock(code):
            continue
        rows[code] = {"stock_id": code, "name": r[1].strip(), "market": "上市",
                      "open": _num(r[5]), "high": _num(r[6]), "low": _num(r[7]), "close": _num(r[8]),
                      "vol": _num(r[2]) / 1000}
    if PRICES_ONLY:
        return list(rows.values()), {"taiex": taiex}
    inst = _get("twse", f"{base}/fund/T86?date={ds}&selectType=ALLBUT0999&response=json")
    mkt_foreign = 0.0
    for r in inst.get("data", []) if inst.get("stat") == "OK" else []:
        code = r[0].strip()
        if code in rows:
            rows[code]["foreign_net"] = _num(r[4]) / 1000
            rows[code]["trust_net"] = _num(r[10]) / 1000
            mkt_foreign += _num(r[4]) / 1000
    mg = _get("twse", f"{base}/marginTrading/MI_MARGN?date={ds}&selectType=ALL&response=json")
    for t in mg.get("tables", []) if mg.get("stat") == "OK" else []:
        if "彙總" in (t.get("title") or ""):
            for r in t["data"]:
                code = r[0].strip()
                if code in rows:
                    rows[code]["margin_bal"] = _num(r[6])
    return list(rows.values()), {"taiex": taiex, "twse_foreign_net": mkt_foreign}


def tpex_day(d):
    ds = d.strftime("%Y/%m/%d")
    base = "https://www.tpex.org.tw/www/zh-tw"
    px = _get("tpex", f"{base}/afterTrading/dailyQuotes?date={ds}&response=json")
    tabs = px.get("tables") or []
    if not tabs or not tabs[0].get("data"):
        return None
    rows = {}
    for r in tabs[0]["data"]:
        code = r[0].strip()
        if not _is_stock(code):
            continue
        rows[code] = {"stock_id": code, "name": r[1].strip(), "market": "上櫃",
                      "open": _num(r[4]), "high": _num(r[5]), "low": _num(r[6]), "close": _num(r[2]),
                      "vol": _num(r[8]) / 1000}
    if PRICES_ONLY:
        return list(rows.values())
    inst = _get("tpex", f"{base}/insti/dailyTrade?type=Daily&sect=AL&date={ds}&response=json")
    for t in inst.get("tables") or []:
        for r in t.get("data") or []:
            code = r[0].strip()
            if code in rows:
                rows[code]["foreign_net"] = _num(r[4]) / 1000
                rows[code]["trust_net"] = _num(r[13]) / 1000
    mg = _get("tpex", f"{base}/margin/balance?date={ds}&response=json")
    for t in mg.get("tables") or []:
        for r in t.get("data") or []:
            code = r[0].strip()
            if code in rows:
                rows[code]["margin_bal"] = _num(r[6])
    return list(rows.values())


def fetch_day(d):
    ds = d.strftime("%Y%m%d")
    out = DAILY / f"{ds}.csv"
    none = DAILY / f"{ds}.none"
    if none.exists():
        return "skip"
    if out.exists():
        head = out.open(encoding="utf-8").readline()
        if FILL_CHIPS:  # 只重抓「只有價格」的舊檔
            if "foreign_net" in head:
                return "skip"
        # 白天抓的檔案還沒有融資（晚上才公布），最近幾天的要重抓
        elif "margin_bal" in head or (date.today() - d).days > 5:
            return "skip"
    twse_rows, idx = twse_day(d)
    if twse_rows is None:
        if d < date.today():  # 今天可能只是還沒公布，不要記成休市
            none.touch()
        return "休市或尚未公布"
    tpex_rows = tpex_day(d) or []
    df = pd.DataFrame(twse_rows + tpex_rows)
    df.insert(0, "date", d.isoformat())
    df.to_csv(out, index=False)
    pd.DataFrame([{"date": d.isoformat(), **idx}]).to_csv(DAILY_INDEX / f"{ds}.csv", index=False)
    return f"{len(twse_rows)} 上市 + {len(tpex_rows)} 上櫃"


def _safe(d):
    try:
        return fetch_day(d)
    except Exception as e:  # 單日失敗不要中斷整批，下次再補
        return f"失敗，下次再補（{e.__class__.__name__}）"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=None, help="預設：從最後一個已抓的日子往後補")
    ap.add_argument("--end", default=date.today().isoformat())
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--prices-only", action="store_true", help="只抓價格，跳過法人與融資（回補舊資料用）")
    ap.add_argument("--fill-chips", action="store_true", help="把只有價格的舊檔重抓成完整資料")
    args = ap.parse_args()
    global PRICES_ONLY, FILL_CHIPS
    PRICES_ONLY, FILL_CHIPS = args.prices_only, args.fill_chips
    DAILY.mkdir(parents=True, exist_ok=True)
    DAILY_INDEX.mkdir(parents=True, exist_ok=True)
    if args.start:
        start = date.fromisoformat(args.start)
    else:
        done = sorted(p.stem for p in DAILY.glob("*.csv"))
        start = date.fromisoformat(f"{done[-1][:4]}-{done[-1][4:6]}-{done[-1][6:]}") if done else date.today() - timedelta(days=30)
    end = date.fromisoformat(args.end)
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    days = [d for d in days if d.weekday() < 5]
    # 由新到舊抓，中斷時最近的資料已經先有了
    days.reverse()
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(args.workers) as ex:
        for d, res in zip(days, ex.map(_safe, days)):
            if res != "skip":
                print(f"{d} {res}", flush=True)


if __name__ == "__main__":
    main()
