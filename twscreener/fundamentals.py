"""基本面資料：每日估值（本益比、殖利率、股價淨值比）與月營收。

用法：
    python -m twscreener.fundamentals valuation            # 補齊 data/daily 有的每個交易日的估值
    python -m twscreener.fundamentals revenue              # 補月營收（最新月 + 能抓到的歷史月份）
    python -m twscreener.fundamentals revenue --start 2023-01

存檔：
    data/valuation/YYYYMMDD.csv : stock_id, pe, dividend_yield, pb
    data/revenue.csv            : ym(YYYY-MM), stock_id, rev(千元), rev_last_year, rev_last_month
月營收在次月 10 號前公布，回測時要等公布後才能用（見 revenue_available）。
"""
import argparse
import io
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import pandas as pd
import requests

from .twse import DAILY, ROOT, UA, _get, _is_stock, _num

VALUATION = ROOT / "data" / "valuation"
REVENUE = ROOT / "data" / "revenue.csv"
REVENUE_DONE = ROOT / "data" / "revenue_months.txt"


def valuation_day(d):
    ds = d.strftime("%Y%m%d")
    out = VALUATION / f"{ds}.csv"
    if out.exists():
        return "skip"
    rows = []
    js = _get("twse", f"https://www.twse.com.tw/rwd/zh/afterTrading/BWIBBU_d?date={ds}&selectType=ALL&response=json")
    if js.get("stat") == "OK":
        f = js["fields"]
        i_pe, i_y, i_pb = f.index("本益比"), f.index("殖利率(%)"), f.index("股價淨值比")
        for r in js.get("data") or []:
            code = str(r[0]).strip()
            if _is_stock(code):
                rows.append({"stock_id": code, "pe": _num(r[i_pe]), "dividend_yield": _num(r[i_y]), "pb": _num(r[i_pb])})
    js = _get("tpex", f"https://www.tpex.org.tw/www/zh-tw/afterTrading/peQryDate?date={d:%Y/%m/%d}&response=json")
    for t in js.get("tables") or []:
        f = t.get("fields") or []
        if "本益比" not in f:
            continue
        i_pe, i_y, i_pb = f.index("本益比"), f.index("殖利率(%)"), f.index("股價淨值比")
        for r in t.get("data") or []:
            code = str(r[0]).strip()
            if _is_stock(code):
                rows.append({"stock_id": code, "pe": _num(r[i_pe]), "dividend_yield": _num(r[i_y]), "pb": _num(r[i_pb])})
    if not rows:
        return "沒有資料"
    pd.DataFrame(rows).to_csv(out, index=False)
    return f"{len(rows)} 檔"


def _safe(d):
    try:
        return valuation_day(d)
    except Exception as e:
        return f"失敗，下次再補（{e.__class__.__name__}）"


def update_valuation(workers=3):
    VALUATION.mkdir(parents=True, exist_ok=True)
    days = sorted((date(int(p.stem[:4]), int(p.stem[4:6]), int(p.stem[6:])) for p in DAILY.glob("*.csv")), reverse=True)
    days = [d for d in days if not (VALUATION / f"{d:%Y%m%d}.csv").exists()]
    print(f"估值：要補 {len(days)} 天", flush=True)
    with ThreadPoolExecutor(workers) as ex:
        for d, res in zip(days, ex.map(_safe, days)):
            print(f"{d} {res}", flush=True)


# ---------- 月營收 ----------

def _rev_mops(y, m, market):
    """公開資訊觀測站的月營收彙總表（含歷史）。market: sii=上市, otc=上櫃"""
    url = f"https://mopsov.twse.com.tw/nas/t21/{market}/t21sc03_{y - 1911}_{m}_0.html"
    r = requests.get(url, headers=UA, timeout=90)
    r.encoding = "big5"
    if r.status_code != 200 or "營業收入" not in r.text:
        raise RuntimeError(f"MOPS 無法取得 {y}-{m:02d} {market}（HTTP {r.status_code}）")
    rows = []
    for t in pd.read_html(io.StringIO(r.text)):
        if t.shape[1] < 5:
            continue
        for _, row in t.iterrows():
            code = str(row.iloc[0]).strip()
            if _is_stock(code):
                rows.append({"stock_id": code, "rev": _num(row.iloc[2]), "rev_last_month": _num(row.iloc[3]),
                             "rev_last_year": _num(row.iloc[4])})
    return rows


def _rev_openapi():
    """證交所、櫃買 OpenAPI：只有最新一個月，但不受 MOPS 封鎖影響。"""
    rows = []
    for url in ["https://openapi.twse.com.tw/v1/opendata/t187ap05_L",
                "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap05_O"]:
        for r in requests.get(url, headers=UA, timeout=90).json():
            code = str(r.get("公司代號", "")).strip()
            ym = str(r.get("資料年月", "")).strip()
            if not _is_stock(code) or len(ym) < 4:
                continue
            rows.append({"ym": f"{int(ym[:-2]) + 1911}-{ym[-2:]}", "stock_id": code,
                         "rev": _num(r.get("營業收入-當月營收")), "rev_last_month": _num(r.get("營業收入-上月營收")),
                         "rev_last_year": _num(r.get("營業收入-去年當月營收"))})
    return rows


def update_revenue_finmind(start="2022-01-01"):
    """觀測站擋雲端 IP 時的備案：FinMind 逐檔抓月營收歷史（免費額度約每小時 300～600 次，可中斷續抓）。"""
    import os
    import time
    done_f = ROOT / "data" / "revenue_finmind_done.txt"
    done = set(done_f.read_text().split()) if done_f.exists() else set()
    ids = sorted({p for f in sorted(DAILY.glob("*.csv"))[-5:] for p in pd.read_csv(f, dtype={"stock_id": str})["stock_id"]})
    ids = [s for s in ids if s not in done]
    print(f"FinMind 月營收：還有 {len(ids)} 檔", flush=True)
    token = os.environ.get("FINMIND_TOKEN", "")
    allr = load_revenue()
    buf = []
    for i, sid in enumerate(ids):
        while True:
            r = requests.get("https://api.finmindtrade.com/api/v4/data", timeout=60,
                             params={"dataset": "TaiwanStockMonthRevenue", "data_id": sid, "start_date": start, "token": token})
            js = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
            if js.get("status") == 200:
                break
            print(f"  ! FinMind {r.status_code} {str(js.get('msg', ''))[:40]}，等 10 分鐘", flush=True)
            time.sleep(600)
        d = pd.DataFrame(js.get("data") or [])
        if len(d):
            d["ym"] = d["revenue_year"].astype(int).astype(str) + "-" + d["revenue_month"].astype(int).map("{:02d}".format)
            d = d.sort_values("ym")
            ly = d.set_index("ym")["revenue"]
            d["rev_last_year"] = [ly.get(f"{int(y) - 1}-{int(m):02d}") for y, m in zip(d["revenue_year"], d["revenue_month"])]
            d["rev_last_month"] = d["revenue"].shift(1)
            buf.append(pd.DataFrame({"ym": d["ym"], "stock_id": sid, "rev": d["revenue"] / 1000,
                                     "rev_last_month": d["rev_last_month"] / 1000, "rev_last_year": d["rev_last_year"] / 1000}))
        done.add(sid)
        if len(buf) >= 50 or i == len(ids) - 1:
            # OpenAPI/觀測站的資料優先（keep="last" 時放後面）
            allr = pd.concat([pd.concat(buf), allr], ignore_index=True) if buf else allr
            _save(allr)
            done_f.write_text("\n".join(sorted(done)))
            buf = []
            print(f"{i + 1}/{len(ids)}", flush=True)
        time.sleep(6)


def load_revenue():
    if not REVENUE.exists():
        return pd.DataFrame(columns=["ym", "stock_id", "rev", "rev_last_month", "rev_last_year"])
    return pd.read_csv(REVENUE, dtype={"stock_id": str})


def _save(df):
    df.drop_duplicates(["ym", "stock_id"], keep="last").sort_values(["ym", "stock_id"]).to_csv(REVENUE, index=False)


def update_revenue(start="2023-01", rounds=1, wait=900):
    """觀測站時好時壞（會回 307 安全頁），所以每個「月份×市場」分開記錄，
    rounds > 1 時每輪之間等 wait 秒再重試還沒抓到的。"""
    import time
    allr = load_revenue()
    try:
        allr = pd.concat([allr, pd.DataFrame(_rev_openapi())], ignore_index=True)
        _save(allr)
        print("最新月營收（OpenAPI）OK", flush=True)
    except Exception as e:
        print(f"OpenAPI 失敗：{e}", file=sys.stderr)
    done = set(REVENUE_DONE.read_text().split()) if REVENUE_DONE.exists() else set()
    # 舊格式（只有月份）代表上市、上櫃都抓到了
    done |= {f"{k}:{mk}" for k in list(done) if ":" not in k for mk in ("sii", "otc")}
    todo = []
    y, m = int(start[:4]), int(start[5:7])
    today = date.today()
    while (y, m) < (today.year, today.month):
        todo += [(y, m, mk) for mk in ("sii", "otc") if f"{y}-{m:02d}:{mk}" not in done]
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    for rnd in range(rounds):
        if not todo:
            break
        if rnd:
            time.sleep(wait)
        left = []
        for y, m, mk in todo:
            key = f"{y}-{m:02d}"
            try:
                rows = [dict(r, ym=key) for r in _rev_mops(y, m, mk)]
                allr = pd.concat([allr, pd.DataFrame(rows)], ignore_index=True)
                _save(allr)
                done.add(f"{key}:{mk}")
                REVENUE_DONE.write_text("\n".join(sorted(done)))
                print(f"{key} {mk} {len(rows)} 筆", flush=True)
            except Exception:
                left.append((y, m, mk))
            time.sleep(5)
        todo = left
        print(f"第 {rnd + 1} 輪結束，還差 {len(todo)} 個（月份×市場）", flush=True)


def revenue_features(rev):
    """每檔每月：營收年增率、近 3 月累計年增率、是否創 12 個月新高。"""
    r = rev.copy()
    r = r.sort_values(["stock_id", "ym"])
    r["yoy"] = r["rev"] / r["rev_last_year"] - 1
    g = r.groupby("stock_id")
    r["rev3"] = g["rev"].transform(lambda s: s.rolling(3).sum())
    r["rev3_ly"] = g["rev_last_year"].transform(lambda s: s.rolling(3).sum())
    r["yoy3"] = r["rev3"] / r["rev3_ly"] - 1
    r["rev_high12"] = r["rev"] >= g["rev"].transform(lambda s: s.rolling(12, min_periods=12).max())
    # 公布日：次月 10 號（之後才能拿來用）
    p = pd.to_datetime(r["ym"] + "-01") + pd.DateOffset(months=1)
    r["available"] = p + pd.Timedelta(days=10)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["valuation", "revenue", "revenue_finmind", "all"])
    ap.add_argument("--start", default="2023-01")
    ap.add_argument("--rounds", type=int, default=1, help="月營收：觀測站擋的時候重試幾輪（每輪間隔 15 分鐘）")
    args = ap.parse_args()
    if args.what in ("revenue", "all"):
        update_revenue(args.start, args.rounds)
    if args.what == "revenue_finmind":
        update_revenue_finmind()
    if args.what in ("valuation", "all"):
        update_valuation()


if __name__ == "__main__":
    main()
