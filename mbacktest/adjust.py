"""抓除權息資料，讓回測用「還原股價」，避免除息缺口被當成下跌。

用法：
    python -m mbacktest.adjust                    # 補到這個月
    python -m mbacktest.adjust --start 2023-09-01

存成 data/exrights.csv：date, stock_id, prev_close(除權息前收盤), ref(除權息參考價)
每個月上市、上櫃各 1 次請求。
"""
import argparse
import re
import time
from datetime import date

import pandas as pd

from .twse import ROOT, _get, _num

PATH = ROOT / "data" / "exrights.csv"


def _roc(s):
    y, m, d = [int(x) for x in re.findall(r"\d+", s)[:3]]
    return date(y + 1911, m, d).isoformat()


def month_ranges(start, end):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        first = date(y, m, 1)
        nxt = date(y + (m == 12), m % 12 + 1, 1)
        last = min(end, date.fromordinal(nxt.toordinal() - 1))
        yield first, last
        y, m = nxt.year, nxt.month


def fetch(first, last):
    rows = []
    js = _get("twse", "https://www.twse.com.tw/rwd/zh/exRight/TWT49U?response=json"
              f"&startDate={first:%Y%m%d}&endDate={last:%Y%m%d}")
    for r in js.get("data") or []:
        rows.append({"date": _roc(r[0]), "stock_id": r[1].strip(), "prev_close": _num(r[3]), "ref": _num(r[4])})
    js = _get("tpex", "https://www.tpex.org.tw/www/zh-tw/bulletin/exDailyQ?response=json"
              f"&startDate={first:%Y/%m/%d}&endDate={last:%Y/%m/%d}")
    for t in js.get("tables") or []:
        for r in t.get("data") or []:
            rows.append({"date": _roc(r[0]), "stock_id": r[1].strip(), "prev_close": _num(r[3]), "ref": _num(r[4])})
    return rows


def load():
    if not PATH.exists():
        return pd.DataFrame(columns=["date", "stock_id", "prev_close", "ref"])
    df = pd.read_csv(PATH, dtype={"stock_id": str})
    df["date"] = pd.to_datetime(df["date"])
    return df


def adjust_prices(df, ex):
    """把除權息日之前的價格乘上調整係數（後復權的反向：以最新價格為準往前調）。"""
    if ex is None or ex.empty:
        return df
    ex = ex[(ex["prev_close"] > 0) & (ex["ref"] > 0)]
    factor = pd.Series(1.0, index=df.index)
    for d, f in zip(ex["date"], ex["ref"] / ex["prev_close"]):
        factor[df.index < d] *= f
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c] * factor
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=None)
    args = ap.parse_args()
    old = load()
    if args.start:
        start = date.fromisoformat(args.start)
    elif len(old):
        start = old["date"].max().date().replace(day=1)
    else:
        start = date(2023, 9, 1)
    done = set()
    donef = ROOT / "data" / "exrights_months.txt"
    if donef.exists() and not args.start:
        done = set(donef.read_text().split())
    elif donef.exists():
        done = set(donef.read_text().split())
    allx = old
    this_month = date.today().strftime("%Y-%m")
    for first, last in month_ranges(start, date.today()):
        key = first.strftime("%Y-%m")
        if key in done and key != this_month:
            continue
        got = fetch(first, last)
        print(f"{key} {len(got)} 筆", flush=True)
        new = pd.DataFrame(got)
        if len(new):
            new["date"] = pd.to_datetime(new["date"])
            allx = pd.concat([allx, new], ignore_index=True).drop_duplicates(["date", "stock_id"], keep="last")
        # 每個月抓完就存，中斷也不會白抓
        allx.sort_values(["date", "stock_id"]).to_csv(PATH, index=False, date_format="%Y-%m-%d")
        if key != this_month:
            done.add(key)
            donef.write_text("\n".join(sorted(done)))
    print(f"共 {len(allx)} 筆除權息 -> {PATH}")


if __name__ == "__main__":
    main()
