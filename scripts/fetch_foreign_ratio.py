"""抓全市場外資持股比例（證交所 MI_QFIIS、櫃買 insti/qfii，一天一檔的批次端點）→ data/foreign_ratio/YYYYMMDD.csv。
欄位：stock_id、issued（發行股數）、foreign（外資及陸資持有股數）、ratio（%）、market。
用途：外資持股單日大減但買賣超對不上的「帳務轉移」（results/手冊未測型態）。已抓過的日子會跳過，可以中斷再跑。
不要和每晚更新（也打證交所）同時跑。
    python scripts/fetch_foreign_ratio.py [起日 YYYY-MM-DD] [迄日]
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from twscreener.twse import _get  # noqa: E402

OUT = ROOT / "data" / "foreign_ratio"
OUT.mkdir(parents=True, exist_ok=True)
days = sorted(p.stem for p in (ROOT / "data" / "daily").glob("*.csv"))
days = [pd.Timestamp(d) for d in days]
a = pd.Timestamp(sys.argv[1]) if len(sys.argv) > 1 else pd.Timestamp("2022-01-01")
b = pd.Timestamp(sys.argv[2]) if len(sys.argv) > 2 else days[-1]


def num(s):
    try:
        return float(str(s).replace(",", "").replace("%", "").strip())
    except ValueError:
        return float("nan")


for d in [x for x in days if a <= x <= b][::-1]:
    f = OUT / f"{d:%Y%m%d}.csv"
    if f.exists():
        continue
    rows = []
    j = _get("twse", f"https://www.twse.com.tw/rwd/zh/fund/MI_QFIIS?date={d:%Y%m%d}&selectType=ALLBUT0999&response=json")
    for r in j.get("data") or []:
        rows.append((str(r[0]).strip(), num(r[3]), num(r[5]), num(r[7]), "上市"))
    j = _get("tpex", f"https://www.tpex.org.tw/www/zh-tw/insti/qfii?date={d:%Y/%m/%d}&response=json")
    for t in j.get("tables") or []:
        for r in t.get("data") or []:
            rows.append((str(r[1]).strip(), num(r[3]), num(r[5]), num(r[7]), "上櫃"))
    t = pd.DataFrame(rows, columns=["stock_id", "issued", "foreign", "ratio", "market"])
    t = t[t["stock_id"].str.fullmatch(r"\d{4}")]
    if len(t) < 500:
        print(d.date(), "資料不足，略過", len(t), flush=True)
        continue
    t.to_csv(f, index=False)
    print(d.date(), len(t), flush=True)
