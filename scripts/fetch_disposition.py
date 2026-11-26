"""抓證交所、櫃買的處置股公告（批次端點，一年一次查詢）→ data/disposition.csv。
欄位：announce（公布日）、stock_id、name、count（累計次數）、start、end（處置期間）、measure（第一次／再次…）、market。
    python scripts/fetch_disposition.py            # 2022 起全部重抓
    python scripts/fetch_disposition.py --update   # 每晚：只抓今年，和舊檔合併（兩次查詢）
"""
import re
import sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from twscreener.twse import _get  # noqa: E402


def roc(s):
    n = re.findall(r"\d+", s)
    if len(n) == 1 and len(n[0]) >= 7:      # 1111201 這種連在一起的
        n = [n[0][:-4], n[0][-4:-2], n[0][-2:]]
    return pd.Timestamp(int(n[0]) + 1911, int(n[1]), int(n[2]))


def js(x):
    return x if isinstance(x, dict) else x.json()


rows = []
this = pd.Timestamp.now().year
OUT = ROOT / "data/disposition.csv"
UPDATE = "--update" in sys.argv and OUT.exists()
for y in range(this if UPDATE else 2022, this + 1):
    a, b = f"{y}0101", f"{y}1231"
    j = js(_get("twse", f"https://www.twse.com.tw/rwd/zh/announcement/punish?startDate={a}&endDate={b}&response=json"))
    for r in j.get("data") or []:
        se = re.split(r"[～~]", r[6])
        rows.append({"announce": roc(r[1]), "stock_id": r[2].strip(), "name": r[3].strip(), "count": r[4], "start": roc(se[0]), "end": roc(se[1]),
                     "measure": r[7].strip(), "market": "上市"})
    j = js(_get("tpex", f"https://www.tpex.org.tw/www/zh-tw/bulletin/disposal?startDate={y}/01/01&endDate={y}/12/31&response=json"))
    for t in j.get("tables") or []:
        for r in t.get("data") or []:
            se = re.split(r"[～~\-]", str(r[5]))
            if len(se) < 2:
                print("略過", r[:6])
                continue
            rows.append({"announce": roc(r[1]), "stock_id": str(r[2]).strip(), "name": re.sub(r"\(.*", "", str(r[3])).strip(), "count": r[4],
                         "start": roc(se[0]), "end": roc(se[1]), "measure": str(r[7]).strip(), "market": "上櫃"})
    print(y, len(rows), flush=True)
d = pd.DataFrame(rows)
if UPDATE:
    old = pd.read_csv(OUT, dtype={"stock_id": str}, parse_dates=["announce", "start", "end"])
    d = pd.concat([old[old["announce"].dt.year < this], d])
d = d.drop_duplicates(["announce", "stock_id", "start"]).sort_values(["announce", "stock_id"])
d.to_csv(OUT, index=False, encoding="utf-8-sig")
print(len(d), d["announce"].min(), d["announce"].max(), d[d["stock_id"].str.fullmatch(r"\d{4}")].groupby("market").size().to_dict())
