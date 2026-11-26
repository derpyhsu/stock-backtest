"""抓集保「股權分散表」最新一週（全市場一檔），存成 data/tdcc/YYYYMMDD.csv（2026-11-26）。

集保開放資料只給最新一週，沒有歷史，所以要每週累積；資料日是週五，隔週一前後更新。
同一個資料日已經存過就不重寫，所以每天跑也沒關係。
持股分級：1＝1~999 股 … 11＝200~400 張、12＝400~600 張、13＝600~800 張、14＝800~1,000 張、
15＝1,000 張以上、16＝差異數調整、17＝合計。常說的「400 張大戶」＝12~15 級。
    python scripts/fetch_tdcc.py
"""
import io
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
URL = "https://opendata.tdcc.com.tw/getOD.ashx?id=1-5"
OUT = ROOT / "data" / "tdcc"


def main():
    r = requests.get(URL, timeout=120, headers={"User-Agent": "curl/8"}, allow_redirects=False)
    r.raise_for_status()
    t = pd.read_csv(io.BytesIO(r.content), dtype=str, encoding="utf-8-sig")
    t.columns = ["date", "stock_id", "level", "holders", "shares", "pct"]
    t["stock_id"] = t["stock_id"].str.strip()
    t = t[t["stock_id"].str.fullmatch(r"\d{4}")]
    if t.empty:
        sys.exit("集保回傳沒有資料")
    day = t["date"].iat[0]
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"{day}.csv"
    if f.exists():
        print(f"{day} 已經有了，略過")
        return
    t.drop(columns="date").to_csv(f, index=False)
    print(f"{day}：{t['stock_id'].nunique()} 檔 → {f.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
