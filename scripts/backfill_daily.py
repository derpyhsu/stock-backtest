"""把 2021 年的全市場日線補到 data/backfill_daily/（不直接放進 data/daily/）。

為什麼分開放：帳戶回測要一年暖身，現在只能從 2023-04 起算，2022 年空頭沒測到；補上 2021 就能從 2022 年初開始。
但補到一半就併進 data/daily/ 會讓每晚的回測吃到沒算還原價的半套資料，所以整年補齊前先放旁邊。
證交所限流很嚴（每小時不到 20 天），所以排在每晚例行更新最後，用 timeout 限制每晚跑多久，可中斷續抓：
    timeout 5400 python3 scripts/backfill_daily.py
印出「2021 已補齊」之後的步驟（人工做一次）：
    mv data/backfill_daily/*.csv data/daily/ ; mv data/backfill_daily_index/*.csv data/daily_index/
    python3 -m twscreener.adjust --start 2021-01-01
    再重跑帳戶回測，看 2022 年的表現。
"""
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from twscreener import twse  # noqa: E402

START, END = date(2021, 1, 1), date(2021, 12, 31)
twse.DAILY = ROOT / "data" / "backfill_daily"
twse.DAILY_INDEX = ROOT / "data" / "backfill_daily_index"


def missing():
    days = [START + timedelta(days=i) for i in range((END - START).days + 1)]
    return [d for d in days if d.weekday() < 5 and not any((twse.DAILY / f"{d:%Y%m%d}.{e}").exists() for e in ("csv", "none"))]


if __name__ == "__main__":
    twse.DAILY.mkdir(parents=True, exist_ok=True)
    if not missing():
        print("2021 已補齊")
        sys.exit(0)
    sys.argv = [sys.argv[0], "--start", START.isoformat(), "--end", END.isoformat(), "--workers", "2"]
    twse.main()
    print("2021 已補齊" if not missing() else f"還差 {len(missing())} 天")
