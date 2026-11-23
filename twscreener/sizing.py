"""依停損算部位大小（參考 tradermonty/claude-trading-skills 的 position-sizer，MIT）。

固定比例風險：每筆交易跌到停損時，最多虧總資金的 RISK（預設 1%）。
股數 = 資金 × RISK ÷（進場價 − 停損價），再限制單檔最多佔資金 MAX_POSITION。
台股一張 1000 股；不足一張就用零股。

    python -m twscreener.sizing 1000000 434.5 344     # 資金、進場價、停損價
"""
import math
import sys

CAPITAL = 1_000_000   # 推播用「每 100 萬」為單位，使用者自己乘上倍數
RISK = 0.01
MAX_POSITION = 0.20


def size(entry, stop, capital=CAPITAL, risk=RISK, max_position=MAX_POSITION):
    """回傳 dict：shares（股）、lots（整張）、odd（零股）、amount（金額）、loss（打到停損的虧損）、capped（是否被單檔上限卡住）。"""
    if not (entry > 0 and 0 < stop < entry):
        return None
    by_risk = capital * risk / (entry - stop)
    by_cap = capital * max_position / entry
    shares = math.floor(min(by_risk, by_cap))
    return {"shares": shares, "lots": shares // 1000, "odd": shares % 1000, "amount": shares * entry,
            "loss": shares * (entry - stop), "capped": by_cap < by_risk}


def describe(entry, stop, capital=CAPITAL):
    """推播用的一行：每 100 萬資金買多少。"""
    s = size(entry, stop, capital)
    if not s or s["shares"] == 0:
        return None
    qty = f"{s['lots']} 張" if s["lots"] >= 1 else f"零股 {s['shares']} 股"
    if s["lots"] >= 1 and s["odd"] >= 100:
        qty += f"又 {s['odd']} 股"
    note = "（已達單檔 20% 上限）" if s["capped"] else ""
    return f"💰 每 100 萬：{qty}，約 {s['amount'] / 1e4:.0f} 萬{note}；打到停損虧 {s['loss'] / 1e4:.1f} 萬"


if __name__ == "__main__":
    cap, entry, stop = map(float, sys.argv[1:4])
    print(size(entry, stop, cap))
    print(describe(entry, stop, cap))
