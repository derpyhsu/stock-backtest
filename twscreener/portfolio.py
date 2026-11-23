"""模擬帳戶：完全照規則買賣，不經人手。每晚跑一次，也可以從過去某天開始回測整個帳戶。

    python -m twscreener.portfolio daily                  # 每晚：執行今天開盤的委託、檢查出場、排明天的委託
    python -m twscreener.portfolio backtest 2024-12-01    # 同一套規則從某天跑到今天，看淨值曲線與最大回撤

規則（v1，全部寫死，中途不改）：
- 選股：simulate.pick 的無偷看版本（同策略同大盤狀態保守期望值 > 0，依分數取前 5、每產業一檔）。
- 進場：訊號日隔天開盤買；開盤已經低於停損就不買。
- 部位：打到停損最多虧淨值 1%，單檔最多淨值 20%，最多同時 10 檔，現金不夠就少買。
- 出場：收盤跌破停損 → 隔天開盤賣；持有滿 60 個交易日 → 隔天開盤賣。不設固定停利（回測顯示會砍掉大漲股）。
- 成本：買手續費 0.1425%，賣手續費 0.1425% + 證交稅 0.3%。
- 比較：同期間加權指數。
狀態存在 results/paper/：positions.csv、orders.csv、trades.csv、nav.csv。
"""
import math
import sys

import numpy as np
import pandas as pd

from .fundamentals import load_industry
from .market import RESULTS, load_market_index, load_panel
from .review import market_context
from .simulate import pick, prices

PAPER = RESULTS / "paper"
CAPITAL = 1_000_000
RISK = 0.01
MAX_POS = 0.20
MAX_HOLD = 10
HOLD_DAYS = 60
FEE, TAX = 0.001425, 0.003


class Account:
    def __init__(self, cash=CAPITAL):
        self.cash = cash
        self.pos = pd.DataFrame(columns=["stock_id", "name", "strategy", "entry_date", "entry", "shares", "stop", "days"])
        self.orders = pd.DataFrame(columns=["date", "side", "stock_id", "name", "strategy", "shares", "stop", "reason"])
        self.trades, self.nav = [], []

    def value(self, close):
        px = self.pos["stock_id"].map(close).fillna(self.pos["entry"])
        return self.cash + float((px * self.pos["shares"]).sum())

    def execute(self, day, opn):
        """前一晚排的委託在今天開盤成交。"""
        for _, o in self.orders.iterrows():
            p = opn.get(o["stock_id"], np.nan)
            if not np.isfinite(p):
                continue
            if o["side"] == "賣":
                h = self.pos[self.pos["stock_id"] == o["stock_id"]]
                if h.empty:
                    continue
                h = h.iloc[0]
                self.cash += h["shares"] * p * (1 - FEE - TAX)
                self.trades.append({"賣出日": day.date(), "stock_id": h["stock_id"], "name": h["name"], "strategy": h["strategy"],
                                    "買進日": h["entry_date"], "買價": h["entry"], "賣價": p, "股數": h["shares"],
                                    "報酬": p * (1 - FEE - TAX) / (h["entry"] * (1 + FEE)) - 1, "原因": o["reason"]})
                self.pos = self.pos[self.pos["stock_id"] != o["stock_id"]]
            else:
                if p <= o["stop"] or (self.pos["stock_id"] == o["stock_id"]).any():
                    continue  # 開盤就低於停損：不進場
                shares = min(o["shares"], math.floor(self.cash / (p * (1 + FEE))))
                if shares <= 0:
                    continue
                self.cash -= shares * p * (1 + FEE)
                self.pos.loc[len(self.pos) + 1000] = [o["stock_id"], o["name"], o["strategy"], day.date(), p, shares, o["stop"], 0]
        self.pos = self.pos.reset_index(drop=True)
        self.orders = self.orders.iloc[0:0]

    def plan(self, day, close, cands):
        """收盤後：檢查出場、排明天的買進。"""
        self.pos["days"] = self.pos["days"] + 1
        sells = []
        for _, h in self.pos.iterrows():
            c = close.get(h["stock_id"], np.nan)
            if np.isfinite(c) and c <= h["stop"]:
                sells.append((h, "跌破停損"))
            elif h["days"] >= HOLD_DAYS:
                sells.append((h, f"持有滿 {HOLD_DAYS} 天"))
        rows = [{"date": day.date(), "side": "賣", "stock_id": h["stock_id"], "name": h["name"], "strategy": h["strategy"],
                 "shares": h["shares"], "stop": h["stop"], "reason": why} for h, why in sells]
        nav = self.value(close)
        slots = MAX_HOLD - (len(self.pos) - len(sells))
        held = set(self.pos["stock_id"])
        budget = self.cash + sum(h["shares"] * close.get(h["stock_id"], 0) for h, _ in sells)
        for _, c in cands.iterrows():
            if slots <= 0 or c["stock_id"] in held:
                continue
            px, stop = c["close_at_pick"], c["close_at_pick"] * (1 + c["stop"])
            if not (np.isfinite(px) and px > stop > 0):
                continue
            shares = math.floor(min(nav * RISK / (px - stop), nav * MAX_POS / px, budget / (px * 1.01)))
            if shares <= 0:
                continue
            budget -= shares * px
            slots -= 1
            rows.append({"date": day.date(), "side": "買", "stock_id": c["stock_id"], "name": c["name"], "strategy": c["strategy"],
                         "shares": shares, "stop": round(stop, 2), "reason": f"分數 {c['score']:+.1%}"})
        self.orders = pd.DataFrame(rows, columns=self.orders.columns)
        self.nav.append({"date": day.date(), "淨值": nav, "現金": self.cash, "持股數": len(self.pos)})

    def save(self):
        PAPER.mkdir(parents=True, exist_ok=True)
        self.pos.to_csv(PAPER / "positions.csv", index=False, encoding="utf-8-sig")
        self.orders.to_csv(PAPER / "orders.csv", index=False, encoding="utf-8-sig")
        old = PAPER / "trades.csv"
        t = pd.DataFrame(self.trades)
        if old.exists() and len(t) == 0:
            pass
        elif old.exists():
            t = pd.concat([pd.read_csv(old, dtype={"stock_id": str}), t])
        if len(t):
            t.to_csv(old, index=False, encoding="utf-8-sig")
        n = PAPER / "nav.csv"
        nv = pd.DataFrame(self.nav)
        if n.exists():
            nv = pd.concat([pd.read_csv(n), nv]).drop_duplicates("date", keep="last")
        nv.to_csv(n, index=False, encoding="utf-8-sig")
        (PAPER / "cash.txt").write_text(f"{self.cash:.2f}")

    @classmethod
    def load(cls):
        a = cls()
        if (PAPER / "cash.txt").exists():
            a.cash = float((PAPER / "cash.txt").read_text())
            a.pos = pd.read_csv(PAPER / "positions.csv", dtype={"stock_id": str})
            o = PAPER / "orders.csv"
            a.orders = pd.read_csv(o, dtype={"stock_id": str}) if o.exists() and o.stat().st_size > 0 else a.orders
        return a


def _inputs():
    panel = load_panel()
    raw, adj, opn, vol = prices(panel)
    raw_open = panel.pivot_table(index="date", columns="stock_id", values="open")
    sig = pd.read_csv(RESULTS / "market_signals.csv", dtype={"stock_id": str})
    sig["date"] = pd.to_datetime(sig["signal_date"])
    today = RESULTS / "picks_today.csv"
    if today.exists():  # 今天的訊號還沒有隔天開盤價，不在 market_signals.csv 裡，從今日選股補上
        t = pd.read_csv(today, dtype={"stock_id": str})
        if len(t):
            t["date"] = pd.to_datetime(t["date"])
            t = t[~t["date"].isin(sig["date"])]
            sig = pd.concat([sig, t[["date", "stock_id", "name", "strategy"]]], ignore_index=True)
    ctx = market_context()[["date", "大盤狀態"]]
    return raw, adj, opn, vol, raw_open, sig, ctx, load_industry()


def run(days, acct, inputs):
    raw, adj, opn, vol, raw_open, sig, ctx, ind = inputs
    for d in days:
        acct.execute(d, raw_open.loc[d])
        c, _ = pick(d, sig, ctx, raw, adj, vol, ind)
        acct.plan(d, raw.loc[d], c if len(c) else pd.DataFrame(columns=["stock_id"]))
    return acct


def stats(nav, taiex):
    nv = nav.set_index(pd.to_datetime(nav["date"]))["淨值"]
    tx = taiex.reindex(nv.index).ffill()
    dd = nv / nv.cummax() - 1
    m = nv.resample("ME").last().pct_change().dropna()
    return {"期間": f"{nv.index[0]:%Y-%m-%d} ~ {nv.index[-1]:%Y-%m-%d}", "帳戶報酬": f"{nv.iat[-1] / nv.iat[0] - 1:+.1%}",
            "加權指數": f"{tx.iat[-1] / tx.iat[0] - 1:+.1%}", "最大回撤": f"{dd.min():.1%}",
            "賺錢月份": f"{(m > 0).sum()}/{len(m)}", "最差月": f"{m.min():+.1%}" if len(m) else "-"}


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "daily"
    inputs = _inputs()
    raw = inputs[0]
    taiex = load_market_index()["taiex"]
    if cmd == "backtest":
        start = pd.Timestamp(sys.argv[2] if len(sys.argv) > 2 else "2024-12-01")
        days = raw.index[raw.index >= start]
        acct = run(days, Account(), inputs)
        nav = pd.DataFrame(acct.nav)
        tr = pd.DataFrame(acct.trades)
        out = RESULTS / "paper_backtest"
        out.mkdir(parents=True, exist_ok=True)
        nav.to_csv(out / "nav.csv", index=False, encoding="utf-8-sig")
        tr.to_csv(out / "trades.csv", index=False, encoding="utf-8-sig")
        print(stats(nav, taiex))
        if len(tr):
            print(f"交易 {len(tr)} 筆，勝率 {(tr['報酬'] > 0).mean():.0%}，平均 {tr['報酬'].mean():+.1%}，"
                  f"出場原因 {tr['原因'].value_counts().to_dict()}")
        return
    acct = Account.load()
    last = pd.read_csv(PAPER / "nav.csv")["date"].max() if (PAPER / "nav.csv").exists() else None
    days = raw.index[raw.index > pd.Timestamp(last)] if last else raw.index[-1:]
    run(days, acct, inputs)
    acct.save()
    print(stats(pd.read_csv(PAPER / "nav.csv"), taiex))
    print(acct.orders.to_string(index=False) if len(acct.orders) else "明天沒有委託")


if __name__ == "__main__":
    main()
