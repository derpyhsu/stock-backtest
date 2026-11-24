"""加碼規則回測（v2 營收加速帳戶上加碼，和 v2 原版比）。

A 再出訊號加碼：手上的股票又出營收加速訊號 → 隔天開盤加碼（單檔上限仍是淨值 20%），停損取新舊較高者；可選擇持有天數重新起算。
B 賺到 X% 加碼：持股收盤報酬 >= X → 隔天開盤加原本股數的一半，整筆停損上移到原始買價（每檔只加一次）。
攤平（跌了加碼）不測：和停損紀律衝突。
    python scripts/pyramid_test.py 2022-08-01
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import twscreener.portfolio as P  # noqa: E402
from twscreener.market import load_market_index  # noqa: E402


class Pyramid(P.Account):
    def __init__(self, mode, x=0.2, reset=False):
        super().__init__()
        self.mode, self.x, self.reset = mode, x, reset
        self.added = set()
        self.adds = []

    def plan(self, day, close, cands):
        super().plan(day, close, cands)
        nav = self.value(close)
        budget = self.cash - float((self.orders.query("side == '買'")["shares"] * self.orders.query("side == '買'")["stock_id"].map(close)).sum())
        selling = set(self.orders.query("side == '賣'")["stock_id"])
        self.added &= set(self.pos["stock_id"])
        rows = []
        held = self.pos[~self.pos["stock_id"].isin(selling)]
        if self.mode == "A" and len(cands):
            for _, c in cands[cands["stock_id"].isin(held["stock_id"])].iterrows():
                h = held[held["stock_id"] == c["stock_id"]].iloc[0]
                px = c["close_at_pick"]
                new_stop = px * (1 + c["stop"])
                room = nav * P.MAX_POS / px - h["shares"]
                n = math.floor(min(nav * P.RISK / (px - new_stop), room, budget / (px * 1.01)))
                if n > 0:
                    budget -= n * px
                    rows.append({"stock_id": h["stock_id"], "shares": n, "stop": max(h["stop"], round(new_stop, 2))})
        if self.mode == "B":
            for _, h in held.iterrows():
                c = close.get(h["stock_id"], np.nan)
                if h["stock_id"] in self.added or not np.isfinite(c) or c / h["entry"] - 1 < self.x:
                    continue
                n = math.floor(min(h["shares"] * 0.5, nav * P.MAX_POS / c - h["shares"], budget / (c * 1.01)))
                if n > 0:
                    budget -= n * c
                    rows.append({"stock_id": h["stock_id"], "shares": n, "stop": max(h["stop"], h["entry"])})
        self.adds = rows

    def execute(self, day, opn, prev=None):
        adds, self.adds = self.adds, []
        super().execute(day, opn, prev)
        for a in adds:
            p = opn.get(a["stock_id"], np.nan)
            pc = prev.get(a["stock_id"], np.nan) if prev is not None else np.nan
            idx = self.pos.index[self.pos["stock_id"] == a["stock_id"]]
            if not np.isfinite(p) or len(idx) == 0 or (np.isfinite(pc) and p >= pc * (1 + P.LIMIT)) or p <= a["stop"]:
                continue
            n = min(a["shares"], math.floor(self.cash / (p * (1 + P.FEE))))
            if n <= 0:
                continue
            i = idx[0]
            old_n, old_e = self.pos.at[i, "shares"], self.pos.at[i, "entry"]
            self.cash -= n * p * (1 + P.FEE)
            self.pos.at[i, "entry"] = (old_n * old_e + n * p) / (old_n + n)
            self.pos.at[i, "shares"] = old_n + n
            self.pos.at[i, "stop"] = a["stop"]
            if self.reset:
                self.pos.at[i, "days"] = 0
            self.added.add(a["stock_id"])


start = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else "2022-08-01")
inputs = P._inputs()
days = inputs[0].index[inputs[0].index >= start]
V = {"v2 原版": P.Account(),
     "A 再出訊號加碼": Pyramid("A"), "A 再出訊號加碼＋天數重算": Pyramid("A", reset=True),
     "B 賺10%加碼": Pyramid("B", 0.10), "B 賺20%加碼": Pyramid("B", 0.20), "B 賺30%加碼": Pyramid("B", 0.30)}
navs, stats = {}, {}
for k, acct in V.items():
    a = P.run(days, acct, inputs, version="v2")
    n = pd.DataFrame(a.nav)
    navs[k] = n.set_index(pd.to_datetime(n["date"]))["淨值"]
    stats[k] = (len(a.trades), getattr(a, "added", set()))
navs = pd.DataFrame(navs)
navs["加權指數"] = load_market_index()["taiex"].reindex(navs.index)
out = Path(__file__).resolve().parent.parent / "results/加碼測試"
out.mkdir(parents=True, exist_ok=True)
navs.to_csv(out / "navs.csv")
seg = navs[navs.index >= "2023-04-01"]


def summ(s):
    s = s.dropna() / s.dropna().iat[0]
    return f"{s.iat[-1] - 1:+.1%}／{(s / s.cummax() - 1).min():.1%}"


h = seg.index[len(seg) // 2]
print(f"期間 {seg.index[0].date()} ~ {seg.index[-1].date()}（報酬／最大回撤）")
for k in seg.columns:
    yr = "  ".join(f"{y} {summ(g[k])}" for y, g in seg.groupby(seg.index.year))
    print(f"{k:16s} 全期 {summ(seg[k])}  前段 {summ(seg[k][seg.index < h])}  後段 {summ(seg[k][seg.index >= h])}  {yr}")
