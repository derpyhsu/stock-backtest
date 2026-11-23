"""月營收公布後續漲（PEAD-TW，規則 B）：模擬帳戶 v2 的選股來源。

營收月份 M 的資料在 M+1 月 11 號（revenue_features 的 available）後才能用。
訊號日 = available 之後第一個交易日；條件（全部要成立）：
- 營收年增 >= 30%
- 年增比前 3 個月平均多 10 個百分點以上（加速）
- 營收創 12 個月新高
- M+1 月 1 號到訊號日，股價比加權指數多漲 5% 以上（市場已認帳）
- 流動性：20 日均量 >= 500 張、股價 >= 10 元
同一天訊號多時，依加速幅度排序。回測見 scripts/revenue_drift_test.py、scripts/pead_portfolio_test.py。
"""
from pathlib import Path

import pandas as pd

from .fundamentals import revenue_features

ROOT = Path(__file__).resolve().parent.parent
YOY, ACCEL, REACT = 0.30, 0.10, 0.05
STOP = -0.15          # 固定停損（2026-11-23 重測：-10%～-25% 每年都贏 v1，-15% 報酬與回撤最平衡）
_cache = {}


def events(raw, adj, vol, taiex):
    """回傳 {訊號日: Series(stock_id -> 加速幅度，已由大到小排序)}。只用訊號日當天以前的資料。"""
    key = (adj.index[-1], len(adj.columns))
    if key in _cache:
        return _cache[key]
    rev = revenue_features(pd.read_csv(ROOT / "data/revenue.csv", dtype={"stock_id": str}))
    rev = rev.sort_values(["stock_id", "ym"])
    rev["yoy_prev3"] = rev.groupby("stock_id")["yoy"].transform(lambda s: s.shift(1).rolling(3).mean())
    liquid = (vol.rolling(20).mean() >= 500) & (raw >= 10)
    taiex = taiex.reindex(adj.index).ffill()
    days = adj.index
    ev = {}
    for _, g in rev.groupby("ym"):
        avail = g["available"].iat[0]
        after = days[days >= avail]
        if len(after) == 0:
            continue
        d = after[0]
        i = days.get_loc(d)
        ms = days[days >= avail - pd.Timedelta(days=10)]
        j = days.get_loc(ms[0]) - 1
        if j < 0:
            continue
        ids = [s for s in g["stock_id"] if s in adj.columns and liquid.at[d, s]]
        gg = g.set_index("stock_id").loc[ids]
        react = adj.iloc[i][ids] / adj.iloc[j][ids] - 1 - (taiex.iat[i] / taiex.iat[j] - 1)
        accel = gg["yoy"] - gg["yoy_prev3"]
        m = (gg["yoy"] >= YOY) & (accel >= ACCEL) & gg["rev_high12"].astype(bool) & (react >= REACT)
        sel = accel[m].sort_values(ascending=False)
        if len(sel):
            ev[d] = sel
    _cache.clear()
    _cache[key] = ev
    return ev


def pick(day, sig, ctx, raw, adj, vol, ind, taiex, names):
    """和 simulate.pick 同樣的輸出格式，給 portfolio.Account.plan 用。"""
    regime = ctx.set_index("date")["大盤狀態"].get(day)
    ev = events(raw, adj, vol, taiex)
    if day not in ev:
        return pd.DataFrame(), regime
    s = ev[day]
    c = pd.DataFrame({"stock_id": s.index, "name": [names.get(x, x) for x in s.index], "strategy": "營收加速",
                      "industry": [ind.get(x, "其他") for x in s.index], "score": s.values, "stop": STOP,
                      "close_at_pick": raw.loc[day, s.index].values})
    return c, regime
