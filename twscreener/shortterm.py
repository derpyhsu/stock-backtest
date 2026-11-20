"""短線策略回測（1/3/5/10 日）：隔天開盤進場、第 N 天收盤出場，每筆扣 0.6% 交易成本。

    python -m twscreener.shortterm            # -> results/shortterm_summary.csv、results/shortterm_signals.csv
    python -m twscreener.shortterm today      # 今天收盤後的短線訊號 -> results/shortterm_today.csv

規則（訊號日 t 收盤後判斷，t+1 開盤買）：
- 超跌反彈：收盤在 200 日均線之上，RSI(2) < 10（兩天內急跌）。
- 連跌反彈：收盤在 200 日均線之上，連跌 3 天且 3 天累計跌超過 7%。
- 爆量長紅：當天漲超過 6%，成交量 > 20 日均量 3 倍。
- 漲停延續：當天漲停（漲幅 >= 9.5%）。
基準：同一天所有流動性夠的股票（隨機買股）。
流動性：20 日均量 >= 500 張、股價 >= 10 元。單日超過 11% 的跳動視為減資、分割，價格已調整。
"""
import sys

import numpy as np
import pandas as pd

from .backtest import COST
from .market import RESULTS, load_panel
from .review import market_context

HORIZONS = [1, 3, 5, 10]


def prepare(panel):
    piv = {k: panel.pivot_table(index="date", columns="stock_id", values=k) for k in ["open", "close", "vol"]}
    raw = piv["close"]
    r = raw.pct_change(fill_method=None)
    jump = r.abs() > 0.11
    r_adj = r.where(~jump, 0.0).fillna(0)
    adj = (1 + r_adj).cumprod().where(raw.notna())
    factor = adj / raw
    o = piv["open"] * factor  # 開盤價用同一個調整係數
    return raw, adj, o, piv["vol"], r.where(~jump)


def rsi2(c):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 2, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 2, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def signals(raw, adj, vol, ret):
    liquid = (vol.rolling(20).mean() >= 500) & (raw >= 10)
    ma200 = adj.rolling(200, min_periods=150).mean()
    above = adj > ma200
    down = ret < 0
    rules = {
        "超跌反彈": above & (rsi2(adj) < 10),
        "連跌反彈": above & down & down.shift(1) & down.shift(2) & (adj / adj.shift(3) - 1 < -0.07),
        "爆量長紅": (ret > 0.06) & (vol > 3 * vol.rolling(20).mean().shift(1)),
        "漲停延續": ret >= 0.095,
        "隨機買股": liquid,
    }
    return {k: (v & liquid).fillna(False) for k, v in rules.items()}


def forward(adj, o):
    """t 日訊號：t+1 開盤買，t+h 收盤賣（h=1 代表當天收盤賣）。"""
    entry = o.shift(-1)
    return {h: adj.shift(-h) / entry - 1 - COST for h in HORIZONS}, entry


def collect(rules, fwd):
    rows = []
    for name, m in rules.items():
        st = m.stack()
        st = st[st]
        if name == "隨機買股" and len(st) > 60000:
            st = st.sample(60000, random_state=0)
        df = st.reset_index()[["date", "stock_id"]]
        df["rule"] = name
        for h in HORIZONS:
            df[f"r{h}"] = fwd[h].stack().reindex(pd.MultiIndex.from_frame(df[["date", "stock_id"]])).to_numpy()
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def stats(g):
    out = {"筆數": len(g)}
    for h in HORIZONS:
        x = g[f"r{h}"].dropna()
        out[f"{h}日勝率"] = (x > 0).mean()
        out[f"{h}日期望值"] = x.mean()
        out[f"{h}日去前5%"] = x[x <= x.quantile(0.95)].mean()
    return pd.Series(out)


def backtest():
    panel = load_panel()
    raw, adj, o, vol, ret = prepare(panel)
    fwd, _ = forward(adj, o)
    sig = collect(signals(raw, adj, vol, ret), fwd)
    ctx = market_context()[["date", "大盤狀態"]]
    sig = sig.merge(ctx, on="date", how="left")
    cut = sig["date"].quantile(0.5)
    sig["期間"] = np.where(sig["date"] <= cut, "前段", "後段")
    sig.to_csv(RESULTS / "shortterm_signals.csv", index=False, encoding="utf-8-sig")
    allp = sig.groupby("rule").apply(stats)
    halves = sig.groupby(["rule", "期間"]).apply(stats)
    regime = sig.groupby(["rule", "大盤狀態"]).apply(stats)
    allp.to_csv(RESULTS / "shortterm_summary.csv", encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(f"期間 {sig['date'].min():%Y-%m-%d} ~ {sig['date'].max():%Y-%m-%d}，前後段以 {cut:%Y-%m-%d} 切開；每筆已扣 {COST:.1%}")
    cols = ["筆數"] + [f"{h}日{k}" for h in HORIZONS for k in ("勝率", "期望值")]
    print(allp[cols].round(4).to_string())
    print("\n前後段：")
    print(halves[[c for c in cols if "期望值" in c or c == "筆數"]].round(4).to_string())
    print("\n大盤狀態：")
    print(regime[[c for c in cols if "期望值" in c or c == "筆數"]].round(4).to_string())
    return sig


def today():
    panel = load_panel()
    raw, adj, o, vol, ret = prepare(panel)
    rules = signals(raw, adj, vol, ret)
    d = raw.index[-1]
    names = panel.drop_duplicates("stock_id", keep="last").set_index("stock_id")["name"]
    rows = [{"date": d.date(), "rule": k, "stock_id": s, "name": names.get(s), "close": raw.loc[d, s]}
            for k, m in rules.items() if k != "隨機買股" for s in m.columns[m.loc[d].to_numpy()]]
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "shortterm_today.csv", index=False, encoding="utf-8-sig")
    print(out.groupby("rule").size() if len(out) else "今天沒有短線訊號")


if __name__ == "__main__":
    today() if (len(sys.argv) > 1 and sys.argv[1] == "today") else backtest()
