"""短線策略回測（1/3/5/10 日）：隔天開盤進場、第 N 天收盤出場，每筆扣 0.6% 交易成本。

    python -m twscreener.shortterm            # -> results/shortterm_summary.csv、results/shortterm_signals.csv
    python -m twscreener.shortterm today      # 今天的短線候選（附 5/10 日停利停損）-> results/shortterm_today.csv
    python -m twscreener.shortterm check      # 對帳：agent 過去的短線推薦 5/10 日後表現 vs 隨機買股

規則（訊號日 t 收盤後判斷，t+1 開盤買）：
- 超跌反彈：收盤在 200 日均線之上，RSI(2) < 10（兩天內急跌）。
- 連跌反彈：收盤在 200 日均線之上，連跌 3 天且 3 天累計跌超過 7%。
- 爆量長紅：當天漲超過 6%，成交量 > 20 日均量 3 倍。
- 漲停延續：當天漲停（漲幅 >= 9.5%）。
基準：同一天所有流動性夠的股票（隨機買股）。

每晚候選只用「拉回買」兩種（連跌反彈優先、超跌反彈其次）：2024-06～2026-11 回測中，
扣成本後 10 日平均報酬在前後兩段都為正（連跌反彈 +1.4%、+2.8%），追漲的兩種不穩定。
大盤「多頭回檔」時兩種都前後段相反（+4% / -7%），暫停。
停損：近 20 日平均單日波動 × 3（更緊的停損和固定停利在回測裡都降低報酬，所以只防大跌）。
停利參考：同規則、同大盤狀態歷史 5/10 日報酬（以各自波動為單位）的第 75 百分位，大約四次一次做得到。
價量特徵（跌幅、離均線、量比…）在前後段方向不一致，不拿來排序；排序只看規則和成交金額，
真正的篩選交給 agent 查新聞（見 skill/agent-review）。
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


PULLBACK = ["連跌反彈", "超跌反彈"]  # 優先順序
PAUSE_REGIMES = ["多頭回檔"]
STOP_MULT = 3
TOP_CANDIDATES = 15


def history_levels(rules, fwd, ctx, atr):
    """同規則、同大盤狀態（不足 200 筆用整個規則）的 5/10 日勝率、平均；停利用「報酬 ÷ 近 20 日平均波動」的第 75 百分位，
    這樣金融股這種波動小的不會拿到和電子股一樣大的目標。"""
    rows = []
    for name in PULLBACK:
        st = rules[name].stack()
        st = st[st]
        df = st.reset_index()[["date", "stock_id"]]
        for h in (5, 10):
            df[f"r{h}"] = fwd[h].stack().reindex(pd.MultiIndex.from_frame(df[["date", "stock_id"]])).to_numpy()
        df["atr"] = atr.stack().reindex(pd.MultiIndex.from_frame(df[["date", "stock_id"]])).to_numpy()
        df["rule"] = name
        rows.append(df.dropna(subset=["r10"]))
    h = pd.concat(rows).merge(ctx, on="date", how="left")

    def m(g):
        return pd.Series({"樣本數": len(g), "5日勝率": (g["r5"] > 0).mean(), "5日平均": g["r5"].mean(),
                          "5日停利": (g["r5"] / g["atr"]).quantile(0.75), "10日勝率": (g["r10"] > 0).mean(),
                          "10日平均": g["r10"].mean(), "10日停利": (g["r10"] / g["atr"]).quantile(0.75)})
    return h.groupby("rule").apply(m), h.groupby(["rule", "大盤狀態"]).apply(m)


def today():
    from .fundamentals import load_industry
    panel = load_panel()
    raw, adj, o, vol, ret = prepare(panel)
    rules = signals(raw, adj, vol, ret)
    fwd, _ = forward(adj, o)
    ctx = market_context()[["date", "大盤狀態"]]
    d = raw.index[-1]
    regime = ctx.set_index("date")["大盤狀態"].get(d)
    names = panel.drop_duplicates("stock_id", keep="last").set_index("stock_id")["name"]
    ind = load_industry()
    atr_all = ret.abs().rolling(20).mean()
    allp, reg = history_levels(rules, fwd, ctx, atr_all)
    atr = atr_all.loc[d]
    value = (vol * raw).rolling(20).mean().loc[d]
    rows = []
    for k in PULLBACK:
        st = reg.loc[(k, regime)] if (k, regime) in reg.index and reg.loc[(k, regime), "樣本數"] >= 200 else allp.loc[k]
        basis = f"{k}｜{regime}" if st is not allp.loc[k] and (k, regime) in reg.index else f"{k}（全部）"
        for s in rules[k].columns[rules[k].loc[d].to_numpy()]:
            c = raw.loc[d, s]
            rows.append({"date": d.date(), "rule": k, "stock_id": s, "name": names.get(s), "industry": ind.get(s, "其他"),
                         "close": c, "3日跌幅": adj.loc[d, s] / adj[s].shift(3).loc[d] - 1, "近20日平均波動": atr.get(s),
                         "20日均成交額億": value.get(s) * 1000 / 1e8,
                         "停損價": round(c * (1 - STOP_MULT * atr.get(s, 0)), 2),
                         "5日停利價": round(c * (1 + st["5日停利"] * atr.get(s, 0) + COST), 2),
                         "10日停利價": round(c * (1 + st["10日停利"] * atr.get(s, 0) + COST), 2),
                         "5日勝率": st["5日勝率"], "5日平均": st["5日平均"], "10日勝率": st["10日勝率"],
                         "10日平均": st["10日平均"], "樣本數": int(st["樣本數"]), "依據組別": basis, "大盤狀態": regime})
    out = pd.DataFrame(rows)
    if len(out):
        out["_p"] = out["rule"].map({k: i for i, k in enumerate(PULLBACK)})
        out = out.sort_values(["_p", "20日均成交額億"], ascending=[True, False]).drop_duplicates("stock_id")
        out = out.groupby("industry", sort=False).head(2).drop(columns="_p").head(TOP_CANDIDATES)
    if regime in PAUSE_REGIMES:
        out = out.iloc[0:0]
        print(f"大盤「{regime}」：拉回買在這種盤歷史上前後段相反，今天不出短線候選")
    out.to_csv(RESULTS / "shortterm_today.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(f"{d:%Y-%m-%d} 大盤 {regime}，短線候選 {len(out)} 檔")
    if len(out):
        print(out.drop(columns=["date", "大盤狀態", "依據組別"]).round(3).to_string(index=False))
    return out


def check():
    """agent 短線推薦（results/shortterm_picks_log.csv）的 5/10 日結果，和同一天隨機買股比。"""
    path = RESULTS / "shortterm_picks_log.csv"
    if not path.exists():
        print("還沒有短線推薦紀錄")
        return
    log = pd.read_csv(path, dtype={"stock_id": str})
    log = log[log["verdict"] == "推薦"].copy()
    log["date"] = pd.to_datetime(log["signal_date"])
    raw, adj, o, vol, ret = prepare(load_panel())
    fwd, _ = forward(adj, o)
    liquid = signals(raw, adj, vol, ret)["隨機買股"]
    for h in (5, 10):
        log[f"r{h}"] = [fwd[h].at[d, s] if d in fwd[h].index and s in fwd[h].columns else np.nan
                        for d, s in zip(log["date"], log["stock_id"])]
        log[f"隨機{h}"] = log["date"].map(lambda d: fwd[h].loc[d][liquid.loc[d]].mean() if d in liquid.index else np.nan)
    print(log[["signal_date", "stock_id", "name", "rule", "r5", "隨機5", "r10", "隨機10"]].round(3).to_string(index=False))
    print("\n平均：", log[["r5", "隨機5", "r10", "隨機10"]].mean().round(4).to_dict())


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    {"today": today, "check": check}.get(cmd, backtest)()
