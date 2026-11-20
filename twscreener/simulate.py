"""假裝回到過去某一天：只用那天以前知道的資訊選股，再看之後的實際表現。

    python -m twscreener.simulate 2026-09-15          # 單日：推薦哪幾檔、到最新一天漲跌多少
    python -m twscreener.simulate 2026-09-01 2026-09-30   # 區間內每個交易日都跑一次，彙總

做法（和每晚的規則選股相同，但不含 agent 查新聞，因為搜尋不到「當時」的新聞）：
1. 那天各策略的訊號（規則只看當天以前的價量，本來就沒有偷看未來）。
2. 策略分數只用「那天之前 60 日結果已經出來」的歷史訊號計算：同策略、同大盤狀態的保守期望值
   （去掉最好 5%，扣成本）。保守期望值 <= 0 的策略整批不選。
3. 依分數由高到低，每個產業最多一檔，選 TOP 檔。
4. 隔天開盤買進；停損 = 那天以前的歷史停損與近 20 日波動 × 3 取較寬者，收盤跌破就隔天出場（以收盤價近似）。
5. 和同一天「隨機買股」（所有流動性夠的股票平均）、加權指數比較。
"""
import sys

import numpy as np
import pandas as pd

from .backtest import COST
from .fundamentals import load_industry
from .levels import VOL_MULT
from .market import RESULTS, load_market_index, load_panel
from .review import market_context

TOP = 5


def prices(panel):
    piv = {k: panel.pivot_table(index="date", columns="stock_id", values=k) for k in ["open", "close", "vol"]}
    raw = piv["close"]
    r = raw.pct_change(fill_method=None)
    r = r.where(r.abs() <= 0.11, 0.0).fillna(0)
    adj = (1 + r).cumprod().where(raw.notna())
    return raw, adj, piv["open"] * adj / raw, piv["vol"]


def score_table(sig, ctx, day):
    """只用 day 之前已經知道 60 日結果的訊號（訊號日 + 約 90 個日曆天 < day）。"""
    known = sig[(sig["date"] + pd.Timedelta(days=90) < day)].dropna(subset=["ret_60d"]).merge(ctx, on="date", how="left")
    def m(g):
        r = g["ret_60d"] - COST
        win = g[g["ret_60d"] > 0]["max_dd_60d"].dropna()
        return pd.Series({"n": len(r), "cons": r[r <= r.quantile(0.95)].mean(), "win": (r > 0).mean(),
                          "stop": win.quantile(0.2) if len(win) >= 10 else -0.1})
    return known.groupby("strategy").apply(m), known.groupby(["strategy", "大盤狀態"]).apply(m)


def pick(day, sig, ctx, raw, adj, vol, ind):
    regime = ctx.set_index("date")["大盤狀態"].get(day)
    allp, reg = score_table(sig, ctx, day)
    today = sig[sig["date"] == day].copy()
    rows = []
    for _, s in today.iterrows():
        st = reg.loc[(s["strategy"], regime)] if (s["strategy"], regime) in reg.index else None
        if st is None or st["n"] < 30:
            st = allp.loc[s["strategy"]] if s["strategy"] in allp.index else None
        if st is None or not st["cons"] > 0:
            continue
        rows.append({"stock_id": s["stock_id"], "name": s["name"], "strategy": s["strategy"].split("_", 1)[-1],
                     "industry": ind.get(s["stock_id"], "其他"), "score": st["cons"], "hist_win": st["win"],
                     "hist_stop": st["stop"]})
    if not rows:
        return pd.DataFrame(), regime
    c = pd.DataFrame(rows).sort_values("score", ascending=False).drop_duplicates("stock_id")
    c = c.groupby("industry").head(1).head(TOP)
    i = adj.index.get_loc(day)
    v = adj.pct_change(fill_method=None).abs().iloc[max(0, i - 19):i + 1].mean()
    c["stop"] = np.minimum(c["hist_stop"], -VOL_MULT * c["stock_id"].map(v).fillna(0))
    c["close_at_pick"] = c["stock_id"].map(raw.loc[day])
    return c, regime


def outcome(c, day, adj, opn, end=None):
    i = adj.index.get_loc(day)
    end_i = len(adj) - 1 if end is None else adj.index.get_loc(end)
    if i + 1 > end_i:
        return c
    res = []
    for sid, stop in zip(c["stock_id"], c["stop"]):
        entry = opn[sid].iat[i + 1]
        path = adj[sid].iloc[i + 1:end_i + 1] / entry - 1
        hold = path.iat[-1] - COST
        hit = path[path <= stop]
        with_stop = (hit.iat[0] - COST) if len(hit) else hold
        r20 = path.iat[19] - COST if len(path) > 19 else np.nan
        r60 = path.iat[59] - COST if len(path) > 59 else np.nan
        res.append({"20日報酬": r20, "60日報酬": r60, "抱到最後": hold, "有停損": with_stop,
                    "停損日": hit.index[0].date() if len(hit) else None, "期間最高": path.max()})
    return pd.concat([c.reset_index(drop=True), pd.DataFrame(res)], axis=1)


def baseline(day, adj, opn, raw, vol):
    i = adj.index.get_loc(day)
    liquid = (vol.rolling(20).mean().loc[day] >= 500) & (raw.loc[day] >= 10)
    ids = liquid[liquid].index
    entry = opn.iloc[i + 1][ids]
    r = adj.iloc[-1][ids] / entry - 1 - COST
    r20 = adj.iloc[min(i + 20, len(adj) - 1)][ids] / entry - 1 - COST
    r60 = adj.iloc[i + 60][ids] / entry - 1 - COST if i + 60 < len(adj) else pd.Series(np.nan, index=ids)
    return r.mean(), r20.mean(), r60.mean()


def run(days):
    panel = load_panel()
    raw, adj, opn, vol = prices(panel)
    sig = pd.read_csv(RESULTS / "market_signals.csv", dtype={"stock_id": str})
    sig["date"] = pd.to_datetime(sig["signal_date"])
    ctx = market_context()[["date", "大盤狀態"]]
    taiex = load_market_index()["taiex"]
    ind = load_industry()
    out = []
    for d in days:
        d = pd.Timestamp(d)
        if d not in adj.index or adj.index.get_loc(d) + 1 >= len(adj):
            continue
        c, regime = pick(d, sig, ctx, raw, adj, vol, ind)
        if c.empty:
            continue
        c = outcome(c, d, adj, opn)
        b_end, b20, b60 = baseline(d, adj, opn, raw, vol)
        nxt = adj.index[adj.index.get_loc(d) + 1]
        t_end = taiex.iloc[-1] / taiex.loc[nxt] - 1
        c["date"], c["regime"], c["隨機買股_抱到最後"], c["隨機買股_20日"], c["加權指數_抱到最後"] = d.date(), regime, b_end, b20, t_end
        c["隨機買股_60日"] = b60
        out.append(c)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main():
    args = sys.argv[1:] or ["2026-09-15"]
    if len(args) == 1:
        res = run([args[0]])
    else:
        idx = load_market_index().index
        res = run(idx[(idx >= args[0]) & (idx <= args[1])])
    tag = "_".join(args)
    res.to_csv(RESULTS / f"simulate_{tag}.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    cols = ["date", "regime", "stock_id", "name", "strategy", "industry", "close_at_pick", "stop", "20日報酬", "抱到最後",
            "有停損", "停損日", "期間最高"]
    if len(args) == 1:
        print(res[cols].round(3).to_string(index=False))
        print(f"\n推薦平均：20日 {res['20日報酬'].mean():+.1%}，抱到最後 {res['抱到最後'].mean():+.1%}，有停損 {res['有停損'].mean():+.1%}")
        print(f"隨機買股：20日 {res['隨機買股_20日'].iat[0]:+.1%}，抱到最後 {res['隨機買股_抱到最後'].iat[0]:+.1%}；加權指數 {res['加權指數_抱到最後'].iat[0]:+.1%}")
    else:
        g = res.groupby("date").agg(n=("stock_id", "size"), 推薦20日=("20日報酬", "mean"), 推薦60日=("60日報酬", "mean"), 隨機60日=("隨機買股_60日", "first"), 推薦抱到最後=("抱到最後", "mean"),
                                     推薦有停損=("有停損", "mean"), 隨機20日=("隨機買股_20日", "first"),
                                     隨機抱到最後=("隨機買股_抱到最後", "first"), 加權指數=("加權指數_抱到最後", "first"))
        print(g.round(3).to_string())
        print("\n平均：", g.mean(numeric_only=True).round(3).to_dict())
        print("推薦 20 日贏隨機的天數：", int((g["推薦20日"] > g["隨機20日"]).sum()), "/", len(g))
        print("推薦 60 日贏隨機的天數：", int((g["推薦60日"] > g["隨機60日"]).sum()), "/", int(g["推薦60日"].notna().sum()))
        print("策略分布：", res["strategy"].value_counts().to_dict())


if __name__ == "__main__":
    main()
