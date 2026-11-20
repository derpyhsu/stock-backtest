"""恐慌誤殺：大盤急跌時，找出「跌很深、但營收還在成長、估值變便宜」的股票。

    python -m twscreener.oversold            # 今天的候選 -> results/oversold_today.csv
    python -m twscreener.oversold backtest   # 過去大盤急跌時，誤殺候選 vs 其他跌深股 -> results/oversold_backtest.csv

規則（固定、可回測）：
- 跌深：收盤比近 60 日最高點跌 DROP 以上（單日超過 11% 的跳動視為減資、分割，不算跌幅），且流動性夠（20 日均量 >= 500 張、股價 >= 10）。
- 營收還在成長：當天已公布的最新月營收，近 3 月累計年增 > 10%，單月年增 > 0。
- 估值變便宜：本益比低於自己過去一年的中位數（本益比要 > 0）。
- 法人有沒有在接（外資＋投信近 10 日買超 > 0）只列出來參考，不當條件。
只用訊號當天之前就知道的資訊（月營收用公布日 available）。
"""
import sys

import numpy as np
import pandas as pd

from .features import ROOT
from .fundamentals import revenue_features
from .market import RESULTS, load_market_index, load_panel

DROP = -0.25
YOY3_STRONG = 0.48  # 2026-11 回測時誤殺候選裡營收成長前 1/3 的門檻（樣本內選的，要再驗證）
VAL = ROOT / "data" / "valuation"


def load_valuation():
    files = sorted(VAL.glob("*.csv"))
    v = pd.concat((pd.read_csv(f, dtype={"stock_id": str}).assign(date=pd.Timestamp(f.stem)) for f in files),
                  ignore_index=True)
    pe = v.pivot_table(index="date", columns="stock_id", values="pe")
    pe = pe.where(pe > 0)
    return pe, pe.rolling(250, min_periods=60).median()


def build(panel):
    p = {k: panel.pivot_table(index="date", columns="stock_id", values=k)
         for k in ["close", "vol", "foreign_net", "trust_net"]}
    # 台股單日漲跌幅上限 10%，超過 11% 的跳動當作減資、分割、除權等公司行為，不算漲跌
    raw = p["close"]
    ret = raw.pct_change(fill_method=None)
    ret = ret.where(ret.abs() <= 0.11, 0.0)
    close = (1 + ret.fillna(0)).cumprod().where(raw.notna()) * (raw.ffill().iloc[-1] / (1 + ret.fillna(0)).cumprod().iloc[-1])
    ctx = {
        "drop": close / close.rolling(60, min_periods=20).max() - 1,
        "liquid": (p["vol"].rolling(20).mean() >= 500) & (close >= 10),
        "inst10": (p["foreign_net"].fillna(0) + p["trust_net"].fillna(0)).rolling(10).sum(),
        "fwd20": close.shift(-20) / close - 1,
        "fwd60": close.shift(-60) / close - 1,
    }
    pe, pe_med = load_valuation()
    ctx["pe"] = pe.reindex(close.index).ffill(limit=3)
    ctx["pe_med"] = pe_med.reindex(close.index).ffill(limit=3)
    return raw, ctx


def revenue_asof(rev, d):
    r = rev[rev["available"] <= d].sort_values("ym").groupby("stock_id").tail(1)
    return r.set_index("stock_id")[["ym", "yoy", "yoy3"]]


def screen_day(d, raw_close, ctx, rev, names):
    row = lambda k: ctx[k].loc[d] if d in ctx[k].index else pd.Series(dtype=float)
    df = pd.DataFrame({k: row(k) for k in ctx})
    df["close"] = raw_close.loc[d]
    df = df[(df["drop"] <= DROP) & (df["liquid"] == True)]  # noqa: E712
    df = df.join(revenue_asof(rev, d), how="left")
    df["營收成長"] = (df["yoy3"] > 0.10) & (df["yoy"] > 0)
    df["估值變便宜"] = df["pe"] < df["pe_med"]
    df["誤殺候選"] = df["營收成長"] & df["估值變便宜"]
    df["法人在接"] = df["inst10"] > 0
    df["date"] = d
    df = df.reset_index().rename(columns={"index": "stock_id"})
    df["name"] = df["stock_id"].map(names)
    return df


def _setup():
    panel = load_panel()
    names = panel.drop_duplicates("stock_id", keep="last").set_index("stock_id")["name"]
    close, ctx = build(panel)
    rev = revenue_features(pd.read_csv(ROOT / "data" / "revenue.csv", dtype={"stock_id": str}))
    return close, ctx, rev, names


def today():
    close, ctx, rev, names = _setup()
    d = close.index[-1]
    df = screen_day(d, close, ctx, rev, names)
    out = df[df["誤殺候選"]].copy()
    out["高成長"] = out["yoy3"] >= YOY3_STRONG
    out = out.sort_values(["高成長", "yoy3"], ascending=False)
    cols = ["date", "stock_id", "name", "close", "drop", "ym", "yoy", "yoy3", "pe", "pe_med", "高成長", "法人在接"]
    out[cols].to_csv(RESULTS / "oversold_today.csv", index=False, encoding="utf-8-sig")
    print(f"{d.date()}：跌深 {len(df)} 檔，其中誤殺候選 {len(out)} 檔")
    print(out[cols].round(3).to_string(index=False))


def backtest(step=5):
    close, ctx, rev, names = _setup()
    m = load_market_index()
    dd = m["taiex"] / m["taiex"].rolling(120, min_periods=20).max() - 1
    days = [d for d in dd[dd <= -0.12].index if d in close.index][::step]
    res = pd.concat([screen_day(d, close, ctx, rev, names) for d in days], ignore_index=True)
    res.to_csv(RESULTS / "oversold_backtest.csv", index=False, encoding="utf-8-sig")
    res["月"] = res["date"].dt.strftime("%Y-%m")
    res["組別"] = np.select([res["誤殺候選"] & res["法人在接"], res["誤殺候選"], res["營收成長"]],
                          ["誤殺＋法人在接", "誤殺候選", "只有營收成長"], "其他跌深股")
    agg = lambda g: pd.Series({"筆數": len(g), "20日勝率": (g["fwd20"] > 0).mean(), "20日中位數": g["fwd20"].median(),
                               "60日勝率": (g["fwd60"] > 0).mean(), "60日中位數": g["fwd60"].median()})
    done = res.dropna(subset=["fwd60"])
    print(f"大盤距高點跌 12% 以上的日子（每 {step} 天取一次）：{len(days)} 天，月份 {sorted(res['月'].unique())}")
    print(done.groupby("組別").apply(agg).round(3).to_string())
    print("\n分月（60 日已有結果）：")
    print(done.groupby(["月", "組別"]).apply(agg).round(3).to_string())
    return res


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "today"
    backtest() if cmd == "backtest" else today()


if __name__ == "__main__":
    main()
