"""跑回測、算勝率，並用 M哥的案例驗證規則。

用法：
    python -m mbacktest.backtest                 # 用 data/ 裡所有已下載的股票
    python -m mbacktest.backtest --stocks 1785 6282

輸出（results/）：
    signals.csv   每一筆訊號與之後的報酬
    summary.csv   各策略的訊號數、勝率、平均/中位數報酬（含大盤基準）
    cases.csv     M哥案例在寫文日期前後 30 天內有沒有被規則抓到
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from .features import DATA, ROOT, load_market, load_stock
from .strategies import MIN_HIST, STRATEGIES, main_force, three_breaks

HORIZONS = [5, 20, 60, 120]
MAX_HOLD = 250
RESULTS = ROOT / "results"


def simulate_exit(df, entry_i):
    """從 entry_i 開盤進場，遇到三破隔天開盤出場，最多持有 MAX_HOLD 天。"""
    tb = three_breaks(df).values
    o, n = df["open"].values, len(df)
    entry = o[entry_i]
    for j in range(entry_i, min(n - 1, entry_i + MAX_HOLD)):
        if tb[j]:
            return o[j + 1] / entry - 1, j + 1 - entry_i, "三破"
    j = min(n - 1, entry_i + MAX_HOLD)
    return df["close"].values[j] / entry - 1, j - entry_i, "到期" if j < n - 1 else "未結束"


def signals_for(df, mkt):
    rows = []
    c, o = df["close"].values, df["open"].values
    n = len(df)
    force = main_force(df)
    for name, fn in STRATEGIES.items():
        sig = fn(df, mkt).fillna(False).values
        lo = MIN_HIST.get(name, 250)
        for i in np.flatnonzero(sig):
            if i + 1 >= n or i < lo:  # 需要隔天開盤，且前面要有足夠的歷史（多數策略要一年）
                continue
            e = i + 1
            entry = o[e]
            if not entry or np.isnan(entry):
                continue
            row = {
                "strategy": name,
                "stock_id": df["stock_id"].iat[0],
                "signal_date": df.index[i].date(),
                "entry_price": entry,
                "main_force": force.iat[i],
            }
            for h in HORIZONS:
                row[f"ret_{h}d"] = c[e + h - 1] / entry - 1 if e + h - 1 < n else np.nan
            win = c[e:min(n, e + 60)]
            row["max_dd_60d"] = win.min() / entry - 1 if len(win) else np.nan
            r, d, why = simulate_exit(df, e)
            row.update({"trade_ret": r, "trade_days": d, "exit_reason": why})
            rows.append(row)
    return rows


def baseline_for(df, start=251):
    """基準：同一檔股票每 10 個交易日隨機進場一次的報酬。"""
    c, o, n = df["close"].values, df["open"].values, len(df)
    rows = []
    for e in range(start, n, 10):
        row = {"strategy": "基準_隨機進場", "stock_id": df["stock_id"].iat[0],
               "signal_date": df.index[e - 1].date(), "_e": e}
        for h in HORIZONS:
            row[f"ret_{h}d"] = c[e + h - 1] / o[e] - 1 if e + h - 1 < n else np.nan
        rows.append(row)
    return rows


def summarize(sig):
    out = []
    for name, g in sig.groupby("strategy", sort=False):
        row = {"strategy": name, "signals": len(g), "stocks": g["stock_id"].nunique()}
        for h in HORIZONS:
            r = g[f"ret_{h}d"].dropna()
            row[f"win_{h}d"] = (r > 0).mean() if len(r) else np.nan
            row[f"mean_{h}d"] = r.mean() if len(r) else np.nan
            row[f"median_{h}d"] = r.median() if len(r) else np.nan
        if "trade_ret" in g and g["trade_ret"].notna().any():
            t = g["trade_ret"].dropna()
            row["trade_win"] = (t > 0).mean()
            row["trade_mean"] = t.mean()
            row["trade_days_median"] = g["trade_days"].median()
        out.append(row)
    return pd.DataFrame(out)


def check_cases(sig):
    path = ROOT / "個股案例表.csv"
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not r["代號"] or len(r["觀察日期"]) < 7:
                continue
            d = pd.Timestamp(r["觀察日期"] if len(r["觀察日期"]) == 10 else r["觀察日期"] + "-15")
            g = sig[(sig["stock_id"] == r["代號"])]
            dates = pd.to_datetime(g["signal_date"])
            hit = g[(dates >= d - pd.Timedelta(days=30)) & (dates <= d + pd.Timedelta(days=30))]
            rows.append({
                "股票": r["股票"], "代號": r["代號"], "觀察日期": r["觀察日期"], "型態": r["型態"],
                "有資料": (DATA / "price" / f"{r['代號']}.csv").exists(),
                "抓到的策略": ";".join(sorted(hit["strategy"].unique())),
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stocks", nargs="*")
    args = ap.parse_args()
    ids = args.stocks or sorted(p.stem for p in (DATA / "price").glob("*.csv"))
    mkt = load_market()
    sig_rows, base_rows = [], []
    for i, sid in enumerate(ids, 1):
        df = load_stock(sid)
        if df is None or len(df) < 300:
            continue
        sig_rows += signals_for(df, mkt)
        base_rows += baseline_for(df)
        if i % 100 == 0:
            print(f"{i}/{len(ids)}")
    RESULTS.mkdir(exist_ok=True)
    sig = pd.DataFrame(sig_rows)
    base = pd.DataFrame(base_rows)
    if sig.empty:
        print("沒有任何訊號。")
        return
    sig.to_csv(RESULTS / "signals.csv", index=False, encoding="utf-8-sig")
    summary = pd.concat([summarize(sig), summarize(base)], ignore_index=True)
    summary.to_csv(RESULTS / "summary.csv", index=False, encoding="utf-8-sig")
    cases = check_cases(sig)
    cases.to_csv(RESULTS / "cases.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    print(summary.round(3).to_string(index=False))
    with_data = cases[cases["有資料"]]
    print(f"\n案例驗證：{(with_data['抓到的策略'] != '').sum()}/{len(with_data)} 個案例在寫文日期前後 30 天內被抓到")


if __name__ == "__main__":
    main()
