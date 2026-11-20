"""全市場版：用 data/daily/ 的每日全市場資料回測、每日選股、對帳。

用法：
    python -m twscreener.market backtest   # 全市場回測 -> results/market_*.csv
    python -m twscreener.market scan       # 今天的推薦 -> results/picks_today.csv，並寫入 picks_log.csv
    python -m twscreener.market all        # 兩個都跑，並輸出儀表板資料 dashboard/data.json

只回測「流動性夠」的股票：訊號當天 20 日均量 >= MIN_VOL 張、股價 >= MIN_PRICE。
"""
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import HORIZONS, baseline_for, signals_for, simulate_exit, summarize
from .adjust import adjust_prices
from .adjust import load as load_exrights
from .features import ROOT, add_features
from .fundamentals import load_industry
from .strategies import MOMENTUM, STRATEGIES, main_force

DAILY = ROOT / "data" / "daily"
DAILY_INDEX = ROOT / "data" / "daily_index"
RESULTS = ROOT / "results"
DASH = ROOT / "dashboard"
MIN_VOL = 500
MIN_PRICE = 10
FORCE_NAME = {"foreign": "外資", "trust": "投信", "margin": "融資"}


def load_panel():
    files = sorted(DAILY.glob("*.csv"))
    if not files:
        sys.exit("data/daily 沒有資料，先跑 python -m twscreener.twse")
    # 只用最後一段連續的日子（中間缺超過 20 天就丟掉前面那段，避免均線跨過斷層）
    dates = [pd.Timestamp(f.stem) for f in files]
    start = 0
    for i in range(1, len(dates)):
        if (dates[i] - dates[i - 1]).days > 20:
            start = i
    files = files[start:]
    df = pd.concat((pd.read_csv(f, dtype={"stock_id": str}) for f in files), ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_market_index():
    files = sorted(DAILY_INDEX.glob("*.csv"))
    m = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    m["date"] = pd.to_datetime(m["date"])
    m = m.drop_duplicates("date").set_index("date").sort_index()
    out = pd.DataFrame(index=m.index)
    out["taiex"] = m["taiex"]
    out["taiex_ret5"] = out["taiex"].pct_change(5)
    out["mkt_foreign_net"] = m["twse_foreign_net"]
    out["mkt_foreign_net5"] = out["mkt_foreign_net"].rolling(5).sum()
    out["mkt_foreign_net5_q05"] = out["mkt_foreign_net5"].rolling(500, min_periods=120).quantile(0.05)
    return out


def stock_frames(panel):
    names = panel.drop_duplicates("stock_id", keep="last").set_index("stock_id")[["name", "market"]]
    ex = load_exrights()
    ex_by = {sid: g for sid, g in ex.groupby("stock_id")}
    for sid, g in panel.groupby("stock_id", sort=True):
        g = g[g["vol"] > 0].set_index("date").sort_index()
        g = g[~g.index.duplicated()]
        if len(g) < 260:
            continue
        df = adjust_prices(g[["open", "high", "low", "close", "vol"]].astype(float).copy(), ex_by.get(sid))
        df["foreign_net"] = g["foreign_net"].fillna(0.0) if "foreign_net" in g else 0.0
        df["trust_net"] = g["trust_net"].fillna(0.0) if "trust_net" in g else 0.0
        # 外資持股用累計淨買超近似：B1 只看增減，效果相同
        df["foreign_hold"] = df["foreign_net"].cumsum()
        df["trust_hold"] = df["trust_net"].cumsum()
        df["margin_bal"] = g["margin_bal"].ffill() if "margin_bal" in g else np.nan
        df["stock_id"] = sid
        yield sid, names.loc[sid, "name"], names.loc[sid, "market"], add_features(df)


def liquid(df):
    return (df["vma20"] >= MIN_VOL) & (df["close"] >= MIN_PRICE)


MOM_TOP = 30       # 動能：每月買進動能最強的前 30 檔
MOM_LOOK = (120, 20)  # 動能：過去 120 天到 20 天前的漲幅（跳過最近一個月，避免短線反轉）


def _forward(c, o, e, n):
    return {f"ret_{h}d": (c[e + h - 1] / o[e] - 1 if e + h - 1 < n else np.nan) for h in HORIZONS}


def index_baseline(mkt, start_date):
    """大盤：每天收盤買加權指數（不含息）。"""
    t = mkt["taiex"].dropna()
    t = t[t.index >= start_date]
    v, rows = t.values, []
    for i in range(len(t) - 1):
        row = {"strategy": "基準_大盤指數", "stock_id": "TAIEX", "signal_date": t.index[i].date()}
        for h in HORIZONS:
            row[f"ret_{h}d"] = v[i + h] / v[i] - 1 if i + h < len(v) else np.nan
        rows.append(row)
    return rows


def run_backtest():
    panel = load_panel()
    mkt = load_market_index()
    days = sorted(panel["date"].unique())
    common_start = pd.Timestamp(days[min(250, len(days) - 1)])  # 型態策略最早能出訊號的日子
    sig_rows, base_rows, n = [], [], 0
    mom_frames = {}
    for sid, name, market, df in stock_frames(panel):
        ok = liquid(df)
        rows = signals_for(df, mkt)
        for r in rows:
            if ok.get(pd.Timestamp(r["signal_date"]), False):
                r["name"], r["market"] = name, market
                sig_rows.append(r)
        okv = ok.values
        for r in baseline_for(df, start=MOM_LOOK[0] + 1):
            if okv[r.pop("_e") - 1]:
                base_rows.append(r)
        c = df["close"]
        mom_frames[sid] = (name, market, df[["open", "close", "ma20", "bigbar_low", "swing_low_prev"]].copy(),
                           (c.shift(MOM_LOOK[1]) / c.shift(MOM_LOOK[0]) - 1).where(ok))
        n += 1
    sig_rows += momentum_signals(mom_frames)
    sig = pd.DataFrame(sig_rows)
    base = pd.DataFrame(base_rows + index_baseline(mkt, pd.Timestamp(days[MOM_LOOK[0]])))
    RESULTS.mkdir(exist_ok=True)
    sig.to_csv(RESULTS / "market_signals.csv", index=False, encoding="utf-8-sig")
    base.to_csv(RESULTS / "market_baseline.csv", index=False, encoding="utf-8-sig")
    in_common = lambda d: pd.to_datetime(d["signal_date"]) >= common_start
    summary = pd.concat([summarize(sig[in_common(sig)]), summarize(base[in_common(base)])], ignore_index=True)
    summary["window"] = f"{common_start.date()} 起"
    long = pd.concat([summarize(sig[sig["strategy"].isin(["T1_投信連買", MOMENTUM])]), summarize(base)], ignore_index=True)
    long["window"] = f"{pd.Timestamp(days[MOM_LOOK[0]]).date()} 起"
    summary.to_csv(RESULTS / "market_summary.csv", index=False, encoding="utf-8-sig")
    long.to_csv(RESULTS / "market_summary_long.csv", index=False, encoding="utf-8-sig")
    print(f"{n} 檔股票，{len(sig)} 筆訊號")
    pd.set_option("display.width", 250)
    cols = ["strategy", "signals", "win_20d", "median_20d", "win_60d", "median_60d", "trade_win", "trade_mean", "window"]
    print(summary.reindex(columns=cols).round(3).to_string(index=False))
    print(long.reindex(columns=cols).round(3).to_string(index=False))
    return sig, summary


def momentum_signals(frames):
    """每個月第一個交易日，把流動性夠的股票依動能排序，買前 MOM_TOP 名。"""
    mom = pd.DataFrame({sid: f[3] for sid, f in frames.items()})
    first_days = mom.index.to_series().groupby(mom.index.to_period("M")).min()
    rows = []
    for d in first_days:
        top = mom.loc[d].dropna().nlargest(MOM_TOP)
        for sid, val in top.items():
            name, market, df = frames[sid][:3]
            i = df.index.get_loc(d)
            if i + 1 >= len(df):
                continue
            c, o, e = df["close"].values, df["open"].values, i + 1
            row = {"strategy": MOMENTUM, "stock_id": sid, "signal_date": d.date(), "entry_price": o[e],
                   "main_force": "momentum", "name": name, "market": market, "mom": float(val)}
            row.update(_forward(c, o, e, len(df)))
            win = c[e:min(len(df), e + 60)]
            row["max_dd_60d"] = win.min() / o[e] - 1 if len(win) else np.nan
            r, dd, why = simulate_exit(df, e)
            row.update({"trade_ret": r, "trade_days": dd, "exit_reason": why})
            rows.append(row)
    return rows


def stats_table(sig):
    """各策略、各主要力量的歷史成績，給今日推薦附上勝率用。"""
    out = {}
    for (strat, force), g in list(sig.groupby(["strategy", "main_force"])) + [((s, "all"), g) for s, g in sig.groupby("strategy")]:
        d = {"n": int(len(g))}
        for h in [20, 60, 120]:
            r = g[f"ret_{h}d"].dropna()
            d[f"n{h}"] = int(len(r))
            d[f"win{h}"] = float((r > 0).mean()) if len(r) else None
            d[f"med{h}"] = float(r.median()) if len(r) else None
        t = g["trade_ret"].dropna()
        d["trade_win"] = float((t > 0).mean()) if len(t) else None
        d["trade_mean"] = float(t.mean()) if len(t) else None
        out[f"{strat}|{force}"] = d
    return out


def run_scan(sig=None, frames=None):
    """今天收盤後的訊號。"""
    panel = load_panel()
    mkt = load_market_index()
    last = panel["date"].max()
    if sig is None:
        sig = pd.read_csv(RESULTS / "market_signals.csv", dtype={"stock_id": str})
    stats = stats_table(sig)
    picks, charts, moms = [], {}, []

    def add(sid, name, market, df, s, force, extra=None):
        st = stats.get(f"{s}|{force}") or stats.get(f"{s}|all") or {}
        if st.get("n60", 0) < 30:  # 樣本太少就用整個策略的成績
            st = stats.get(f"{s}|all") or {}
        past = sig[(sig["stock_id"] == sid) & (sig["strategy"] == s)]["ret_60d"].dropna()
        picks.append({
            "date": last.date().isoformat(), "stock_id": sid, "name": name, "market": market,
            "strategy": s, "close": float(df["close"].iat[-1]), "main_force": FORCE_NAME.get(force, force),
            "vol_ratio": float(df["vol"].iat[-1] / df["vma20"].iat[-1]),
            "win60": st.get("win60"), "med60": st.get("med60"), "win120": st.get("win120"),
            "med120": st.get("med120"), "hist_n": st.get("n60"),
            "own_n": int(len(past)), "own_win60": float((past > 0).mean()) if len(past) else None,
            **(extra or {}),
        })
        tail = df.iloc[-130:]
        charts[sid] = {
            "d": [x.strftime("%m/%d") for x in tail.index],
            "c": tail["close"].round(2).tolist(),
            "ma20": tail["ma20"].round(2).tolist(),
            "ma60": tail["ma60"].round(2).tolist(),
            "ma240": tail["ma240"].round(2).tolist(),
            "v": tail["vol"].round(0).tolist(),
        }

    for sid, name, market, df in stock_frames(panel):
        if df.index[-1] != last or not liquid(df).iat[-1]:
            continue
        force = main_force(df).iat[-1]
        for s, fn in STRATEGIES.items():
            if bool(fn(df, mkt).fillna(False).iat[-1]):
                add(sid, name, market, df, s, force)
        c = df["close"]
        if len(c) > MOM_LOOK[0]:
            m = c.iat[-1 - MOM_LOOK[1]] / c.iat[-1 - MOM_LOOK[0]] - 1
            if np.isfinite(m):
                moms.append((m, sid, name, market, df))
                moms = sorted(moms, key=lambda x: -x[0])[:10]  # 只留前 10，省記憶體
    # 動能：列出目前排名前 10（每月第一個交易日換股時就買這一群）
    for m, sid, name, market, df in moms:
        add(sid, name, market, df, MOMENTUM, "momentum", {"mom": float(m)})
    picks = pd.DataFrame(picks)
    if not picks.empty:
        picks["industry"] = picks["stock_id"].map(load_industry()).fillna("")
        picks = picks.sort_values(["win60", "vol_ratio"], ascending=False)
    picks.to_csv(RESULTS / "picks_today.csv", index=False, encoding="utf-8-sig")
    log_picks(picks)
    print(f"{last.date()} 推薦 {len(picks)} 筆")
    return last, picks, charts, stats


def log_picks(picks):
    path = RESULTS / "picks_log.csv"
    cols = ["date", "stock_id", "name", "strategy", "close", "main_force", "win60"]
    old = pd.read_csv(path, dtype={"stock_id": str}) if path.exists() else pd.DataFrame(columns=cols)
    if not picks.empty:
        old = old[old["date"] != picks["date"].iat[0]]
        old = pd.concat([old, picks[cols]], ignore_index=True)
    old.to_csv(path, index=False, encoding="utf-8-sig")


def grade_log(panel):
    """推薦對帳：每筆推薦隔天開盤買進，20/60 個交易日後的報酬。"""
    path = RESULTS / "picks_log.csv"
    if not path.exists():
        return []
    log = pd.read_csv(path, dtype={"stock_id": str})
    px = panel.set_index(["stock_id", "date"]).sort_index()
    out = []
    for r in log.itertuples():
        try:
            g = px.loc[r.stock_id]
        except KeyError:
            continue
        g = g[g.index > pd.Timestamp(r.date)]
        row = {"date": r.date, "stock_id": r.stock_id, "name": r.name, "strategy": r.strategy,
               "days": int(len(g))}
        if len(g):
            entry = g["open"].iat[0]
            row["entry"] = float(entry)
            row["ret_now"] = float(g["close"].iat[-1] / entry - 1)
            for h in [20, 60]:
                row[f"ret{h}"] = float(g["close"].iat[h - 1] / entry - 1) if len(g) >= h else None
        out.append(row)
    return out


def walk_forward(sig):
    """每一季訊號的 60 日勝率，看策略是不是穩定。"""
    s = sig.dropna(subset=["ret_60d"]).copy()
    s["q"] = pd.PeriodIndex(pd.to_datetime(s["signal_date"]), freq="Q").astype(str)
    out = {}
    for strat, g in s.groupby("strategy"):
        out[strat] = [{"q": q, "n": int(len(x)), "win": float((x["ret_60d"] > 0).mean()), "med": float(x["ret_60d"].median())}
                      for q, x in g.groupby("q")]
    return out


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean(v) for v in o]
    if isinstance(o, float) and (np.isnan(o) or np.isinf(o)):
        return None
    if isinstance(o, (np.floating,)):
        return clean(float(o))
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


def export_dashboard(sig, summary, last, picks, charts, source=""):
    panel = load_panel()
    cases_path = RESULTS / "cases.csv"
    cases = pd.read_csv(cases_path, dtype={"代號": str}).fillna("").to_dict("records") if cases_path.exists() else []
    base = summary[summary["strategy"] == "基準_隨機進場"].iloc[0].to_dict()
    case_sum = RESULTS / "summary.csv"
    data = {
        "summary_cases": pd.read_csv(case_sum).to_dict("records") if case_sum.exists() else [],
        "summary_long": pd.read_csv(RESULTS / "market_summary_long.csv").to_dict("records") if (RESULTS / "market_summary_long.csv").exists() else [],
        "updated": last.date().isoformat(),
        "universe": {"stocks": int(sig["stock_id"].nunique()) if len(sig) else 0, "min_vol": MIN_VOL, "source": source,
                     "from": str(pd.to_datetime(sig["signal_date"]).min().date()) if len(sig) else None,
                     "to": str(pd.to_datetime(sig["signal_date"]).max().date()) if len(sig) else None},
        "summary": summary.to_dict("records"),
        "baseline": base,
        "picks": picks.to_dict("records") if not picks.empty else [],
        "charts": charts,
        "log": grade_log(panel),
        "walk": walk_forward(sig),
        "cases": cases,
    }
    DASH.mkdir(exist_ok=True)
    js = json.dumps(clean(data), ensure_ascii=False).replace("</", "<\\/")
    (DASH / "data.json").write_text(js, encoding="utf-8")
    html = (DASH / "template.html").read_text(encoding="utf-8").replace("/*__DATA__*/", js)
    (DASH / "index.html").write_text(html, encoding="utf-8")
    print("儀表板 -> dashboard/index.html")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "backtest":
        run_backtest()
    elif cmd == "scan":
        run_scan()
    else:
        sig, summary = run_backtest()
        n60 = int(sig["ret_60d"].notna().sum())
        source = f"全市場實測（{n60} 筆已滿 60 日的訊號）"
        if sig["ret_60d"].notna().sum() < 200:
            # 全市場歷史還不夠長、還算不出 60 日後報酬時，先用案例股（FinMind，2017 起）的統計
            sig = pd.read_csv(RESULTS / "signals.csv", dtype={"stock_id": str})
            summary = pd.read_csv(RESULTS / "summary.csv")
            source = f"暫用 手冊案例股 {sig['stock_id'].nunique()} 檔（全市場歷史還在補）"
        last, picks, charts, _ = run_scan(sig)
        export_dashboard(sig, summary, last, picks, charts, source)


if __name__ == "__main__":
    main()
