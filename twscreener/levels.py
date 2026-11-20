"""停利、停損參考價：用「同一組」歷史訊號之後的實際走勢算出來。

    python -m twscreener.levels      # 幫今天的 agent 推薦加上參考價 -> results/agent_levels_today.csv

同一組：策略訊號 = 同策略且同大盤狀態（不足 30 筆就用整個策略）；恐慌誤殺 = 同分數組。
每個持有期間（30/60/90/120 個交易日）：
- 中位數報酬、勝率（已扣 0.6% 成本）
- 停利參考 = 該期間報酬的第 75 百分位（四次裡大約一次做得到，偏樂觀）
- 保守目標 = 中位數
停損參考（不分期間）= 最後在 60 日賺錢的那些訊號，持有期間「最深跌幅」的第 20 百分位：
  八成會賺錢的交易都沒跌破這裡，跌破代表多半是看錯了。
  市況很亂時再放寬到「近 20 日平均單日波動 × 3」，取兩者較寬的。
歷史統計，不保證未來；價格用訊號日收盤價換算。
"""
import numpy as np
import pandas as pd

from .backtest import COST
from .market import RESULTS, load_panel
from .oversold import _tiers
from .review import market_context

HORIZONS = [30, 60, 90, 120]
MIN_N = 30
VOL_MULT = 3
MIN_N_OVERSOLD = 100  # 急跌次數少，同組樣本要多一點才用，不然用全部誤殺候選（較保守）


def adjusted_close(panel):
    raw = panel.pivot_table(index="date", columns="stock_id", values="close")
    ret = raw.pct_change(fill_method=None)
    ret = ret.where(ret.abs() <= 0.11, 0.0).fillna(0)
    return (1 + ret).cumprod().where(raw.notna())


def path_stats(close, events):
    """events: DataFrame(date, stock_id) -> 加上 ret_30..ret_120、mae_60（60 日內最深跌幅）。"""
    idx = {d: i for i, d in enumerate(close.index)}
    arr = close.to_numpy()
    cols = {s: j for j, s in enumerate(close.columns)}
    out = []
    for d, sid in zip(events["date"], events["stock_id"]):
        i, j = idx.get(d), cols.get(sid)
        row = {f"ret_{h}": np.nan for h in HORIZONS} | {"mae_60": np.nan}
        if i is not None and j is not None and np.isfinite(arr[i, j]):
            p0 = arr[i, j]
            for h in HORIZONS:
                if i + h < len(arr):
                    row[f"ret_{h}"] = arr[i + h, j] / p0 - 1 - COST
            if i + 60 < len(arr):
                row["mae_60"] = np.nanmin(arr[i + 1:i + 61, j]) / p0 - 1
        out.append(row)
    return pd.concat([events.reset_index(drop=True), pd.DataFrame(out)], axis=1)


def group_levels(g):
    r = {"樣本數": len(g)}
    for h in HORIZONS:
        x = g[f"ret_{h}"].dropna()
        r[f"{h}日勝率"] = (x > 0).mean() if len(x) >= 10 else np.nan
        r[f"{h}日中位數"] = x.median() if len(x) >= 10 else np.nan
        r[f"{h}日停利"] = x.quantile(0.75) if len(x) >= 10 else np.nan
        r[f"{h}日樣本"] = len(x)
    win = g[g["ret_60"] > 0]["mae_60"].dropna()
    r["停損"] = win.quantile(0.2) if len(win) >= 10 else np.nan
    return pd.Series(r)


def history():
    panel = load_panel()
    close = adjusted_close(panel)
    ctx = market_context()[["date", "大盤狀態"]]
    sig = pd.read_csv(RESULTS / "market_signals.csv", dtype={"stock_id": str})
    sig["date"] = pd.to_datetime(sig["signal_date"])
    sig = sig.merge(ctx, on="date", how="left")
    sig["來源"] = sig["strategy"].str.split("_", n=1).str[-1]
    over = pd.read_csv(RESULTS / "oversold_backtest.csv", dtype={"stock_id": str})
    over = over[over["誤殺候選"] == True].copy()  # noqa: E712
    over["date"] = pd.to_datetime(over["date"])
    over["cell"] = _tiers(over)
    over["來源"] = "恐慌誤殺"
    return close, path_stats(close, sig[["date", "stock_id", "來源", "大盤狀態"]]), \
        path_stats(close, over[["date", "stock_id", "來源", "cell"]])


def levels_for(picks):
    close, sig, over = history()
    rows = []
    for _, p in picks.iterrows():
        if p["source"] == "恐慌誤殺":
            o = pd.read_csv(RESULTS / "oversold_scored.csv", dtype={"stock_id": str}).set_index("stock_id")
            cell = o.loc[p["stock_id"], "cell"] if p["stock_id"] in o.index else None
            g = over[over["cell"] == cell]
            label = f"恐慌誤殺｜{cell}"
            if len(g) < MIN_N:
                g, label = over, "恐慌誤殺（全部）"
        else:
            regime = market_context()["大盤狀態"].dropna().iat[-1]
            g = sig[(sig["來源"] == p["source"]) & (sig["大盤狀態"] == regime)]
            label = f"{p['source']}｜{regime}"
            if len(g.dropna(subset=["ret_60"])) < MIN_N:
                g, label = sig[sig["來源"] == p["source"]], f"{p['source']}（全部）"
        lv = group_levels(g)
        broad = group_levels(over if p["source"] == "恐慌誤殺" else sig[sig["來源"] == p["source"]])
        thin = []
        for h in HORIZONS:  # 該期間樣本不夠就改用同來源的全部訊號
            need = MIN_N_OVERSOLD if p["source"] == "恐慌誤殺" else MIN_N
            if lv[f"{h}日樣本"] < need:
                for k in ["勝率", "中位數", "停利", "樣本"]:
                    lv[f"{h}日{k}"] = broad[f"{h}日{k}"]
                thin.append(f"{h}日")
        if thin:
            label += f"（{'、'.join(thin)}樣本不足，改用同來源全部訊號）"
        lv["急跌次數"] = g.dropna(subset=["ret_120"])["date"].dt.strftime("%Y-%m").nunique() if "date" in g else np.nan
        lv["stock_id"], lv["name"], lv["close"], lv["依據組別"] = p["stock_id"], p["name"], p["close"], label
        rows.append(lv)
    out = pd.DataFrame(rows)
    # 依近期波動放寬停損：至少留「近 20 日平均單日波動 × 3」的空間，避免恐慌盤一天就被洗出場
    vol = close.pct_change(fill_method=None).abs().iloc[-20:].mean()
    out["近20日平均波動"] = out["stock_id"].map(vol)
    out["歷史停損"] = out["停損"]
    out["停損"] = np.minimum(out["停損"], -VOL_MULT * out["近20日平均波動"].fillna(0))
    out["停損依據"] = np.where(out["停損"] < out["歷史停損"], "依近期波動放寬", "歷史統計")
    out["停損價"] = (out["close"] * (1 + out["停損"])).round(2)
    for h in HORIZONS:
        out[f"{h}日目標價"] = (out["close"] * (1 + out[f"{h}日中位數"] + COST)).round(2)
        out[f"{h}日停利價"] = (out["close"] * (1 + out[f"{h}日停利"] + COST)).round(2)
    return out


def main():
    log = pd.read_csv(RESULTS / "agent_picks_log.csv", dtype={"stock_id": str})
    day = log["pick_date"].max()
    picks = log[(log["pick_date"] == day) & (log["verdict"] == "推薦")]
    out = levels_for(picks)
    out.insert(0, "pick_date", day)
    out.to_csv(RESULTS / "agent_levels_today.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(out.round(3).T.to_string())


if __name__ == "__main__":
    main()
