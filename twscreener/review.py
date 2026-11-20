"""策略健檢：把歷史訊號按情境切開，找出特別容易輸的地方。

    python -m twscreener.review            # -> results/健檢/YYYY-MM.md、results/健檢/buckets.csv

只做診斷，不改策略。一個情境要算「弱點」，必須在前後兩段期間都明顯比該策略平均差，
而且兩段各至少 MIN_N 筆，避免只是剛好湊到。
"""
from datetime import date

import numpy as np
import pandas as pd

from .market import RESULTS, load_market_index, load_panel

OUT = RESULTS / "健檢"
MIN_N = 40
GAP = 0.08  # 勝率比策略平均低 8 個百分點以上才算


def stock_context(panel):
    """每檔股票每天的情境特徵（未還原股價，只拿來分組）。"""
    close = panel.pivot_table(index="date", columns="stock_id", values="close")
    vol = panel.pivot_table(index="date", columns="stock_id", values="vol")
    ma20, ma60 = close.rolling(20).mean(), close.rolling(60).mean()
    feats = {
        "量比": vol / vol.rolling(20).mean(),
        "前20日漲幅": close.pct_change(20, fill_method=None),
        "離季線": close / ma60 - 1,
        "離月線": close / ma20 - 1,
        "股價": close,
    }
    out = pd.concat({k: v.stack() for k, v in feats.items()}, axis=1)
    out.index.names = ["date", "stock_id"]
    return out.reset_index()


def market_context():
    m = load_market_index()
    t = m["taiex"]
    ma20, ma60 = t.rolling(20).mean(), t.rolling(60).mean()
    regime = np.where(t > ma60, np.where(t > ma20, "多頭", "多頭回檔"), np.where(t < ma20, "空頭", "空頭反彈"))
    return pd.DataFrame({
        "date": m.index.values,
        "大盤狀態": regime,
        "大盤離高點": (t / t.rolling(120, min_periods=20).max() - 1).values,
        "大盤前20日": t.pct_change(20).values,
    })


def bucketize(df):
    df["量比分組"] = pd.cut(df["量比"], [0, 1, 2, 4, np.inf], labels=["量縮<1倍", "1~2倍", "2~4倍", "爆量>4倍"])
    df["前20日分組"] = pd.cut(df["前20日漲幅"], [-np.inf, -0.1, 0, 0.1, 0.3, np.inf],
                          labels=["跌>10%", "跌0~10%", "漲0~10%", "漲10~30%", "漲>30%"])
    df["離季線分組"] = pd.cut(df["離季線"], [-np.inf, -0.1, 0, 0.1, 0.25, np.inf],
                          labels=["低於季線10%+", "季線下0~10%", "季線上0~10%", "季線上10~25%", "季線上25%+"])
    df["股價分組"] = pd.cut(df["股價"], [0, 30, 100, 300, np.inf], labels=["<30元", "30~100元", "100~300元", ">300元"])
    df["大盤離高點分組"] = pd.cut(df["大盤離高點"], [-np.inf, -0.15, -0.05, np.inf],
                            labels=["大盤距高點跌>15%", "跌5~15%", "接近高點"])
    df["主力"] = df["main_force"].map({"foreign": "外資", "trust": "投信", "margin": "融資"}).fillna("其他")
    return df


DIMS = ["大盤狀態", "大盤離高點分組", "market", "主力", "量比分組", "前20日分組", "離季線分組", "股價分組"]


def analyze():
    sig = pd.read_csv(RESULTS / "market_signals.csv", dtype={"stock_id": str})
    sig["date"] = pd.to_datetime(sig["signal_date"])
    sig = sig.dropna(subset=["ret_60d"])
    panel = load_panel()
    sig = sig.merge(stock_context(panel), on=["date", "stock_id"], how="left")
    sig = sig.merge(market_context(), on="date", how="left")
    sig = bucketize(sig)
    cut = sig["date"].quantile(0.5)
    sig["期間"] = np.where(sig["date"] <= cut, "前段", "後段")
    sig["win"] = sig["ret_60d"] > 0

    base = pd.read_csv(RESULTS / "market_baseline.csv", dtype={"stock_id": str})
    base["date"] = pd.to_datetime(base["signal_date"])
    base = base.dropna(subset=["ret_60d"]).merge(market_context(), on="date", how="left")
    base["win"] = base["ret_60d"] > 0

    rows = []
    for s, g in sig.groupby("strategy"):
        overall = g["win"].mean()
        half_overall = g.groupby("期間")["win"].mean()
        for dim in DIMS:
            for val, b in g.groupby(dim, observed=True):
                halves = b.groupby("期間")["win"].agg(["mean", "size"])
                row = {"strategy": s, "dim": dim, "bucket": str(val), "n": len(b),
                       "win60": b["win"].mean(), "med60": b["ret_60d"].median(), "strategy_win60": overall}
                for h in ["前段", "後段"]:
                    row[f"{h}_n"] = int(halves["size"].get(h, 0))
                    row[f"{h}_gap"] = halves["mean"].get(h, np.nan) - half_overall.get(h, np.nan)
                row["weak"] = all(row[f"{h}_n"] >= MIN_N and row[f"{h}_gap"] <= -GAP for h in ["前段", "後段"])
                row["watch"] = not row["weak"] and all(row[f"{h}_n"] >= 30 and row[f"{h}_gap"] <= -GAP / 2 for h in ["前段", "後段"])
                row["strong"] = all(row[f"{h}_n"] >= MIN_N and row[f"{h}_gap"] >= GAP for h in ["前段", "後段"])
                rows.append(row)
    res = pd.DataFrame(rows)
    regime_base = base.groupby("大盤狀態")["win"].agg(["mean", "size"])
    return sig, res, regime_base, cut


def recent_section(since_days=70):
    """最近一段（60 日結果還沒出來）先看 20 日，和隨機買股比。"""
    sig = pd.read_csv(RESULTS / "market_signals.csv")
    base = pd.read_csv(RESULTS / "market_baseline.csv")
    start = (pd.Timestamp.today() - pd.Timedelta(days=since_days)).strftime("%Y-%m-%d")
    sig, base = (d[d["signal_date"] >= start].dropna(subset=["ret_20d"]) for d in (sig, base))
    if sig.empty:
        return []
    lines = ["", f"## 最近（{start} 起，60 日結果還沒出來，先看 20 日）", "",
             "| 策略 | 20 日勝率 | 20 日中位數 | 筆數 |", "|---|---|---|---|"]
    for s, g in sig.groupby("strategy"):
        r = g["ret_20d"]
        lines.append(f"| {s.split('_', 1)[-1]} | {(r > 0).mean():.0%} | {r.median():+.1%} | {len(r)} |")
    r = base["ret_20d"]
    lines.append(f"| 隨機買股 | {(r > 0).mean():.0%} | {r.median():+.1%} | {len(r)} |")
    return lines


def report(sig, res, regime_base, cut):
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "buckets.csv", index=False, encoding="utf-8-sig")
    pct = lambda x: f"{x:.0%}" if pd.notna(x) else "-"
    lines = [f"# 策略健檢 {date.today():%Y-%m}", "",
             f"資料：{sig['date'].min():%Y-%m-%d} ~ {sig['date'].max():%Y-%m-%d} 已有 60 日結果的訊號 {len(sig)} 筆。"
             f"前段／後段以 {cut:%Y-%m-%d} 切開，兩段都成立才算數（各至少 {MIN_N} 筆、勝率差 {GAP:.0%} 以上）。",
             "這些是歷史統計，用來找改善方向，不代表要直接改規則。", "",
             "## 大盤狀態對照（隨機買股的 60 日勝率）", "", "| 大盤狀態 | 隨機買股勝率 | 筆數 |", "|---|---|---|"]
    for k, r in regime_base.iterrows():
        lines.append(f"| {k} | {pct(r['mean'])} | {int(r['size'])} |")
    lines += recent_section()
    for s, g in res.groupby("strategy"):
        lines += ["", f"## {s.split('_', 1)[-1]}（整體 60 日勝率 {pct(g['strategy_win60'].iat[0])}）", ""]
        for flag, title in [("weak", "弱點（兩段期間都明顯較差）"), ("watch", "觀察中（兩段都差 4 個百分點以上）"),
                            ("strong", "強項（兩段期間都明顯較好）")]:
            sub = g[g[flag]].sort_values("win60", ascending=(flag == "weak"))
            lines.append(f"**{title}**：" + ("無" if sub.empty else ""))
            for _, r in sub.iterrows():
                lines.append(f"- {r['dim']}＝{r['bucket']}：勝率 {pct(r['win60'])}、中位數 {r['med60']:+.1%}（{r['n']} 筆；"
                             f"前段差 {r['前段_gap']:+.0%}、後段差 {r['後段_gap']:+.0%}）")
    path = OUT / f"{date.today():%Y-%m}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path)
    return path


def main():
    report(*analyze())


if __name__ == "__main__":
    main()
