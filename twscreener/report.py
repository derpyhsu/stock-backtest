"""回測總報告：勝率之外，加上扣成本後的期望值、賺賠比、最差回撤，並按大盤狀態分開。

    python -m twscreener.report      # -> results/回測報告.md

先跑 python -m twscreener.market backtest（或 all）。
"""
from datetime import date

import numpy as np
import pandas as pd

from .backtest import COST
from .market import RESULTS
from .review import market_context

NAME = lambda s: s.split("_", 1)[-1]


def metrics(g):
    r = g["ret_60d"].dropna() - COST
    if len(r) == 0:
        return pd.Series(dtype=float)
    win, loss = r[r > 0], r[r <= 0]
    trim = r[r <= r.quantile(0.95)]  # 去掉最好的 5%，看是不是靠少數暴漲股撐起來
    return pd.Series({
        "筆數": len(r), "勝率": (r > 0).mean(), "中位數": r.median(),
        "平均賺": win.mean() if len(win) else np.nan, "平均賠": loss.mean() if len(loss) else np.nan,
        "賺賠比": -win.mean() / loss.mean() if len(win) and len(loss) else np.nan,
        "期望值": r.mean(), "去掉前5%期望值": trim.mean(),
        "最差10%回撤": g["max_dd_60d"].quantile(0.1) if "max_dd_60d" in g else np.nan,
    })


def load():
    sig = pd.read_csv(RESULTS / "market_signals.csv", dtype={"stock_id": str})
    base = pd.read_csv(RESULTS / "market_baseline.csv", dtype={"stock_id": str})
    base = base[base["strategy"] == "基準_隨機進場"]
    allsig = pd.concat([sig, base], ignore_index=True)
    allsig["date"] = pd.to_datetime(allsig["signal_date"])
    ctx = market_context()
    ctx["急跌"] = ctx["大盤離高點"] <= -0.12
    allsig = allsig.merge(ctx, on="date", how="left")
    return allsig


def fmt(df):
    out = df.copy()
    for c in out.columns:
        if c == "筆數":
            out[c] = out[c].map(lambda x: f"{int(x)}")
        elif c == "賺賠比":
            out[c] = out[c].map(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
        elif c == "勝率":
            out[c] = out[c].map(lambda x: f"{x:.0%}" if pd.notna(x) else "-")
        else:
            out[c] = out[c].map(lambda x: f"{x:+.1%}" if pd.notna(x) else "-")
    return out


def md_table(df, index_name):
    df = df.reset_index().rename(columns={"index": index_name})
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "---|" * len(df.columns)
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return [head, sep, *rows]


def main():
    s = load()
    s["策略"] = s["strategy"].map(lambda x: "隨機買股" if x.startswith("基準") else NAME(x))
    done = s.dropna(subset=["ret_60d"])
    lines = [f"# 回測報告（{date.today():%Y-%m-%d}）", "",
             f"期間：{done['date'].min():%Y-%m-%d} ~ {done['date'].max():%Y-%m-%d} 的訊號（60 日結果已出來的）。"
             f"持有 60 個交易日，每筆已扣交易成本 {COST:.1%}（手續費兩次＋證交稅）。"
             "減資、分割造成的價格跳動已修正。", "",
             "**怎麼看**：勝率高不代表賺錢，要看「期望值」（平均每筆賺多少，已扣成本）。"
             "「去掉前5%期望值」是把最好的 5% 拿掉後的期望值，用來看是不是只靠少數暴漲股撐起來。"
             "歷史統計，不保證未來。", "", "## 全期間", ""]
    tab = done.groupby("策略").apply(metrics).sort_values("期望值", ascending=False)
    lines += md_table(fmt(tab), "策略")
    for col, title in [("大盤狀態", "按大盤狀態（訊號當天，大盤和月線、季線的位置）"),
                       ("急跌", "大盤急跌時（距 120 日高點跌 12% 以上）")]:
        lines += ["", f"## {title}", ""]
        sub = done.dropna(subset=[col])
        if col == "急跌":
            sub = sub[sub["急跌"] == True]  # noqa: E712
            t = sub.groupby("策略").apply(metrics)
        else:
            t = sub.groupby(["策略", col]).apply(metrics)[["筆數", "勝率", "中位數", "期望值"]].unstack(col)
            t.columns = [f"{b}{a}" for a, b in t.columns]
            keep = [c for c in t.columns if c.endswith(("勝率", "期望值"))]
            t = t[sorted(keep, key=lambda c: (c[:-3] if c.endswith("期望值") else c[:-2], c))]
            lines += md_table(t.map(lambda x: f"{x:.0%}" if pd.notna(x) else "-"), "策略")
            continue
        lines += md_table(fmt(t.sort_values("期望值", ascending=False)), "策略")
    path = RESULTS / "回測報告.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
