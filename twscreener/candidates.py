"""agent 複審用的候選表：把今天各策略的訊號和恐慌誤殺高分股合在一起，每檔附上歷史數據。

    python -m twscreener.candidates     # -> results/candidates_today.csv

agent 只能從這張表裡挑（見 skill/agent-review/SKILL.md）。
排序用「歷史期望值」：策略訊號用該策略在目前大盤狀態下的期望值，誤殺股用同組股票過去急跌時的結果。
"""
import pandas as pd

from .backtest import COST
from .fundamentals import load_industry
from .market import RESULTS
from .report import load, metrics

TOP_OVERSOLD = 15


def strategy_stats():
    s = load()
    done = s.dropna(subset=["ret_60d"])
    allp = done.groupby("strategy").apply(metrics)
    regime = done.groupby(["strategy", "大盤狀態"]).apply(metrics)
    return allp, regime


def build():
    allp, regime = strategy_stats()
    ctx = load()
    today_regime = ctx.dropna(subset=["大盤狀態"]).sort_values("date")["大盤狀態"].iat[-1]
    ind = load_industry()
    rows = []
    picks = pd.read_csv(RESULTS / "picks_today.csv", dtype={"stock_id": str})
    for _, p in picks.iterrows():
        st = allp.loc[p["strategy"]] if p["strategy"] in allp.index else pd.Series(dtype=float)
        key = (p["strategy"], today_regime)
        rg = regime.loc[key] if key in regime.index and regime.loc[key, "筆數"] >= 30 else st
        rows.append({
            "date": p["date"], "stock_id": p["stock_id"], "name": p["name"], "industry": ind.get(p["stock_id"], ""),
            "來源": p["strategy"].split("_", 1)[-1], "close": p["close"],
            "歷史勝率": rg.get("勝率"), "歷史期望值": rg.get("期望值"), "保守期望值": rg.get("去掉前5%期望值"),
            "歷史中位數": rg.get("中位數"), "樣本數": rg.get("筆數"),
            "依據": f"{p['strategy'].split('_', 1)[-1]}在「{today_regime}」時的歷史表現；主力 {p['main_force']}，量比 {p['vol_ratio']:.1f}",
        })
    path = RESULTS / "oversold_scored.csv"
    if path.exists():
        o = pd.read_csv(path, dtype={"stock_id": str})
        o = o[o["誤殺候選"] & o["法人在接"]].sort_values(["win_est", "med_est", "yoy3"], ascending=False).head(TOP_OVERSOLD)
        for _, r in o.iterrows():
            rows.append({
                "date": r["date"][:10], "stock_id": r["stock_id"], "name": r["name"], "industry": r["industry"],
                "來源": "恐慌誤殺", "close": r["close"],
                "歷史勝率": r["win_est"], "歷史期望值": None, "保守期望值": None,
                "歷史中位數": r["med_est"] - COST, "樣本數": r["n"],
                "依據": f"比 60 日高點跌 {r['drop']:.0%}，近 3 月營收年增 {r['yoy3']:.0%}，本益比 {r['pe']:.1f}（過去一年中位數 {r['pe_med']:.1f}），法人近 10 日買超",
            })
    out = pd.DataFrame(rows)
    out["排序分數"] = out["保守期望值"].fillna(out["歷史中位數"])
    out = out.sort_values("排序分數", ascending=False).drop_duplicates("stock_id")
    out["大盤狀態"] = today_regime
    out.to_csv(RESULTS / "candidates_today.csv", index=False, encoding="utf-8-sig")
    print(f"大盤狀態：{today_regime}，候選 {len(out)} 檔")
    print(out.drop(columns=["依據"]).round(3).to_string(index=False))
    return out


if __name__ == "__main__":
    build()
