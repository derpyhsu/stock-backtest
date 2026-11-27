"""「處置期間跌深、解除當天買」加進 v4 帳戶的處置股那一份實測（2026-11-27）。逐筆結果見 scripts/disposition_rebound_test.py。
訊號：處置最後一天收盤比處置首日開盤低 X% 以上 → 隔天開盤買，抱 N 個交易日那天收盤賣；每筆和處置股一樣佔那份資金 5%，不設停損。
  A 現在的 v4
  H 加跌 10% 以上、抱 20 天（和處置股共用 20 個位子）
  I 加跌 10% 以上、抱 10 天
  J 加跌 15% 以上、抱 20 天
  K 加跌 10% 以上、抱 40 天
  L H，但位子放寬到 30 檔
  M 加跌 5% 以上、抱 20 天（門檻放寬，看是不是高原）
  N 處置股改成第 2 天開盤才買（scripts/entry_timing_test.py），不加跌深
  O N＋H
  Q N＋J
    python scripts/disposition_rebound_account_test.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import twscreener.portfolio as P  # noqa: E402
from twscreener.market import load_market_index, load_panel  # noqa: E402

inputs = P._inputs()
raw, vol = inputs[0], inputs[3]
days = raw.index
p = load_panel()
ro = p.pivot_table(index="date", columns="stock_id", values="open").reindex(index=days, columns=raw.columns)
dd = pd.read_csv(ROOT / "data/disposition.csv", dtype={"stock_id": str}, parse_dates=["announce", "start", "end"])
dd = dd[dd["stock_id"].str.fullmatch(r"\d{4}") & dd["stock_id"].isin(raw.columns) & dd["start"].isin(days)]
dd["last"] = [days[days.searchsorted(e, side="right") - 1] for e in dd["end"]]     # 處置最後一個交易日
dd["pret"] = [raw.at[l, s] / ro.at[st, s] - 1 for l, s, st in zip(dd["last"], dd["stock_id"], dd["start"])]
by_last = {d: g for d, g in dd.dropna(subset=["pret"]).groupby("last")}
CFG = {"O": (-0.10, 20, 20), "Q": (-0.15, 20, 20), "H": (-0.10, 20, 20), "I": (-0.10, 10, 20), "J": (-0.15, 20, 20), "K": (-0.10, 40, 20), "L": (-0.10, 20, 30), "M": (-0.05, 20, 20)}
MODE = {"m": "A"}
P.SLEEVES["disp"] = {"處置股", "處置跌深"}
orig = P.disp_pick


def disp_pick(day, raw_, vol_, ind, names):
    m = MODE["m"]
    if m in "NOQ":      # 處置首日收盤後才排單，第 2 天開盤買
        i_ = days.get_loc(day)
        g = orig(days[i_ - 1], raw_, vol_, ind, names) if i_ > 0 else pd.DataFrame()
        if len(g):
            g = g[g["exit_date"] > str(day.date())].assign(close_at_pick=lambda x: raw_.loc[day, x["stock_id"]].values).dropna(subset=["close_at_pick"])
    else:
        g = orig(day, raw_, vol_, ind, names)
    if m in "AN" or day not in by_last:
        return g
    th, hold, _ = CFG[m]
    e = by_last[day]
    e = e[e["pret"] <= th].drop_duplicates("stock_id")
    i = days.get_loc(day)
    liq = (vol_.iloc[max(0, i - 19):i + 1].mean() >= 500) & (raw_.iloc[i] >= 10)
    ids = [x for x in e["stock_id"] if liq.get(x, False) and not (len(g) and x in set(g["stock_id"]))]
    if not ids:
        return g
    k = i + hold
    out = str(days[k].date()) if k < len(days) else "2099-12-31"
    x = pd.DataFrame({"stock_id": ids, "name": [names.get(s, s) for s in ids], "strategy": "處置跌深",
                      "industry": [ind.get(s, "其他") for s in ids], "score": e.set_index("stock_id").loc[ids, "pret"].values, "stop": -0.999,
                      "close_at_pick": raw_.loc[day, ids].values, "sleeve": "disp", "weight": P.DISP_W, "exit_date": out})
    return pd.concat([g, x], ignore_index=True) if len(g) else x


P.disp_pick = disp_pick
taiex = load_market_index()["taiex"].reindex(days).ffill()
navs, info = {}, {}
for m in "AHIJKLMNOQ":
    MODE["m"] = m
    P.SLEEVE_MAX["disp"] = CFG[m][2] if m in CFG else 20
    a = P.run(days[days >= "2023-01-01"], P.Account(), inputs, version="v4")
    n = pd.DataFrame(a.nav)
    navs[m] = n.set_index(pd.to_datetime(n["date"]))["淨值"]
    t = pd.DataFrame(a.trades)
    info[m] = "；".join(f"{k} {len(s)} 筆、勝率 {(s['報酬'] > 0).mean():.0%}、平均 {s['報酬'].mean():+.1%}、中位 {s['報酬'].median():+.1%}、最差 {s['報酬'].min():+.0%}"
                       for k, s in t.groupby("strategy") if k.startswith("處置"))
P.SLEEVE_MAX["disp"] = 20
navs = pd.DataFrame(navs)
navs["加權指數"] = taiex.reindex(navs.index)
seg = navs[navs.index >= "2023-04-01"]


def summ(s):
    s = s.dropna() / s.dropna().iat[0]
    return f"{s.iat[-1] - 1:+.1%}／{(s / s.cummax() - 1).min():.1%}"


half = seg.index[len(seg) // 2]
lines = [f"期間 {seg.index[0].date()} ~ {seg.index[-1].date()}（報酬／最大回撤）"]
for k in seg.columns:
    yr = "  ".join(f"{y} {summ(x[k])}" for y, x in seg.groupby(seg.index.year))
    lines.append(f"{k:5s} 全期 {summ(seg[k])}  前半 {summ(seg.loc[seg.index < half, k])}  後半 {summ(seg.loc[seg.index >= half, k])}  {yr}  {info.get(k, '')}")
text = "\n".join(lines)
print(text)
(ROOT / "results" / "處置股出場" / "rebound_account.txt").write_text(text, encoding="utf-8")
