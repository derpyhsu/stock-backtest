"""季報條件當成濾網放進 v5 模擬帳戶實測（2026-11-27）。逐筆結果見 scripts/quarterly_factors_test.py。
只過濾營收加速、一年新高兩份的新買進（處置股不動），看的是訊號當天已經公布的最新一季（用法定申報期限當公布日）。
  A 現在的 v5
  B 最新一季虧損的不買
  C 最新一季 EPS、營益率都比去年同期差的不買
  D 虧損或都比去年差的不買
  E 只買最新一季 EPS 比去年同期高的
  F 只買最新一季 EPS 創近 4 季新高的
季報資料在 ../private-data/financials/（FinMind，不放 repo）。
    python scripts/quarterly_filter_account_test.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import twscreener.portfolio as P  # noqa: E402
from twscreener.market import load_market_index  # noqa: E402

inputs = P._inputs()
raw = inputs[0]
days = raw.index
fs = [pd.read_csv(f, dtype={"stock_id": str}) for f in (ROOT.parent / "private-data" / "financials").glob("*.csv") if f.stat().st_size > 10]
q = pd.concat([f for f in fs if len(f)]).pivot_table(index=["stock_id", "date"], columns="type", values="value").reset_index()
q["date"] = pd.to_datetime(q["date"])
q = q.sort_values(["stock_id", "date"])
DEAD = {3: (0, 5, 15), 6: (0, 8, 14), 9: (0, 11, 14), 12: (1, 3, 31)}
q["avail"] = [pd.Timestamp(d.year + DEAD[d.month][0], DEAD[d.month][1], DEAD[d.month][2]) for d in q["date"]]
g = q.groupby("stock_id")
q["om"] = q["OperatingIncome"] / q["Revenue"]
q["EPS_y"], q["om_y"] = g["EPS"].shift(4), g["om"].shift(4)
q["loss"] = q["EPS"] < 0
q["worse"] = (q["EPS"] < q["EPS_y"]) & (q["om"] < q["om_y"])
q["up"] = q["EPS"] > q["EPS_y"]
q["hi4"] = (q["EPS"] >= g["EPS"].transform(lambda s: s.rolling(4, min_periods=4).max())) & (q["EPS"] > 0)
pos = days.searchsorted(q["avail"], side="right")   # 期限後第一個交易日才看得到
q = q[pos < len(days)].assign(day=days[pos[pos < len(days)]])


def daily(col):
    w = q.pivot_table(index="day", columns="stock_id", values=col, aggfunc="last").astype(float)
    return w.reindex(days.union(w.index)).ffill().reindex(days)


F = {c: daily(c) for c in ("loss", "worse", "up", "hi4")}


def flag(col, day, sid):
    w = F[col]
    v = w.at[day, sid] if sid in w.columns else float("nan")
    return None if pd.isna(v) else bool(v)


def keep(m, day, sid):
    lo, wo, up, hi = (flag(c, day, sid) for c in ("loss", "worse", "up", "hi4"))
    if m == "B":
        return lo is not True
    if m == "C":
        return wo is not True
    if m == "D":
        return lo is not True and wo is not True
    if m == "E":
        return up is True
    if m == "F":
        return hi is True
    return True


MODE = {"m": "A"}
SKIP = {P.REBOUND, "處置股"}
orig_plan = P.Account.plan
cut = {}


def plan(self, day, close, cands, prev=None, trend=None):
    m = MODE["m"]
    if m != "A" and len(cands) and "strategy" in cands.columns:
        ok = [s in SKIP or keep(m, day, sid) for sid, s in zip(cands["stock_id"], cands["strategy"])]
        cut[m] = cut.get(m, 0) + len(ok) - sum(ok)
        cands = cands[ok].reset_index(drop=True)
        if not len(cands):
            cands = pd.DataFrame(columns=["stock_id"])
    return orig_plan(self, day, close, cands, prev, trend=trend)


P.Account.plan = plan
taiex = load_market_index()["taiex"].reindex(days).ffill()
navs, info = {}, {}
for m in "ABCDEF":
    MODE["m"] = m
    a = P.run(days[days >= "2023-01-01"], P.Account(), inputs, version="v5")
    n = pd.DataFrame(a.nav)
    navs[m] = n.set_index(pd.to_datetime(n["date"]))["淨值"]
    t = pd.DataFrame(a.trades)
    info[m] = "；".join(f"{k} {len(s)} 筆、勝率 {(s['報酬'] > 0).mean():.0%}、平均 {s['報酬'].mean():+.1%}"
                       for k, s in t.groupby("strategy") if k not in SKIP)
    print(m, "done", flush=True)
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
(ROOT / "results" / "季報因子" / "account_filter.txt").write_text(text, encoding="utf-8")
