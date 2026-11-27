"""處置股不同出場規則放進 v4 模擬帳戶實測（2026-11-27）。逐筆結果見 scripts/disposition_exit_variants_test.py。
處置最後一天收盤時判斷，條件成立就多抱 N 個交易日（那天收盤賣），不成立照舊當天收盤賣。其他帳戶規則不變。
  A 現在的 v4（處置最後一天收盤賣）
  B 再次處置多抱 20 天
  C 處置期間賠 10% 以上多抱 20 天
  D 處置期間融資減 10% 以上多抱 20 天
  E 再次處置或賠 10% 以上多抱 20 天
  F 全部多抱 20 天
  G 再次處置多抱 10 天
    python scripts/disposition_exit_account_test.py
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
raw = inputs[0]
days = raw.index
pos_of = {d: i for i, d in enumerate(days)}
p = load_panel()
mg = p.pivot_table(index="date", columns="stock_id", values="margin_bal").reindex(index=days, columns=raw.columns)
dd = pd.read_csv(ROOT / "data/disposition.csv", dtype={"stock_id": str}, parse_dates=["announce", "start", "end"])
dd["again"] = dd["measure"].str.contains("再次|第二次")
again = {(r.stock_id, str(r.end.date())): r.again for r in dd.itertuples()}
MODE = {"m": "A"}
EXT = {"B": 20, "C": 20, "D": 20, "E": 20, "F": 20, "G": 10}
orig_plan = P.Account.plan
done = set()


def plan(self, day, close, cands, prev=None, trend=None):
    m = MODE["m"]
    if m != "A" and len(self.pos):
        today = str(day.date())
        for idx, h in self.pos.iterrows():
            if pd.isna(h["exit_date"]) or str(h["exit_date"])[:10] != today:
                continue
            key = (h["stock_id"], str(h["entry_date"]))
            if key in done:
                continue
            done.add(key)
            sid, c = h["stock_id"], close.get(h["stock_id"], np.nan)
            ag = again.get((sid, today), False)
            lose = np.isfinite(c) and c / h["entry"] - 1 <= -0.10
            i0 = pos_of.get(pd.Timestamp(h["entry_date"]), None)
            m0 = mg.iat[i0 - 1, mg.columns.get_loc(sid)] if i0 and sid in mg.columns else np.nan
            m1 = mg.at[day, sid] if sid in mg.columns else np.nan
            mdrop = np.isfinite(m0) and np.isfinite(m1) and m0 > 0 and m1 / m0 - 1 <= -0.10
            ok = {"B": ag, "C": lose, "D": mdrop, "E": ag or lose, "F": True, "G": ag}[m]
            k = pos_of[day] + EXT[m]
            if ok and k < len(days):
                self.pos.at[idx, "exit_date"] = str(days[k].date())
            elif ok:
                self.pos.at[idx, "exit_date"] = "2099-12-31"
    return orig_plan(self, day, close, cands, prev, trend=trend)


P.Account.plan = plan
taiex = load_market_index()["taiex"].reindex(days).ffill()
navs, info = {}, {}
for m in "ABCDEFG":
    MODE["m"] = m
    done.clear()
    a = P.run(days[days >= "2023-01-01"], P.Account(), inputs, version="v4")
    n = pd.DataFrame(a.nav)
    navs[m] = n.set_index(pd.to_datetime(n["date"]))["淨值"]
    t = pd.DataFrame(a.trades)
    s = t[t["strategy"] == "處置股"]
    hold = (pd.to_datetime(s["賣出日"]) - pd.to_datetime(s["買進日"])).dt.days.mean()
    info[m] = (f"處置股 {len(s)} 筆、勝率 {(s['報酬'] > 0).mean():.0%}、平均 {s['報酬'].mean():+.1%}、中位 {s['報酬'].median():+.1%}、"
               f"最差 {s['報酬'].min():+.0%}、平均抱 {hold:.0f} 天")
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
out = ROOT / "results" / "處置股出場"
out.mkdir(parents=True, exist_ok=True)
(out / "account_variants.txt").write_text(text, encoding="utf-8")
