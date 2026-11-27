"""季報因子驗證（2026-11-27）。資料：scripts/fetch_financials.py 抓的季損益表（專案外 ../private-data/financials/）。
財報狗、菲比斯等影片說台股最有效的是財報因子（EPS、毛利率、營益率成長或創新高、轉虧為盈），之前沒資料測不了。

不偷看：季報用法定申報期限當可用日（Q1 5/15、Q2 8/14、Q3 11/14、Q4 隔年 3/31），期限後第一個交易日收盤當訊號日，
隔天開盤買（開盤漲停買不到）、抱 N 個交易日收盤賣，扣 0.6%。實際上多數公司更早公布，所以這裡是保守的。
比隨機＝每天一票：同一天訊號平均減同一天所有流動性股票平均。流動性：20 日均量 >= 500 張、股價 >= 10。
    python scripts/quarterly_factors_test.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from twscreener.market import load_panel  # noqa: E402
from twscreener.simulate import prices  # noqa: E402

H = (20, 60, 120)
COST = 0.006
raw, adj, opn, vol = prices(load_panel())
days = raw.index
ro = load_panel().pivot_table(index="date", columns="stock_id", values="open").reindex(index=days, columns=raw.columns)
liquid = (vol.rolling(20).mean() >= 500) & (raw >= 10)
can_buy = ro.shift(-1) < raw * 1.095
entry = opn.shift(-1)
fwd = {h: adj.shift(-h) / entry - 1 - COST for h in H}
base = {h: fwd[h].where(liquid).mean(axis=1) for h in H}

fs = [pd.read_csv(f, dtype={"stock_id": str}) for f in (ROOT.parent / "private-data" / "financials").glob("*.csv") if f.stat().st_size > 10]
q = pd.concat([f for f in fs if len(f)]).pivot_table(index=["stock_id", "date"], columns="type", values="value").reset_index()
q["date"] = pd.to_datetime(q["date"])
q = q.sort_values(["stock_id", "date"])
DEAD = {3: (0, 5, 15), 6: (0, 8, 14), 9: (0, 11, 14), 12: (1, 3, 31)}
q["avail"] = [pd.Timestamp(d.year + DEAD[d.month][0], DEAD[d.month][1], DEAD[d.month][2]) for d in q["date"]]
pos = days.searchsorted(q["avail"], side="right")
q = q[pos < len(days)].assign(day=days[pos[pos < len(days)]])
q = q[q["day"] >= days[0]]
g = q.groupby("stock_id")
q["gm"], q["om"] = q["GrossProfit"] / q["Revenue"], q["OperatingIncome"] / q["Revenue"]
q["nm"] = q["IncomeAfterTaxes"] / q["Revenue"]
g = q.groupby("stock_id")
for c in ("EPS", "gm", "om", "nm", "OperatingIncome", "Revenue"):
    q[c + "_y"], q[c + "_q"] = g[c].shift(4), g[c].shift(1)
q["eps_up"] = q["EPS"] > q["EPS_y"]
g = q.groupby("stock_id")
q["eps_up2"] = q["eps_up"] & g["eps_up"].shift(1).fillna(False).astype(bool)
q["eps_up3"] = q["eps_up2"] & g["eps_up"].shift(2).fillna(False).astype(bool)
q["eps_hi4"] = (q["EPS"] >= g["EPS"].transform(lambda s: s.rolling(4, min_periods=4).max())) & (q["EPS"] > 0)
q["eps_hi8"] = (q["EPS"] >= g["EPS"].transform(lambda s: s.rolling(8, min_periods=8).max())) & (q["EPS"] > 0)
q["eps_g"] = (q["EPS"] - q["EPS_y"]) / q["EPS_y"].abs()
q["three_y"] = (q["gm"] > q["gm_y"]) & (q["om"] > q["om_y"]) & (q["nm"] > q["nm_y"])
q["three_q"] = (q["gm"] > q["gm_q"]) & (q["om"] > q["om_q"]) & (q["nm"] > q["nm_q"])
q["turn"] = (q["EPS"] > 0) & (q["EPS_y"] < 0)
q["turn_q"] = (q["EPS"] > 0) & (q["EPS_q"] < 0)
q["worse"] = (q["EPS"] < q["EPS_y"]) & (q["om"] < q["om_y"])
q["oi_g"] = (q["OperatingIncome"] - q["OperatingIncome_y"]) / q["OperatingIncome_y"].abs()
q["has"] = q["EPS"].notna() & q["EPS_y"].notna()


def S(mask):
    x = q[mask.fillna(False) & q["stock_id"].isin(raw.columns)]
    a = np.zeros(raw.shape, bool)
    a[days.get_indexer(x["day"]), raw.columns.get_indexer(x["stock_id"])] = True
    return pd.DataFrame(a, index=days, columns=raw.columns)


sig = {
    "所有有季報的（對照）": S(q["has"]),
    "EPS 比去年同期高": S(q["eps_up"]),
    "EPS 連 2 季比去年同期高": S(q["eps_up2"]),
    "EPS 連 3 季比去年同期高": S(q["eps_up3"]),
    "EPS 年增 50% 以上（去年同期賺錢）": S((q["eps_g"] >= 0.5) & (q["EPS_y"] > 0)),
    "EPS 年增 100% 以上（去年同期賺錢）": S((q["eps_g"] >= 1.0) & (q["EPS_y"] > 0)),
    "EPS 創近 4 季新高": S(q["eps_hi4"]),
    "EPS 創近 8 季新高": S(q["eps_hi8"]),
    "三率（毛利率、營益率、淨利率）都比去年同期高": S(q["three_y"]),
    "三率都比上一季高": S(q["three_q"]),
    "三率年增、季增都升": S(q["three_y"] & q["three_q"]),
    "三率年增都升＋EPS 創近 8 季新高": S(q["three_y"] & q["eps_hi8"]),
    "營業利益年增 30% 以上（去年同期為正）": S((q["oi_g"] >= 0.3) & (q["OperatingIncome_y"] > 0)),
    "轉虧為盈（去年同期虧、這季賺）": S(q["turn"]),
    "轉虧為盈（上一季虧、這季賺）": S(q["turn_q"]),
    "EPS、營益率都比去年同期差": S(q["worse"]),
    "這季虧損": S(q["EPS"] < 0),
}
half = days[len(days) // 2]
rows = []
for name, s in sig.items():
    m = (s & liquid & can_buy).fillna(False)
    for h in H:
        r = fwd[h].where(m)
        st = r.stack().dropna()
        if len(st) < 30:
            continue
        x = r.sub(base[h], axis=0).mean(axis=1).dropna()
        ev = r.sub(base[h], axis=0).stack().dropna()
        yr = ev.groupby(ev.index.get_level_values(0).year).mean()
        rows.append({"策略": name, "天數": h, "筆數": len(st), "勝率": (st > 0).mean(), "中位數": st.median(), "比隨機": ev.mean(),
                     "贏的年數": f"{(yr > 0).sum()}/{len(yr)}", "每年": " ".join(f"{y % 100}:{v:+.1%}" for y, v in yr.items())})
t = pd.DataFrame(rows)
lines = [f"{days[0]:%Y-%m-%d} ~ {days[-1]:%Y-%m-%d}，{q['stock_id'].nunique()} 檔有季報，成本 {COST:.1%}",
         "訊號日集中在每季申報期限後一天，所以「比隨機」是每筆訊號減同一天全部流動性股票的平均，再對所有筆平均。"]
for title, col, fmt in (("比隨機（每筆超額）", "比隨機", lambda v: f"{v:+.1%}"), ("筆數", "筆數", lambda v: f"{v:.0f}"),
                        ("勝率", "勝率", lambda v: f"{v:.0%}"), ("贏的年數", "贏的年數", str)):
    pv = t.pivot(index="策略", columns="天數", values=col).reindex([o for o in sig if o in set(t["策略"])])
    lines += ["", f"== {title}", pv.map(lambda v: fmt(v) if pd.notna(v) else "").to_string()]
lines += ["", "== 60 天逐年超額"] + [f"{r['策略']}：{r['每年']}" for _, r in t[t["天數"] == 60].iterrows()]
text = "\n".join(lines)
print(text)
out = ROOT / "results" / "季報因子"
out.mkdir(parents=True, exist_ok=True)
(out / "quarterly_factors.txt").write_text(text, encoding="utf-8")
