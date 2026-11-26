"""族群一起漲停（消息面行情）之後怎麼走、該抱幾天（2026-11-26）。

族群用證交所的產業別（data/industry.csv），比「低軌衛星」這種題材粗，但有完整歷史。
族群漲停日＝同一產業當天有 3 檔以上有流動性的股票收盤漲停（漲 9.5% 以上）。
  漲停股：當天漲停的那幾檔
  跟漲股：同產業、當天漲 3%~9.5%
  落後股：同產業、當天漲不到 3%
做法：訊號日收盤判斷、隔天開盤買（開盤漲停買不到）、抱 N 個交易日收盤賣（N=1 就是當天收盤賣），扣 0.6%。
比隨機＝每天一票：同一天訊號平均減同一天所有流動性股票平均。只用 2022 年以後的資料。
    python scripts/group_limitup_test.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from twscreener.market import load_panel  # noqa: E402
from twscreener.simulate import prices  # noqa: E402

H = (1, 2, 3, 5, 10, 20, 60)
COST = 0.006
p = load_panel()
p = p[p["date"] >= "2022-01-01"]
raw, adj, opn, vol = prices(p)
ro = p.pivot_table(index="date", columns="stock_id", values="open").reindex(index=raw.index, columns=raw.columns)
days = raw.index
liquid = (vol.rolling(20).mean() >= 500) & (raw >= 10)
can_buy = ro.shift(-1) < raw * 1.095
entry = opn.shift(-1)
fwd = {h: adj.shift(-h) / entry - 1 - COST for h in H}
base = {h: fwd[h].where(liquid).mean(axis=1) for h in H}
gap = {"隔天開盤跳空": ro.shift(-1) / raw - 1}

chg = raw / raw.shift(1) - 1
lim = (chg >= 0.095) & liquid
ind = pd.read_csv(ROOT / "data" / "industry.csv", dtype={"stock_id": str}).set_index("stock_id")["industry"].reindex(raw.columns).fillna("其他")
cnt = lim.T.groupby(ind.values).sum().T                       # 每天每個產業的漲停檔數
grp = (cnt >= 3).drop(columns=[c for c in ("其他",) if c in cnt.columns])
hot = pd.DataFrame({s: grp[ind[s]] if ind[s] in grp.columns else False for s in raw.columns}, index=days)
first_grp = grp & ~(grp.astype(float).rolling(5).max().shift(1).fillna(0) > 0)
hot1 = pd.DataFrame({s: first_grp[ind[s]] if ind[s] in grp.columns else False for s in raw.columns}, index=days)
streak = lim & lim.shift(1).fillna(False).astype(bool)

sig = {
    "所有漲停股": lim,
    "　單獨漲停（族群沒跟）": lim & ~hot,
    "族群漲停：漲停股": lim & hot,
    "　族群 5 日內第一次漲停：漲停股": lim & hot1,
    "　族群漲停：漲停股是連第 2 根以上": lim & hot & streak,
    "族群漲停：跟漲股（漲 3%~9.5%）": hot & liquid & ~lim & (chg >= 0.03),
    "　族群第一次：跟漲股": hot1 & liquid & ~lim & (chg >= 0.03),
    "族群漲停：落後股（漲不到 3%）": hot & liquid & (chg < 0.03),
    "　族群第一次：落後股": hot1 & liquid & (chg < 0.03),
}
rows = []
for name, s in sig.items():
    m = (s & liquid & can_buy).fillna(False)
    blocked = 1 - (s & liquid & can_buy).sum().sum() / max((s & liquid).sum().sum(), 1)
    g = gap["隔天開盤跳空"].where(m).stack().mean()
    for h in H:
        r = fwd[h].where(m)
        st = r.stack().dropna()
        if len(st) < 30:
            continue
        x = r.sub(base[h], axis=0).mean(axis=1).dropna()
        yr = x.groupby(x.index.year).mean()
        rows.append({"策略": name, "天數": h, "筆數": len(st), "訊號天數": len(x), "勝率": (st > 0).mean(), "平均": st.mean(), "中位數": st.median(),
                     "比隨機": x.mean(), "贏的年數": f"{(yr > 0).sum()}/{len(yr)}", "每年": " ".join(f"{y % 100}:{v:+.1%}" for y, v in yr.items()),
                     "隔天買不到": blocked, "買得到的平均跳空": g})
t = pd.DataFrame(rows)
out = ROOT / "results" / "消息面族群"
out.mkdir(parents=True, exist_ok=True)
t.to_csv(out / "group_limitup.csv", index=False, encoding="utf-8-sig")
order = list(sig)
lines = [f"{days[0]:%Y-%m-%d} ~ {days[-1]:%Y-%m-%d}，成本 {COST:.1%}；族群漲停日共 {int(grp.sum().sum())} 個（產業×日），其中 5 日內第一次 {int(first_grp.sum().sum())} 個"]
for title, col, fmt in (("比隨機（每筆超額）", "比隨機", lambda v: f"{v:+.1%}"), ("平均報酬（扣成本）", "平均", lambda v: f"{v:+.1%}"),
                        ("勝率", "勝率", lambda v: f"{v:.0%}"), ("贏的年數", "贏的年數", str), ("筆數", "筆數", lambda v: f"{v:.0f}")):
    pv = t.pivot(index="策略", columns="天數", values=col).reindex([o for o in order if o in set(t["策略"])])
    lines += ["", f"== {title}", pv.map(lambda v: fmt(v) if pd.notna(v) else "").to_string()]
lines += ["", "== 隔天開盤"]
for _, r in t[t["天數"] == 5].iterrows():
    lines.append(f"{r['策略']}：隔天開盤就漲停買不到的 {r['隔天買不到']:.0%}；買得到的平均開盤已經跳空 {r['買得到的平均跳空']:+.1%}；5 天逐年 {r['每年']}")
text = "\n".join(lines)
print(text)
(out / "group_limitup.txt").write_text(text, encoding="utf-8")
