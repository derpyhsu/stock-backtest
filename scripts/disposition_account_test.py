"""處置股「處置期間抱著」做成帳戶回測＋穩健度（disposition_test.py 的追加，2026-11-26）。

規則：公告處置（收盤後公告）→ 處置第一天開盤買，處置最後一天收盤賣（約 5～10 個交易日）。
成交限制：開盤已漲停（>= 昨收 +9.5%）買不到就放棄；出場日收盤跌停賣不掉 → 隔天開盤賣。
流動性：處置前 20 日均量 >= 500 張、股價 >= 10。成本：買 0.1425%、賣 0.1425%＋0.3%。
帳戶：100 萬，每筆固定淨值 5%（最多同時 20 檔），現金不夠就不買；另測加 -10% 停損（收盤跌破隔天開盤賣）。
穩健度：晚一點買（首日收盤、第二天開盤）、2026-08-10 規則改後、上市／上櫃分開。
    python scripts/disposition_account_test.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from twscreener.market import load_market_index, load_panel  # noqa: E402
FEE, TAX = 0.001425, 0.003
p = load_panel()
piv = lambda k: p.pivot_table(index="date", columns="stock_id", values=k)
raw, ro, vol = piv("close"), piv("open"), piv("vol")
days = raw.index
ex = pd.read_csv(ROOT / "data/exrights.csv", dtype={"stock_id": str}, parse_dates=["date"])
ex = ex[ex["date"].isin(days) & ex["stock_id"].isin(raw.columns) & (ex["ref"] > 0)]
ret = raw.pct_change(fill_method=None)
for r in ex.itertuples():
    c = raw.at[r.date, r.stock_id]
    if np.isfinite(c):
        ret.at[r.date, r.stock_id] = c / r.ref - 1
chg = ret.copy()
ret = ret.where(ret.abs() <= 0.11, 0.0).fillna(0)
tri = (1 + ret).cumprod().where(raw.notna())
tro = tri / raw * ro
liquid = (vol.rolling(20).mean() >= 500) & (raw >= 10)
gap = ro / raw.shift(1) - 1
T, O, L, G, C = tri.values, tro.values, liquid.values, gap.values, chg.values
col = {s: j for j, s in enumerate(raw.columns)}
d = pd.read_csv(ROOT / "data/disposition.csv", dtype={"stock_id": str}, parse_dates=["announce", "start", "end"])
d = d[d["stock_id"].isin(raw.columns) & d["stock_id"].str.fullmatch(r"\d{4}")]
d["again"] = d["measure"].str.contains("再次|第二次")
idx = pd.Series(np.arange(len(days)), index=days)
d["i_s"] = idx.reindex(d["start"]).values
d["i_e"] = np.searchsorted(days.values, d["end"].values, side="right") - 1
d = d.dropna(subset=["i_s"])
d["i_s"] = d["i_s"].astype(int)
d = d[(d["i_s"] > 21) & (d["i_e"] >= d["i_s"])]
d["j"] = d["stock_id"].map(col)
d = d[[L[i - 1, j] for i, j in zip(d["i_s"], d["j"])]]
d = d.sort_values(["i_s", "stock_id"]).drop_duplicates(["i_s", "stock_id"])
NET = (1 - FEE - TAX) / (1 + FEE)
lines = [f"流動性夠的處置事件 {len(d)} 筆（第一次 {(~d['again']).sum()}、再次 {d['again'].sum()}）；處置天數中位數 {(d['i_e'] - d['i_s'] + 1).median():.0f} 個交易日"]


def trade(r, mode="open", stop=None):
    """回傳 (進場 index, 出場 index, 報酬)；買不到回傳 None。"""
    i, j, e = r.i_s, r.j, r.i_e
    if mode == "open":
        if not np.isfinite(O[i, j]) or G[i, j] >= 0.095:
            return None
        i0, px = i, O[i, j]
    elif mode == "close1":
        if not np.isfinite(T[i, j]) or C[i, j] >= 0.095:
            return None
        i0, px = i, T[i, j]
    else:  # open2
        if i + 1 > e or not np.isfinite(O[i + 1, j]) or G[i + 1, j] >= 0.095:
            return None
        i0, px = i + 1, O[i + 1, j]
    k = e
    if stop:
        for q in range(i0, e):
            if T[q, j] / px - 1 <= stop:
                k = q
                break
    if k >= len(days) - 1:
        return None
    if k < e or C[k, j] <= -0.095:      # 停損或出場日跌停：隔天開盤賣
        out, k1 = O[k + 1, j], k + 1
    else:
        out, k1 = T[k, j], k
    if not np.isfinite(out):
        return None
    return i0, k1, out / px * NET - 1, px


def table(name, g, **kw):
    t = [trade(r, **kw) for r in g.itertuples()]
    t = pd.DataFrame([x[:3] for x in t if x], columns=["i0", "i1", "ret"])
    if len(t) < 30:
        return f"{name:<26} 筆數不足"
    t["d"] = days[t["i0"]]
    yr = t.groupby(t["d"].dt.year)["ret"].mean()
    return (f"{name:<26}{len(t):>6}{(t['ret'] > 0).mean():>6.0%}{t['ret'].mean():>+8.2%}{t['ret'].median():>+8.2%}"
            f"{t['ret'].quantile(0.05):>+8.1%}{t['ret'].min():>+8.1%}  " + " ".join(f"{y % 100}:{v:+.1%}" for y, v in yr.items()))


lines += ["", f"{'每筆（已扣成本）':<26}{'筆數':>6}{'勝率':>6}{'平均':>8}{'中位':>8}{'最差5%':>8}{'最差':>8}  每年平均"]
for lab, g in [("全部", d), ("第一次處置", d[~d["again"]]), ("再次處置", d[d["again"]]), ("上市", d[d["market"] == "上市"]), ("上櫃", d[d["market"] == "上櫃"]),
               ("2026-08-10 規則改後", d[d["start"] >= "2026-08-10"])]:
    lines.append(table(f"{lab}・首日開盤買", g))
lines.append(table("全部・首日收盤買", d, mode="close1"))
lines.append(table("全部・第二天開盤買", d, mode="open2"))
lines.append(table("全部・首日開盤買・停損 -10%", d, stop=-0.10))
lines.append(table("再次・首日開盤買・停損 -10%", d[d["again"]], stop=-0.10))


def account(g, w=0.05, max_n=20, **kw):
    tr = [(r, trade(r, **kw)) for r in g.itertuples()]
    by_in, by_out = {}, {}
    cash, pos, nav = 1.0, {}, []
    ev = {}
    for r, t in tr:
        if t:
            ev.setdefault(t[0], []).append((r.j, t))
    last = 1.0
    for i in range(int(d["i_s"].min()), len(days)):
        for key in [k for k, v in pos.items() if v[1] == i]:      # 出場（收盤或開盤，都算當天）
            amt, _, rr, _ = pos.pop(key)
            cash += amt * (1 + rr)
        for j, t in ev.get(i, []):
            if len(pos) >= max_n or (j, t[0]) in pos or any(k[0] == j for k in pos):
                continue
            amt = min(last * w, cash)
            if amt < last * w * 0.5:
                continue
            cash -= amt
            pos[(j, t[0])] = (amt, t[1], t[2], t[3])
        last = cash + sum(v[0] for v in pos.values())          # 部位大小用成本計
        mtm = cash + sum(v[0] * (T[i, k[0]] / v[3] if np.isfinite(T[i, k[0]]) else 1) for k, v in pos.items())   # 淨值每天用收盤價計
        nav.append(mtm)
    return pd.Series(nav, index=days[int(d["i_s"].min()):])


tx = load_market_index()["taiex"].reindex(days).ffill()
lines += ["", "== 帳戶回測（每筆淨值 5%、最多 20 檔；淨值每天用收盤價計）"]
navs = {}
for lab, g, kw in [("全部處置・每筆 2.5%", d, {"w": 0.025, "max_n": 20}), ("全部處置", d, {}), ("全部處置＋停損-10%", d, {"stop": -0.10}), ("只做再次處置", d[d["again"]], {}), ("全部・第二天開盤才買", d, {"mode": "open2"})]:
    n = account(g, **kw)
    navs[lab] = n
    yrs = (n.index[-1] - n.index[0]).days / 365.25
    m = n.resample("ME").last().pct_change().dropna()
    yr = n.groupby(n.index.year).apply(lambda s: s.iat[-1] / s.iat[0] - 1)
    lines.append(f"{lab:<16} 總報酬 {n.iat[-1] - 1:+.1%}（年化 {n.iat[-1] ** (1 / yrs) - 1:+.1%}）／回撤 {(n / n.cummax() - 1).min():.1%}／賺錢月 {(m > 0).mean():.0%}／最差月 {m.min():+.1%}／"
                 f"和大盤月相關 {m.corr(tx.resample('ME').last().pct_change()):.2f}  " + " ".join(f"{y}:{x:+.0%}" for y, x in yr.items()))
text = "\n".join(lines)
print(text)
pd.DataFrame(navs).to_csv(ROOT / "results/大盤以外/disposition_navs.csv")
(ROOT / "results/大盤以外/disposition_account.txt").write_text(text, encoding="utf-8")
