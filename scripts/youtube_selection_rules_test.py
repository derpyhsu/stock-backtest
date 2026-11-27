"""YouTube 選股教學影片裡講得出具體條件的規則，用全市場日線驗證（2026-11-27）。影片整理在專案外的 strategy/ 筆記。

做法同 handbook_untested_test.py：訊號日收盤判斷、隔天開盤買（開盤漲停買不到）、抱 N 個交易日收盤賣，扣 0.6%。
比隨機＝每天一票：同一天訊號平均減同一天所有流動性股票平均。流動性：20 日均量 >= 500 張、股價 >= 10。

一、價格動能（財報狗：台股過去 6~12 個月漲最多的，下個月沒有比較好；跌最多的繼續弱）
二、月營收（翁士峻：創新高還要季增、月增擴大、站上均線、投信買；阮慕驊：連續創新高）
三、投信（永豐張力允：前兩天賣超、今天買超超過前兩天賣的＋收紅＋收 5 日最高；愛德恩：投信連買＋40 日突破）
四、技術面（永豐臧家治的技術面那一半：量 >= 5000 張＋收盤穿過布林上緣＋高於季線 10~15%；雷老闆：盤整後第一根帶量長紅）
五、一年新高的出場方式（阿魯米：停損 8~10%、4 天沒再創高就走），用 results/market_signals.csv 的 S1 訊號逐筆模擬。
沒有季報資料，所以「毛利率、營益率、EPS 連續成長／創四季新高」測不了。
    python scripts/youtube_selection_rules_test.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from twscreener.fundamentals import load_revenue, revenue_features  # noqa: E402
from twscreener.market import load_panel  # noqa: E402
from twscreener.simulate import prices  # noqa: E402

H = (5, 20, 40, 60, 120)
COST = 0.006
p = load_panel()
raw, adj, opn, vol = prices(p)


def piv(col):
    return p.pivot_table(index="date", columns="stock_id", values=col).reindex(index=raw.index, columns=raw.columns)


ro = piv("open")
days = raw.index
liquid = (vol.rolling(20).mean() >= 500) & (raw >= 10)
can_buy = ro.shift(-1) < raw * 1.095
entry = opn.shift(-1)
fwd = {h: adj.shift(-h) / entry - 1 - COST for h in H}
base = {h: fwd[h].where(liquid).mean(axis=1) for h in H}
c = adj
ma = {n: c.rolling(n).mean() for n in (5, 10, 20, 60, 120, 240)}
v20 = vol.rolling(20).mean()
recent = lambda s, n: s.astype(float).rolling(n).max().shift(1).fillna(0).astype(bool)  # noqa: E731
sig = {}

# 一、價格動能：每月第一個交易日，在流動性股票裡排名
first = pd.Series(days, index=days).groupby([days.year, days.month]).transform("min") == pd.Series(days, index=days)
mfirst = pd.DataFrame(np.repeat(first.values[:, None], raw.shape[1], 1), index=days, columns=raw.columns)
for name, m in (("12 個月（跳過最近 1 個月）", c.shift(20) / c.shift(250) - 1), ("6 個月", c / c.shift(120) - 1), ("1 個月", c / c.shift(20) - 1)):
    rk = m.where(liquid).rank(axis=1, pct=True)
    sig[f"動能 {name} 漲最多 10%"] = mfirst & (rk >= 0.9)
    sig[f"動能 {name} 跌最多 10%"] = mfirst & (rk <= 0.1)

# 二、月營收：公布後第一個交易日
rf = revenue_features(load_revenue())
g = rf.groupby("stock_id")
n = g.cumcount() + 1
rf["hi_all"] = (rf["rev"] >= g["rev"].cummax()) & (n >= 24)            # 資料從 2021-01，至少 24 個月才算「歷史新高」
rf["hi_prev"] = g["hi_all"].shift(1).fillna(False).astype(bool)
rf["qoq"] = rf["rev3"] / g["rev3"].shift(3) - 1
rf["mom"] = rf["rev"] / g["rev"].shift(1) - 1
rf["mom_up"] = rf["mom"] > g["mom"].shift(1)
rf["day"] = rf["available"].map(lambda d: days[days.searchsorted(d)] if days.searchsorted(d) < len(days) else pd.NaT)
rf = rf.dropna(subset=["day"])


def rev_sig(mask):
    x = rf[mask & rf["stock_id"].isin(raw.columns)]
    a = np.zeros(raw.shape, bool)
    a[days.get_indexer(x["day"]), raw.columns.get_indexer(x["stock_id"])] = True
    return pd.DataFrame(a, index=days, columns=raw.columns)


tn = piv("trust_net").fillna(0)
fnet = piv("foreign_net").fillna(0)
t20 = tn.rolling(20).sum()
i20 = t20 + fnet.rolling(20).sum()
above_all = (c > ma[5]) & (c > ma[10]) & (c > ma[20]) & (c > ma[60])
hi = rev_sig(rf["hi_all"])
sig["營收創歷史新高"] = hi
sig["營收創歷史新高＋連續第 2 個月以上（阮慕驊）"] = rev_sig(rf["hi_all"] & rf["hi_prev"])
sig["營收創歷史新高＋第一次（前一個月不是）"] = rev_sig(rf["hi_all"] & ~rf["hi_prev"])
sig["營收創歷史新高＋近 3 月比前 3 月多 10% 以上"] = rev_sig(rf["hi_all"] & (rf["qoq"] >= 0.10))
sig["營收創歷史新高＋月增擴大"] = rev_sig(rf["hi_all"] & rf["mom_up"])
sig["營收創歷史新高＋站上 5/10/20/60 日線"] = hi & above_all
sig["營收創歷史新高＋投信 20 日買超"] = hi & (t20 > 0)
good = rev_sig(rf["hi_all"] & (rf["qoq"] >= 0.10) & rf["mom_up"])
sig["翁士峻全套（新高＋季增＋月增擴大＋站上均線＋投信買）"] = good & above_all & (t20 > 0)
sig["翁士峻避開（新高但法人 20 日賣超、跌破季線）"] = hi & (i20 < 0) & (c < ma[60])

# 現行營收加速（twscreener/pead.py）當對照，再看它和「連續創歷史新高」重疊與否
from twscreener import pead  # noqa: E402
from twscreener.market import load_market_index  # noqa: E402

a = np.zeros(raw.shape, bool)
for d, ser in pead.events(raw, adj, vol, load_market_index()["taiex"]).items():
    a[days.get_indexer([d])[0], raw.columns.get_indexer(ser.index)] = True
pe = pd.DataFrame(a, index=days, columns=raw.columns)
streak = rev_sig(rf["hi_all"] & rf["hi_prev"])
sig["現行營收加速（對照）"] = pe
sig["現行營收加速＋連續第 2 個月以上歷史新高"] = pe & streak
sig["現行營收加速，但不是連續歷史新高"] = pe & ~streak
sig["連續歷史新高，但不在現行營收加速裡"] = streak & ~pe

# 三、投信
red = raw > ro
top5 = c >= c.rolling(5).max()
turn = (tn.shift(1) < 0) & (tn.shift(2) < 0) & (tn > -(tn.shift(1) + tn.shift(2)))
mkt = p.drop_duplicates("stock_id", keep="last").set_index("stock_id")["market"].reindex(raw.columns)
listed = pd.DataFrame(np.repeat((mkt == "上市").values[None, :], len(days), 0), index=days, columns=raw.columns)
sig["投信賣轉買＋收紅＋收 5 日最高（張力允）"] = turn & red & top5
sig["投信賣轉買＋收紅＋收 5 日最高，只看上市"] = turn & red & top5 & listed
sig["投信賣轉買（不加 K 線條件）"] = turn
buy3 = (tn > 0) & (tn.shift(1) > 0) & (tn.shift(2) > 0)
brk40 = c > c.shift(1).rolling(40).max()
sig["投信連 3 買＋40 日突破（愛德恩）"] = buy3 & brk40 & ~recent(buy3 & brk40, 10)
sig["只有 40 日突破（對照）"] = brk40 & ~recent(brk40, 10)

# 四、技術面
sd = c.rolling(20).std(ddof=0)
upper = ma[20] + 2 * sd
cross = (c > upper) & (c.shift(1) <= upper.shift(1))
gap60 = c / ma[60] - 1
sig["量 5000 張＋穿過布林上緣＋高於季線 10~15%（臧家治技術面）"] = (vol >= 5000) & cross & (gap60 >= 0.10) & (gap60 <= 0.15)
sig["量 5000 張＋穿過布林上緣（不限季線距離）"] = (vol >= 5000) & cross
rng60 = c.shift(1).rolling(60).max() / c.shift(1).rolling(60).min() - 1
bigred = (raw / ro - 1 >= 0.04) & (vol >= 2 * v20.shift(1))
brk60 = c > c.shift(1).rolling(60).max()
first_red = bigred & brk60 & (rng60 <= 0.25)
sig["盤整 60 日（高低差 25% 內）後第一根帶量長紅突破（雷老闆）"] = first_red & ~recent(first_red, 20)
gapv = (ro / raw.shift(1) - 1 >= 0.03) & (vol >= 3 * v20.shift(1))
hi250 = c >= c.rolling(250).max()
fh = hi250 & ~recent(hi250, 20)
sig["一年新高（20 日內第一次）"] = fh
sig["一年新高，當天跳空 3% 以上且量 3 倍（阿魯米說要避開）"] = fh & gapv
sig["一年新高，當天不是跳空爆量"] = fh & ~gapv

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
        yr = x.groupby(x.index.year).mean()
        rows.append({"策略": name, "天數": h, "筆數": len(st), "勝率": (st > 0).mean(), "平均": st.mean(), "中位數": st.median(),
                     "比隨機": x.mean(), "前半": x[x.index < half].mean(), "後半": x[x.index >= half].mean(),
                     "贏的年數": f"{(yr > 0).sum()}/{len(yr)}", "每年": " ".join(f"{y % 100}:{v:+.1%}" for y, v in yr.items())})
t = pd.DataFrame(rows)
out = ROOT / "results" / "選股影片規則"
out.mkdir(parents=True, exist_ok=True)
t.to_csv(out / "youtube_selection.csv", index=False, encoding="utf-8-sig")
order = list(sig)
lines = [f"{days[0]:%Y-%m-%d} ~ {days[-1]:%Y-%m-%d}，成本 {COST:.1%}；比隨機＝每天一票的平均超額；隨機 60 天平均 {base[60].mean():+.1%}"]
for title, col, fmt in (("比隨機（每筆超額）", "比隨機", lambda v: f"{v:+.1%}"), ("筆數", "筆數", lambda v: f"{v:.0f}"),
                        ("勝率", "勝率", lambda v: f"{v:.0%}"), ("中位數（扣成本，沒減隨機）", "中位數", lambda v: f"{v:+.1%}"), ("贏的年數", "贏的年數", str)):
    pv = t.pivot(index="策略", columns="天數", values=col).reindex([o for o in order if o in set(t["策略"])])
    lines += ["", f"== {title}", pv.map(lambda v: fmt(v) if pd.notna(v) else "").to_string()]
lines += ["", "== 60 天逐年超額"]
for _, r in t[t["天數"] == 60].iterrows():
    lines.append(f"{r['策略']}：{r['每年']}（前半 {r['前半']:+.1%}、後半 {r['後半']:+.1%}）")

# 五、一年新高的出場方式（逐筆，收盤判斷、隔天開盤賣）
s1 = pd.read_csv(ROOT / "results" / "market_signals.csv", dtype={"stock_id": str})
s1 = s1[s1["strategy"] == "S1_一年新高"]
A, O = adj.values, opn.values
di, ci = days.get_indexer(pd.to_datetime(s1["signal_date"])), raw.columns.get_indexer(s1["stock_id"])


def trade(i, j, stop, stale, maxd=60, trail=None):
    e = O[i + 1, j]
    if not np.isfinite(e):
        return None
    peak, last_high = e, i + 1
    for k in range(i + 1, min(i + 1 + maxd, len(days) - 1)):
        x = A[k, j]
        if not np.isfinite(x):
            continue
        if x > peak:
            peak, last_high = x, k
        if x <= e * (1 - stop) or (stale and k - last_high >= stale) or (trail and x <= peak * (1 - trail)):
            nx = O[k + 1, j]
            return (nx if np.isfinite(nx) else x) / e - 1 - COST, k + 1 - i
    k = min(i + maxd, len(days) - 1)
    x = A[k, j]
    return (x / e - 1 - COST, k - i) if np.isfinite(x) else None


variants = [("現在：停損 -15%、60 天", .15, 0, None), ("停損 -10%", .10, 0, None), ("停損 -8%", .08, 0, None),
            ("停損 -15%＋4 天沒創高就走", .15, 4, None), ("停損 -15%＋10 天沒創高就走", .15, 10, None),
            ("停損 -15%＋20 天沒創高就走", .15, 20, None), ("停損 -10%＋4 天沒創高就走（阿魯米）", .10, 4, None),
            ("停損 -15%＋從最高收盤回落 15% 就走", .15, 0, .15)]
lines += ["", f"== 一年新高（S1 訊號 {len(s1)} 筆，只算抱得滿 60 天的）不同出場方式，逐筆、扣成本"]
ok = (di >= 0) & (ci >= 0) & (di + 61 < len(days))
for name, stop, stale, trail in variants:
    r = [trade(i, j, stop, stale, 60, trail) for i, j in zip(di[ok], ci[ok])]
    r = pd.DataFrame([x for x in r if x], columns=["ret", "d"])
    lines.append(f"{name}：{len(r)} 筆、勝率 {(r.ret > 0).mean():.0%}、平均 {r.ret.mean():+.1%}、中位數 {r.ret.median():+.1%}、"
                 f"平均抱 {r.d.mean():.0f} 天、每抱一天 {r.ret.sum() / r.d.sum():+.2%}、賺最多的一成平均 {r.ret.nlargest(max(len(r) // 10, 1)).mean():+.0%}")
text = "\n".join(lines)
print(text)
(out / "youtube_selection.txt").write_text(text, encoding="utf-8")
