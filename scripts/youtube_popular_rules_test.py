"""YouTube 高觀看數投資影片裡能量化的規則，用全市場日線驗證（2026-11-27）。影片整理在專案外的 strategy/ 筆記。

做法同 youtube_selection_rules_test.py：訊號日收盤判斷、隔天開盤買（開盤漲停買不到）、抱 N 個交易日收盤賣，扣 0.6%。
比隨機＝每天一票：同一天訊號平均減同一天所有流動性股票平均。流動性：20 日均量 >= 500 張、股價 >= 10。

  J Law：200 日線上＋10 日線 > 20 日線＋近 1（、6）個月漲幅是大盤 2 倍；Weinstein 30 週均線向上的突破；窄幅整理突破
  LEI：均線密集（20/60/120 相距 2% 內）後轉多頭排列；多頭排列中第一次回到 20、60 日線
  麥克連：外資投信同買第一天＋在 20 日線附近＋隔天上漲
  權證小哥：布林帶寬 5% 以下＋收盤突破上軌
  犀利媽：5 日 RSI < 10；黃世聰：5/10/20/60 均線剛全部向上排列
  出場（CMoney、麥克連、權證小哥）：ATR 移動停利、回落 10%、跌破 20 日線、漲 50% 賣一半，套在 S1 一年新高訊號上
    python scripts/youtube_popular_rules_test.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from twscreener.market import load_market_index, load_panel  # noqa: E402
from twscreener.simulate import prices  # noqa: E402

H = (5, 10, 20, 60, 120)
COST = 0.006
p = load_panel()
raw, adj, opn, vol = prices(p)
days = raw.index


def piv(col):
    return p.pivot_table(index="date", columns="stock_id", values=col).reindex(index=days, columns=raw.columns)


ro, rh, rl = piv("open"), piv("high"), piv("low")
tn, fnet = piv("trust_net").fillna(0), piv("foreign_net").fillna(0)
liquid = (vol.rolling(20).mean() >= 500) & (raw >= 10)
can_buy = ro.shift(-1) < raw * 1.095
entry = opn.shift(-1)
fwd = {h: adj.shift(-h) / entry - 1 - COST for h in H}
base = {h: fwd[h].where(liquid).mean(axis=1) for h in H}
c = adj
ma = {n: c.rolling(n).mean() for n in (5, 10, 20, 50, 60, 120, 150, 200)}
recent = lambda s, n: s.astype(float).rolling(n).max().shift(1).fillna(0).astype(bool)  # noqa: E731
first = lambda s, n: s & ~recent(s, n)  # noqa: E731
tx = load_market_index()["taiex"].reindex(days).ffill()
sig = {}

# J Law 強勢股篩選
r20, r120 = c / c.shift(20) - 1, c / c.shift(120) - 1
t20, t120 = tx / tx.shift(20) - 1, tx / tx.shift(120) - 1
trend = (c > ma[200]) & (ma[10] > ma[20])
s1m = r20.gt(0) & r20.ge(2 * t20, axis=0)
s6m = r120.gt(0) & r120.ge(2 * t120, axis=0)
sig["J Law：200 日線上＋10 日線>20 日線（剛符合）"] = first(trend, 20)
sig["J Law：再加近 1 個月漲幅是大盤 2 倍"] = first(trend & s1m, 20)
sig["J Law：再加近 1、6 個月漲幅都是大盤 2 倍"] = first(trend & s1m & s6m, 20)

# Weinstein 第二階段：30 週（150 日）均線向上時突破 120 日高點
brk120 = c > c.shift(1).rolling(120).max()
rng120 = c.shift(1).rolling(120).max() / c.shift(1).rolling(120).min() - 1
up150 = ma[150] > ma[150].shift(20)
sig["Weinstein：150 日線向上＋突破 120 日高"] = first(brk120 & up150, 20)
sig["Weinstein：同上＋之前 120 日高低差 30% 內（打底）"] = first(brk120 & up150 & (rng120 <= 0.30), 20)
sig["對照：150 日線向下時突破 120 日高"] = first(brk120 & ~up150, 20)

# J Law Swing：上升趨勢＋前 5 天高低差 5% 內＋收盤突破
tight = rh.shift(1).rolling(5).max() / rl.shift(1).rolling(5).min() - 1
sw = (raw > rh.shift(1).rolling(5).max()) & (tight <= 0.05) & (c > ma[200]) & (ma[50] > ma[50].shift(10))
sig["Swing：上升趨勢＋5 天窄幅（5% 內）突破"] = sw
sig["對照：5 天窄幅突破，不看趨勢"] = (raw > rh.shift(1).rolling(5).max()) & (tight <= 0.05)

# LEI 均線密集後多頭排列、第一次回測均線
mx = pd.concat([ma[20], ma[60], ma[120]]).groupby(level=0).max()
mn = pd.concat([ma[20], ma[60], ma[120]]).groupby(level=0).min()
dense = (mx / mn - 1 <= 0.02)
bull = (ma[20] > ma[60]) & (ma[60] > ma[120])
sig["LEI：均線密集（2% 內）後剛轉多頭排列"] = first(bull, 5) & recent(dense, 10)
sig["LEI：同上＋之前 120 日高低差 30% 內"] = first(bull, 5) & recent(dense, 10) & (rng120 <= 0.30)
sig["對照：剛轉多頭排列，之前沒有密集"] = first(bull, 5) & ~recent(dense, 10)
held = bull.astype(float).rolling(20).min().fillna(0).astype(bool) & (ma[60] > ma[60].shift(5))
t20_ = held & (rl <= raw / c * ma[20] * 1.01) & (c > ma[20])
t60_ = held & (rl <= raw / c * ma[60] * 1.01) & (c > ma[60])
sig["LEI：多頭排列中回到 20 日線收在線上（20 日內第一次）"] = first(t20_, 20)
sig["LEI：多頭排列中回到 60 日線收在線上（20 日內第一次）"] = first(t60_, 20)

# 麥克連：外資投信同買第一天＋20 日線附近＋隔天上漲
both = (fnet > 0) & (tn > 0)
start = both & ~recent(both, 5) & ((c / ma[20] - 1).abs() <= 0.03)
sig["麥克連：外資投信同買第一天＋20 日線附近"] = start
sig["麥克連：同上＋隔天上漲才進場"] = start.shift(1).fillna(False) & (c > c.shift(1))
sig["對照：外資投信同買第一天（不看位置）"] = both & ~recent(both, 5)

# 權證小哥：布林帶寬窄＋突破上軌
sd = c.rolling(20).std(ddof=0)
upper, lower = ma[20] + 2 * sd, ma[20] - 2 * sd
bw = (upper / lower - 1).shift(1)
out_up = (c > upper) & (c.shift(1) <= upper.shift(1))
sig["權證小哥：帶寬 5% 內＋突破布林上軌"] = out_up & (bw <= 0.05)
sig["權證小哥：帶寬 5~10%＋突破布林上軌"] = out_up & (bw > 0.05) & (bw <= 0.10)
sig["對照：帶寬 20% 以上時突破布林上軌"] = out_up & (bw >= 0.20)
sig["權證小哥：帶寬 5% 內＋突破上軌＋當天外資投信合計買超"] = out_up & (bw <= 0.05) & (fnet + tn > 0)

# 犀利媽：RSI
d = c.diff()
rsi = lambda n: 100 - 100 / (1 + d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean() / (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean())  # noqa: E731
r5, r10 = rsi(5), rsi(10)
sig["犀利媽：5 日 RSI < 10"] = first(r5 < 10, 5)
sig["犀利媽：5 日 RSI < 10 且在 200 日線上"] = first(r5 < 10, 5) & (c > ma[200])
sig["犀利媽說不追的：10 日 RSI >= 95"] = first(r10 >= 95, 5)

# 黃世聰：5/10/20/60 均線剛全部向上且多頭排列
fan = (ma[5] > ma[10]) & (ma[10] > ma[20]) & (ma[20] > ma[60])
for n in (5, 10, 20, 60):
    fan &= ma[n] > ma[n].shift(1)
sig["黃世聰：5/10/20/60 均線剛全部向上排列"] = first(fan, 10)

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
        rows.append({"策略": name, "天數": h, "筆數": len(st), "勝率": (st > 0).mean(), "中位數": st.median(), "比隨機": x.mean(),
                     "前半": x[x.index < half].mean(), "後半": x[x.index >= half].mean(),
                     "贏的年數": f"{(yr > 0).sum()}/{len(yr)}", "每年": " ".join(f"{y % 100}:{v:+.1%}" for y, v in yr.items())})
t = pd.DataFrame(rows)
order = list(sig)
lines = [f"{days[0]:%Y-%m-%d} ~ {days[-1]:%Y-%m-%d}，成本 {COST:.1%}；比隨機＝每天一票的平均超額；隨機 60 天平均 {base[60].mean():+.1%}"]
for title, col, fmt in (("比隨機（每筆超額）", "比隨機", lambda v: f"{v:+.1%}"), ("筆數", "筆數", lambda v: f"{v:.0f}"),
                        ("勝率", "勝率", lambda v: f"{v:.0%}"), ("贏的年數", "贏的年數", str)):
    pv = t.pivot(index="策略", columns="天數", values=col).reindex([o for o in order if o in set(t["策略"])])
    lines += ["", f"== {title}", pv.map(lambda v: fmt(v) if pd.notna(v) else "").to_string()]
lines += ["", "== 60 天逐年超額"]
for _, r in t[t["天數"] == 60].iterrows():
    lines.append(f"{r['策略']}：{r['每年']}（前半 {r['前半']:+.1%}、後半 {r['後半']:+.1%}）")

# 出場方式，套在 S1 一年新高訊號（收盤判斷、隔天開盤賣；最多 60 天）
s1 = pd.read_csv(ROOT / "results" / "market_signals.csv", dtype={"stock_id": str})
s1 = s1[s1["strategy"] == "S1_一年新高"]
A, O = adj.values, opn.values
pc = raw.shift(1)
tr = pd.concat([rh - rl, (rh - pc).abs(), (rl - pc).abs()]).groupby(level=0).max()
ATR = (tr.rolling(14).mean() / raw).values
M20, M5 = ma[20].values, ma[5].values
di, ci = days.get_indexer(pd.to_datetime(s1["signal_date"])), raw.columns.get_indexer(s1["stock_id"])


def trade(i, j, mode, maxd=60):
    e = O[i + 1, j]
    if not np.isfinite(e):
        return None
    peak, got, left = e, 0.0, 1.0
    for k in range(i + 1, min(i + 1 + maxd, len(days) - 1)):
        x = A[k, j]
        if not np.isfinite(x):
            continue
        peak = max(peak, x)
        nx = O[k + 1, j] if np.isfinite(O[k + 1, j]) else x
        if mode == "half50" and left == 1.0 and x >= e * 1.5:
            got, left = 0.5 * (nx / e - 1), 0.5
        hit = x <= e * 0.85
        if mode == "atr3":
            hit |= np.isfinite(ATR[k, j]) and x <= peak * (1 - 3 * ATR[k, j])
        elif mode == "atr2":
            hit |= np.isfinite(ATR[k, j]) and x <= peak * (1 - 2 * ATR[k, j])
        elif mode == "trail10":
            hit |= x <= peak * 0.90
        elif mode == "ma20":
            hit |= x < M20[k, j]
        elif mode == "ma5far":
            hit |= (peak >= e * 1.10) and x < M5[k, j]
        if hit:
            return got + left * (nx / e - 1) - COST, k + 1 - i
    k = min(i + maxd, len(days) - 1)
    x = A[k, j]
    return (got + left * (x / e - 1) - COST, k - i) if np.isfinite(x) else None


ok = (di >= 0) & (ci >= 0) & (di + 61 < len(days))
lines += ["", f"== 一年新高（S1 訊號，只算抱得滿 60 天的）不同出場方式，逐筆、扣成本"]
for name, mode in (("現在：停損 -15%、60 天", "base"), ("加：從最高收盤回落 3 倍 ATR 就走（CMoney）", "atr3"), ("加：回落 2 倍 ATR 就走", "atr2"),
                   ("加：從最高收盤回落 10% 就走（CMoney）", "trail10"), ("加：收盤跌破 20 日線就走（麥克連）", "ma20"),
                   ("加：漲過 10% 後跌破 5 日線就走（麥克連、布魯斯）", "ma5far"), ("加：漲 50% 先賣一半（CMoney）", "half50")):
    res = [trade(i, j, mode) for i, j in zip(di[ok], ci[ok])]
    res = pd.DataFrame([r for r in res if r], columns=["r", "d"])
    top = res["r"].nlargest(len(res) // 10).mean()
    lines.append(f"{name}：{len(res)} 筆、勝率 {(res['r'] > 0).mean():.0%}、平均 {res['r'].mean():+.1%}、中位 {res['r'].median():+.1%}、"
                 f"平均抱 {res['d'].mean():.0f} 天、每天 {res['r'].mean() / res['d'].mean():+.2%}、賺最多的一成平均 {top:+.0%}")
text = "\n".join(lines)
print(text)
out = ROOT / "results" / "選股影片規則"
t.to_csv(out / "popular_rules.csv", index=False, encoding="utf-8-sig")
(out / "popular_rules.txt").write_text(text, encoding="utf-8")
