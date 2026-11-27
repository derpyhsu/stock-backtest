"""沒字幕的高觀看數影片（用 AI Studio 讀的 34 支）收斂出的日線規則，用全市場日線驗證（2026-11-27）。筆記在專案外的 strategy/。

做法同 youtube_popular_rules_test.py：訊號日收盤判斷、隔天開盤買（開盤漲停買不到）、抱 N 個交易日收盤賣，扣 0.6%。
比隨機＝每天一票：同一天訊號平均減同一天所有流動性股票平均。流動性：20 日均量 >= 500 張、股價 >= 10。

  朱家泓：關鍵紅 K（收盤過前一天高、量 > 昨天 1.3 倍、實體 > 2%）；回後買上漲；盤整突破；盤整 2 個月後爆量長紅；一字底；低檔爆大量後過高
  喬哥：爆量長紅隔天守住那根的 1/2 才買
  阿魯米：大盤大跌後最先漲回「大盤起跌點」的股票
  阿水：布林（20, 2.1σ）變窄＋量 2~10 倍＋站上上軌＋大盤月線上揚
  出場：朱家泓的跌破 5 日線、跌破月線、進場紅 K 低點停損，套在「回後買上漲」訊號上和抱 60 天比
    python scripts/youtube_nocc_rules_test.py
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
f = adj / raw
c, ao, ah, al = adj, ro * f, rh * f, rl * f
liquid = (vol.rolling(20).mean() >= 500) & (raw >= 10)
can_buy = ro.shift(-1) < raw * 1.095
entry = opn.shift(-1)
fwd = {h: adj.shift(-h) / entry - 1 - COST for h in H}
base = {h: fwd[h].where(liquid).mean(axis=1) for h in H}
ma = {n: c.rolling(n).mean() for n in (5, 10, 20, 60)}
recent = lambda s, n: s.astype(float).rolling(n).max().shift(1).fillna(0).astype(bool)  # noqa: E731
first = lambda s, n: s & ~recent(s, n)  # noqa: E731
tx = load_market_index()["taiex"].reindex(days).ffill()
v5 = vol.shift(1).rolling(5).mean()
sig = {}

# 朱家泓：關鍵紅 K、多頭、回後買上漲、盤整突破
body = c / ao - 1
key = (c > ah.shift(1)) & (vol > vol.shift(1) * 1.3) & (body > 0.02)
red_over = (c > ah.shift(1)) & (c > ao)
bull = (c > ma[20]) & (ma[20] > ma[20].shift(5))
stack = (ma[5] > ma[10]) & (ma[10] > ma[20])
hi10 = c.shift(1).rolling(10).max()
pulled = c.shift(1) <= hi10 * 0.97                      # 昨天比近 10 天高點低 3% 以上＝有回檔
hold_low = al.shift(1).rolling(5).min() > al.shift(6).rolling(15).min()   # 這次回檔沒破前一段低點
back = bull & pulled & hold_low
sig["對照：關鍵紅 K（不看位置）"] = key
sig["朱家泓：月線上揚＋關鍵紅 K"] = key & bull
sig["朱家泓：回後買上漲（回檔不破前低＋紅 K 收盤過昨高）"] = back & red_over
sig["朱家泓：回後買上漲＋關鍵紅 K（量 1.3 倍、實體 2%）"] = back & key
sig["朱家泓：回後買上漲＋關鍵紅 K＋三線多排"] = back & key & stack
sig["朱家泓：回後買上漲＋關鍵紅 K＋三線多排＋季線上揚"] = back & key & stack & (ma[60] > ma[60].shift(5)) & (ma[20] > ma[60])
rng20 = ah.shift(1).rolling(20).max() / al.shift(1).rolling(20).min() - 1
brk20 = c > ah.shift(1).rolling(20).max()
sig["朱家泓：盤整突破（20 天高低 15% 內）＋關鍵紅 K＋月線上揚"] = brk20 & (rng20 <= 0.15) & key & (ma[20] > ma[20].shift(5))
rng40 = ah.shift(1).rolling(40).max() / al.shift(1).rolling(40).min() - 1
long_red = (body >= 0.04) & (vol >= vol.shift(1) * 2)
sig["朱家泓：盤整 2 個月（40 天高低 20% 內）＋爆量長紅突破＋三線多排"] = (c > ah.shift(1).rolling(40).max()) & (rng40 <= 0.20) & long_red & stack
sig["朱家泓：一字底（40 天高低 10% 內）＋量 2 倍紅 K 突破"] = (c > ah.shift(1).rolling(40).max()) & (rng40 <= 0.10) & (vol >= v5 * 2) & (c > ao)
# 低檔爆大量：空頭中連跌後爆量 K，2 天內收盤過那根高點
down = (c < ma[20]) & (ma[20] < ma[20].shift(5)) & (c.shift(1) < c.shift(4))
burst = down & (vol >= v5 * 2)
bh = ah.where(burst)
lowbuy = pd.DataFrame(False, index=days, columns=raw.columns)
for k in (1, 2):
    lowbuy |= burst.shift(k).fillna(False).astype(bool) & (c > bh.shift(k))
sig["朱家泓：空頭低檔爆大量後 2 天內收盤過那根高點"] = first(lowbuy, 3)

# 喬哥：爆量長紅隔天守住 1/2
big = (body >= 0.05) & (vol >= v5 * 2) & (c > ah.shift(1))
mid = (ao + c) / 2
big1 = big.shift(1).fillna(False).astype(bool)
sig["喬哥對照：爆量長紅當天收盤就決定買（隔天開盤追）"] = big
sig["喬哥：爆量長紅隔天收盤守住那根的 1/2 才買"] = big1 & (c >= mid.shift(1))
sig["喬哥：隔天守住 1/2 而且沒再大漲（漲幅 3% 內）"] = big1 & (c >= mid.shift(1)) & (c <= c.shift(1) * 1.03)
sig["喬哥對照：隔天收盤跌破那根的 1/2"] = big1 & (c < mid.shift(1))

# 阿魯米：大盤大跌後最先漲回大盤起跌點
W = 120
txv = tx.values
pk = np.array([i - W + 1 + int(np.argmax(txv[i - W + 1:i + 1])) if i >= W - 1 else -1 for i in range(len(days))])
cv = c.values
p0 = np.full(cv.shape, np.nan)
ok = pk >= 0
p0[ok] = cv[pk[ok]]
txpk = np.where(ok, txv[np.clip(pk, 0, None)], np.nan)
age = np.arange(len(days)) - pk
dd = pd.Series(txv / txpk - 1, index=days)
since_low = pd.Series([txv[pk[i]:i + 1].min() / txpk[i] - 1 if pk[i] >= 0 else np.nan for i in range(len(days))], index=days)
P0 = pd.DataFrame(p0, index=days, columns=raw.columns)
rec = c > P0
for name, cond in (("阿魯米：大盤從高點跌過 10% 且還差 5% 以上，個股最先漲回起跌點", (since_low <= -0.10) & (dd <= -0.05)),
                   ("阿魯米：同上，大盤跌過 15% 且還差 8% 以上", (since_low <= -0.15) & (dd <= -0.08))):
    m = rec.mul((cond & (age >= 10)).values, axis=0).astype(bool)
    sig[name] = first(m, 20)
    sig[name + "＋當天創 120 日新高"] = first(m, 20) & (c >= c.rolling(120).max())
sig["對照：大盤跌過 10% 且還差 5% 以上時，創 120 日新高"] = first((c >= c.rolling(120).max()).mul(((since_low <= -0.10) & (dd <= -0.05)).values, axis=0).astype(bool), 20)

# 阿水：布林 2.1σ 變窄＋量 2~10 倍＋站上上軌＋大盤月線上揚
sd = c.rolling(20).std()
up, lo = ma[20] + 2.1 * sd, ma[20] - 2.1 * sd
bw = up / lo - 1
txup = (tx.rolling(20).mean() > tx.rolling(20).mean().shift(5))
ashui = (bw.shift(1) <= 0.12) & (bw.shift(1) <= bw.shift(1).rolling(60).quantile(0.25)) & (vol >= v5 * 2) & (vol <= v5 * 10) & (c > ao) & (c > up)
sig["阿水：布林變窄＋量 2~10 倍＋紅 K 站上上軌"] = ashui
sig["阿水：同上＋大盤月線上揚"] = ashui.mul(txup.values, axis=0).astype(bool)

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

# 出場方式：套在「回後買上漲＋關鍵紅 K」訊號（收盤判斷、隔天開盤賣；最多 120 天）
src = (sig["朱家泓：回後買上漲＋關鍵紅 K（量 1.3 倍、實體 2%）"] & liquid & can_buy).fillna(False)
A, O, L = adj.values, opn.values, al.values
M5, M20 = ma[5].values, ma[20].values
ii, jj = np.where(src.values)


def trade(i, j, mode):
    e = O[i + 1, j]
    if not np.isfinite(e):
        return None
    stop_k = L[i, j]
    maxd = 60 if mode in ("hold60", "v5") else 120
    for k in range(i + 1, min(i + 1 + maxd, len(days) - 1)):
        x = A[k, j]
        if not np.isfinite(x):
            continue
        nx = O[k + 1, j] if np.isfinite(O[k + 1, j]) else x
        out = False
        if mode == "v5":
            out = x <= e * 0.85
        elif mode == "ma5":
            out = x < M5[k, j] or x < stop_k
        elif mode == "ma20":
            out = x < M20[k, j] or x < stop_k
        elif mode == "ma20_15":
            out = x < M20[k, j] or x <= e * 0.85
        if out:
            return nx / e - 1 - COST, k - i
    k = min(i + maxd, len(days) - 1)
    x = A[k, j]
    return (x / e - 1 - COST, k - i) if np.isfinite(x) else None


lines += ["", f"== 出場方式，套在「回後買上漲＋關鍵紅 K」{len(ii)} 筆訊號上"]
for mode, label in (("hold60", "抱 60 天（不停損）"), ("v5", "v5 的出場：收盤 -15% 停損，抱 60 天"), ("ma5", "朱家泓短線：跌破 5 日線或進場紅 K 低點"),
                    ("ma20", "朱家泓波段：跌破月線或進場紅 K 低點"), ("ma20_15", "跌破月線或 -15%")):
    r = [trade(i, j, mode) for i, j in zip(ii, jj) if i + 2 < len(days)]
    r = pd.DataFrame([x for x in r if x], columns=["ret", "d"])
    lines.append(f"{label:24s} {len(r)} 筆 勝率 {(r['ret'] > 0).mean():.0%} 平均 {r['ret'].mean():+.2%} 中位 {r['ret'].median():+.2%} 平均抱 {r['d'].mean():.0f} 天 每抱 1 天 {r['ret'].sum() / r['d'].sum():+.3%}")
text = "\n".join(lines)
print(text)
out = ROOT / "results" / "選股影片規則"
(out / "nocc_rules.txt").write_text(text, encoding="utf-8")
t.to_csv(out / "nocc_rules.csv", index=False)
