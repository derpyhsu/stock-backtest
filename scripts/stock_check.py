"""個股健檢：把一檔股票現在的狀態分組，找全市場歷史上「同組」的股票-日，看之後的表現，用規則給結論。
不看新聞、不靠主觀判斷。流程見 skill/stock-check/SKILL.md。

    python scripts/stock_check.py 2449            # 只看現況與相似情境
    python scripts/stock_check.py 2449 315        # 加上持有成本

分組（只用當天已知的資料）：
  回撤：還原收盤 / 120 日最高 - 1 →  >-10%、-10~-25%、-25~-45%、<=-45%
  趨勢：收盤在 20 日均線上／下
  營收：最新已公布月營收年增 → 衰退、0~20%、>=20%
  外資：近 20 日買賣超（佔 20 日均量）→ 買超／賣超（2024-06 起完整）
  融資：20 日增減 → 增／減
相似情境由寬到嚴列出：回撤＋趨勢 → ＋營收 → ＋外資 → ＋融資。每 5 個交易日取樣一次，降低重疊。
超額 = 減同日所有流動性合格股票的平均；「每月一票」= 先把同一個月的事件平均，再平均各月，避免單一崩盤月份灌票。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from twscreener.fundamentals import revenue_features  # noqa: E402
from twscreener.market import load_panel  # noqa: E402
from twscreener.simulate import prices  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
COST = 0.006
H = [20, 60, 120]
sid = sys.argv[1]
cost = float(sys.argv[2]) if len(sys.argv) > 2 else None

panel = load_panel()
raw, adj, opn, vol = prices(panel)
adj = adj.loc["2022-01-01":]
raw, vol = raw.reindex_like(adj), vol.reindex_like(adj)
pv = lambda k: panel.pivot_table(index="date", columns="stock_id", values=k).reindex_like(adj)
fn, mb = pv("foreign_net"), pv("margin_bal")
v20 = vol.rolling(20).mean()
liquid = (v20 >= 500) & (raw >= 10)

dd = adj / adj.rolling(120, min_periods=100).max() - 1
above = adj > adj.rolling(20).mean()
f20 = fn.rolling(20, min_periods=15).sum() / v20
m20 = mb / mb.shift(20) - 1
rev = revenue_features(pd.read_csv(ROOT / "data/revenue.csv", dtype={"stock_id": str}))
yoy = rev.pivot_table(index="available", columns="stock_id", values="yoy").sort_index()
yoy = yoy.reindex(adj.index.union(yoy.index)).ffill(limit=70).reindex(adj.index).reindex(columns=adj.columns)
fwd = {h: adj.shift(-h) / adj - 1 for h in H}
base = {h: fwd[h].where(liquid).mean(axis=1) for h in H}


def bucket(d, a, y, f, m):
    b = {"回撤": np.select([d > -0.10, d > -0.25, d > -0.45], [">-10%", "-10~-25%", "-25~-45%"], "<=-45%"),
         "趨勢": np.where(a, "月線上", "月線下"),
         "營收": np.select([y >= 0.2, y >= 0], [">=20%", "0~20%"], "衰退"),
         "外資": np.where(f > 0, "買超", "賣超"), "融資": np.where(m < 0, "減", "增")}
    b["營收"] = np.where(np.isnan(y), "無", b["營收"])
    b["外資"] = np.where(np.isnan(f), "無", b["外資"])
    b["融資"] = np.where(np.isnan(m), "無", b["融資"])
    return b


# 現況
last = adj[sid].last_valid_index()
cur_vals = dict(d=dd.at[last, sid], a=bool(above.at[last, sid]), y=yoy.at[last, sid], f=f20.at[last, sid], m=m20.at[last, sid])
cur = {k: v[0] for k, v in bucket(*[np.array([v]) for v in cur_vals.values()]).items()}
px = raw.at[last, sid]
print(f"# {sid} 個股健檢（{last.date()} 收盤 {px:g}）")
print(f"回撤 {cur_vals['d']:+.1%}（{cur['回撤']}）、{cur['趨勢']}（20 日均線 {raw[sid].rolling(20).mean().at[last]:.1f}）、"
      f"營收年增 {cur_vals['y']:+.1%}、外資 20 日 {cur_vals['f']:+.2f} 倍均量、融資 20 日 {cur_vals['m']:+.1%}")
if cost:
    print(f"持有成本 {cost:g}，目前 {px / cost - 1:+.1%}（成本不影響結論）")

# 歷史樣本（每 5 天取樣、要有 60 日結果）
idx = adj.index[::5]
rows = []
for d in idx:
    ok = liquid.loc[d] & dd.loc[d].notna() & fwd[60].loc[d].notna()
    s = ok[ok].index
    if len(s) == 0:
        continue
    b = bucket(dd.loc[d, s].values, above.loc[d, s].values, yoy.loc[d, s].values, f20.loc[d, s].values, m20.loc[d, s].values)
    df = pd.DataFrame(b, index=s)
    df["date"] = d
    for h in H:
        df[f"r{h}"] = fwd[h].loc[d, s].values - COST
        df[f"x{h}"] = fwd[h].loc[d, s].values - base[h].at[d]
    rows.append(df)
hist = pd.concat(rows)
hist["月"] = hist["date"].dt.to_period("M")
hist["年"] = hist["date"].dt.year
print(f"\n歷史樣本：{hist['date'].min().date()} ~ {hist['date'].max().date()}，每 5 個交易日取樣，報酬已扣成本 0.6%")


def judge(g):
    if len(g) < 30:
        return "樣本不足"
    mm = g.groupby("月")["x60"].mean()
    yy = g.groupby("年")["x60"].mean()
    if len(g) >= 100 and mm.mean() > 0 and (yy > 0).mean() >= 0.6:
        return "有優勢"
    if mm.mean() < 0 and (yy < 0).mean() >= 0.6:
        return "劣勢"
    return "無優勢"


levels = [("回撤", "趨勢"), ("回撤", "趨勢", "營收"), ("回撤", "趨勢", "營收", "外資"), ("回撤", "趨勢", "營收", "外資", "融資")]
print("\n| 相似條件 | 件數 | 60日平均 | 勝率 | 60日超額 | 每月一票超額 | 贏的月份 | 逐年 60 日超額 | 判定 |")
print("|---|---|---|---|---|---|---|---|---|")
verdicts = []
for lv in levels:
    mask = np.ones(len(hist), bool)
    for k in lv:
        mask &= hist[k].values == cur[k]
    g = hist[mask]
    if len(g) == 0:
        print(f"| {'＋'.join(cur[k] for k in lv)} | 0 | | | | | | | 樣本不足 |")
        verdicts.append("樣本不足")
        continue
    mm = g.groupby("月")["x60"].mean()
    yy = " ".join(f"{y % 100:02d}:{v:+.0%}" for y, v in g.groupby("年")["x60"].mean().items())
    v = judge(g)
    verdicts.append(v)
    print(f"| {'＋'.join(cur[k] for k in lv)} | {len(g)} | {g['r60'].mean():+.1%} | {(g['r60'] > 0).mean():.0%} | "
          f"{g['x60'].mean():+.1%} | {mm.mean():+.1%} | {(mm > 0).mean():.0%} | {yy} | {v} |")

# 系統狀態
notes = []
for f, label in [("results/candidates_today.csv", "今日候選"), ("results/picks_today.csv", "今日訊號")]:
    p = ROOT / f
    if p.exists() and sid in set(pd.read_csv(p, dtype={"stock_id": str})["stock_id"]):
        notes.append(label)
hp = ROOT / "results/paper/positions.csv"
if hp.exists() and sid in set(pd.read_csv(hp, dtype={"stock_id": str})["stock_id"]):
    notes.append("模擬帳戶持有中")
print(f"\n系統狀態：{'、'.join(notes) if notes else '不在今日候選、訊號或模擬帳戶裡'}")

usable = [v for v in verdicts if v != "樣本不足"]
final = usable[-1] if usable else "樣本不足"
print(f"規則判定（最嚴且樣本 >= 30 的那層）：{final}")
if final == "有優勢":
    print("動作：可續抱；用系統停損（-15%）與 60 日期限管理。")
else:
    rule = "收盤站回 20 日均線後隔天賣，最晚 60 個交易日" if cur["回撤"] == "<=-45%" else "照停損或 60 日期限出場，不加碼"
    print(f"動作：沒有續抱優勢 → {rule}；新資金照系統規則（v2 訊號、0050 定期定額）。")
