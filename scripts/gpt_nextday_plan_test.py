"""ChatGPT 建議的「隔日當沖選股＋交易計畫」照規格回測（2026-11-26，Derpy 貼來的規格）。

第二層 基本篩選（訊號日收盤）：股價 20~500、成交金額 > 1 億、量比（今天／20 日均量）> 1.5、當日漲 1%~7%、
  收盤／最高 > 0.97、收盤高於 5 日線與 20 日線。
第三層 動能與籌碼（各自加上去看有沒有幫助）：外資和投信當天都買超、RSI14 在 50~75、收盤創 5 日新高、
  收盤位置 (收-低)/(高-低) >= 0.8。大戶持股、分點我們沒有歷史資料，沒測。
第四層 風險排除：處置中的股票、近 5 日已漲超過 20%。（漲停、長上影線已被第二層的條件排掉；注意股、當沖資格沒有歷史資料。）
第五層 隔日計畫（照範例的比例）：開盤落在昨收的 0%~+2% 才進場（開盤價買），跳空超過 2% 或開低就取消；
  停損昨收 -1.5%、目標昨收 +3%（另測 +4%），都沒碰到就收盤賣。成本 0.435%（當沖）。
日線看不出停損和目標誰先到，所以給兩個數字：悲觀（兩個都碰到算停損）、樂觀（兩個都碰到算停利）。真實結果在兩者之間。
最近約 60 天另外用 5 分 K（data/intraday/，成交最大 300 檔）照實際先後順序算。
    python scripts/gpt_nextday_plan_test.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from twscreener.market import load_panel  # noqa: E402

COST, STOP, GAP_MAX = 0.00435, 0.015, 0.02
p = load_panel()
p = p[p["date"] >= "2022-01-01"]


def piv(c):
    return p.pivot_table(index="date", columns="stock_id", values=c)


O, H, L, C, V, FN, TN = (piv(c) for c in ("open", "high", "low", "close", "vol", "foreign_net", "trust_net"))
pc = C.shift(1)
chg = C / pc - 1
v20 = V.rolling(20).mean().shift(1)
d = C.diff()
rsi = 100 - 100 / (1 + d.clip(lower=0).rolling(14).mean() / (-d.clip(upper=0)).rolling(14).mean())
base = ((C >= 20) & (C <= 500) & (V * C * 1000 > 1e8) & (V / v20 > 1.5) & (chg >= 0.01) & (chg <= 0.07)
        & (C / H > 0.97) & (C > C.rolling(5).mean()) & (C > C.rolling(20).mean()))
dis = pd.read_csv(ROOT / "data" / "disposition.csv", dtype={"stock_id": str}, parse_dates=["start", "end"])
indis = pd.DataFrame(False, index=C.index, columns=C.columns)
for r in dis.itertuples():
    if r.stock_id in indis.columns:
        indis.loc[r.start:r.end, r.stock_id] = True
risk_ok = ~indis & ~indis.shift(-1, fill_value=False) & (C / C.shift(5) - 1 <= 0.20)
liquid = (C >= 20) & (C <= 500) & (V * C * 1000 > 1e8)
sig = {
    "對照：只看股價與成交金額": liquid,
    "第二層 基本篩選": base,
    "第二層＋第四層風險排除": base & risk_ok,
    "　再加 外資投信同買": base & risk_ok & (FN > 0) & (TN > 0),
    "　再加 RSI 50~75": base & risk_ok & (rsi >= 50) & (rsi <= 75),
    "　再加 創 5 日新高": base & risk_ok & (C >= C.rolling(5).max()),
    "　再加 收盤位置 0.8 以上": base & risk_ok & ((C - L) / (H - L).replace(0, np.nan) >= 0.8),
    "　四個動能籌碼條件全加": base & risk_ok & (FN > 0) & (TN > 0) & (rsi >= 50) & (rsi <= 75) & (C >= C.rolling(5).max())
                    & ((C - L) / (H - L).replace(0, np.nan) >= 0.8),
}
# 隔天（都用訊號日收盤 = 昨收當基準）
O1, H1, L1, C1 = (x.shift(-1) for x in (O, H, L, C))
gap = O1 / C - 1
enter = (gap >= 0) & (gap <= GAP_MAX)
stop_px = C * (1 - STOP)


def plan(tp, mode):
    tgt = C * (1 + tp)
    hs, ht = L1 <= stop_px, H1 >= tgt
    r_stop, r_tgt, r_close = stop_px / O1 - 1, tgt / O1 - 1, C1 / O1 - 1
    if mode == "悲觀":
        r = r_close.where(~ht, r_tgt).where(~hs, r_stop)
    else:
        r = r_close.where(~hs, r_stop).where(~ht, r_tgt)
    return r - COST


def summ(x):
    st = x.stack().dropna()
    if len(st) < 30:
        return None
    dm = x.mean(axis=1).dropna()
    yr = dm.groupby(dm.index.year).mean()
    return len(st), (st > 0).mean(), dm.mean(), f"{(yr > 0).sum()}/{len(yr)}", " ".join(f"{a % 100}:{b:+.2%}" for a, b in yr.items())


pd.set_option("display.width", 250, "display.unicode.east_asian_width", True)
rows = []
for name, s in sig.items():
    s = s.fillna(False)
    m = s & enter
    row = {"條件": name, "每天平均幾檔": s.sum(axis=1).mean(), "隔天符合進場區間的比例": m.sum().sum() / max(s.sum().sum(), 1),
           "跳空超過 2% 取消": (s & (gap > GAP_MAX)).sum().sum() / max(s.sum().sum(), 1), "開低取消": (s & (gap < 0)).sum().sum() / max(s.sum().sum(), 1)}
    a = summ((C1 / O1 - 1 - COST).where(m))
    if a:
        row["不設停損停利_每筆"], row["不設_賺的年數"] = a[2], a[3]
    for tp in (0.03, 0.04):
        for mode in ("悲觀", "樂觀"):
            a = summ(plan(tp, mode).where(m))
            if a:
                row[f"目標 {tp:.0%}_{mode}"], row[f"目標 {tp:.0%}_{mode}_勝率"], row[f"目標 {tp:.0%}_{mode}_賺的年數"], row[f"目標 {tp:.0%}_{mode}_每年"] = a[2], a[1], a[3], a[4]
    # 被取消的那些如果硬做會怎樣（開盤買收盤賣）
    a = summ((C1 / O1 - 1 - COST).where(s & (gap > GAP_MAX)))
    if a:
        row["跳空超過 2% 硬追_每筆"] = a[2]
    rows.append(row)
R = pd.DataFrame(rows)
out = ROOT / "results" / "隔天空間"
out.mkdir(parents=True, exist_ok=True)
R.to_csv(out / "gpt_nextday_plan.csv", index=False, encoding="utf-8-sig")
print(f"{C.index[0].date()} ~ {C.index[-1].date()}；成本 {COST:.3%}；停損昨收 -{STOP:.1%}；開盤在昨收 0~+{GAP_MAX:.0%} 才進場；每天一票\n")
v = R.copy()
for c in v.columns[1:]:
    if c.endswith("每年") or c.endswith("年數"):
        continue
    v[c] = v[c].map((lambda z: f"{z:.1f}") if "幾檔" in c else (lambda z: "" if pd.isna(z) else f"{z:.0%}") if ("比例" in c or "取消" in c or "勝率" in c) else (lambda z: "" if pd.isna(z) else f"{z:+.2%}"))
print(v[["條件", "每天平均幾檔", "隔天符合進場區間的比例", "跳空超過 2% 取消", "開低取消"]].to_string(index=False))
print()
print(v[["條件", "不設停損停利_每筆", "目標 3%_悲觀", "目標 3%_樂觀", "目標 4%_悲觀", "目標 4%_樂觀", "目標 3%_悲觀_勝率", "目標 3%_樂觀_勝率", "目標 3%_樂觀_賺的年數", "跳空超過 2% 硬追_每筆"]].to_string(index=False))
print("\n== 目標 3% 逐年（悲觀｜樂觀）")
for _, x in R.iterrows():
    print(f"{x['條件']}：{x.get('目標 3%_悲觀_每年', '')}｜{x.get('目標 3%_樂觀_每年', '')}")

# 5 分 K 照實際先後（最近約 60 天、成交最大 300 檔）
BARS = ROOT / "data" / "intraday"
res = {k: [] for k in sig}
for f in sorted(BARS.glob("*.csv")):
    sid = f.stem
    if sid not in C.columns:
        continue
    b = pd.read_csv(f, parse_dates=["time"])
    b = b[b["time"].dt.time <= pd.Timestamp("13:30").time()]
    b["day"] = b["time"].dt.normalize()
    for day, g in b.groupby("day"):
        if day not in C.index:
            continue
        i = C.index.get_loc(day)
        if i == 0:
            continue
        prev = C.index[i - 1]
        flags = [k for k, s in sig.items() if bool(s.at[prev, sid]) if pd.notna(s.at[prev, sid])]
        if not flags:
            continue
        base_px = C.at[prev, sid]
        o = g["open"].iloc[0]
        if not (0 <= o / base_px - 1 <= GAP_MAX):
            continue
        sp, tg = base_px * (1 - STOP), base_px * 1.03
        r = C.at[day, sid] / o - 1 if pd.notna(C.at[day, sid]) else g["close"].iloc[-1] / o - 1
        for lo, hi in zip(g["low"].to_numpy(), g["high"].to_numpy()):
            if lo <= sp:
                r = sp / o - 1
                break
            if hi >= tg:
                r = tg / o - 1
                break
        for k in flags:
            res[k].append((day, r - COST))
print("\n== 5 分 K 照實際先後（目標 3%、停損 1.5%；同一根都碰到算停損）")
for k, x in res.items():
    if len(x) < 20:
        print(f"{k}：只有 {len(x)} 筆，太少")
        continue
    x = pd.DataFrame(x, columns=["day", "r"])
    dm = x.groupby("day")["r"].mean()
    print(f"{k}：{len(x)} 筆、{len(dm)} 天，勝率 {(x['r'] > 0).mean():.0%}，每筆 {dm.mean():+.2%}")
