"""前一晚能不能挑出「隔天有上漲空間」的股票（2026-11-26，Derpy 提的目標）。

目標不是收盤賣，而是隔天盤中有空間可以操作：隔天最高價比開盤價高多少（上漲空間）、最低比開盤低多少（下跌空間）。
做法：用訊號日收盤就知道的 30 個左右的數字（漲跌、量、振幅、K 棒位置、法人、融資、離高點距離、這檔股票過去
20 天盤中平均怎麼走…），每天在有量的股票裡排名，用線性模型（ridge）學「哪些數字高的股票隔天空間大」。
只用過去的年份學、下一年驗證（2022 學→驗 2023，2022~23 學→驗 2024…），所以驗證那年模型完全沒看過。
三個模型：學「上漲空間」、學「上漲空間減下跌空間」、學「開盤到收盤報酬」。每天買分數最高的 5 檔，隔天開盤買。
另外測「掛低接」：不追開盤，掛在開盤價下面 2%／3%，跌到才買、收盤賣。
出場用日線能確定的掛單：開盤買進後掛停利 +2%／+3%／+5%，沒到就收盤賣（當沖成本 0.435%）。
停損和停利誰先到，日線看不出來，所以這裡不加停損；最近 60 天另外用 5 分 K 驗（data/intraday/）。
    python scripts/next_day_room_test.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from twscreener.market import load_panel  # noqa: E402

COST, TOP, LAM = 0.00435, 5, 10.0
p = load_panel()
p = p[p["date"] >= "2022-01-01"]


def piv(c):
    return p.pivot_table(index="date", columns="stock_id", values=c)


O, H, L, C, V = (piv(c) for c in ("open", "high", "low", "close", "vol"))
FN, TN, MB = piv("foreign_net"), piv("trust_net"), piv("margin_bal")
r = C.pct_change(fill_method=None)
r = r.where(r.abs() <= 0.11)
adj = (1 + r.fillna(0)).cumprod().where(C.notna())
pc = C.shift(1)
v20 = V.rolling(20).mean()
liquid = (v20 >= 1000) & (C >= 10)
oc = C / O - 1
rng = (H - L) / pc
mkt = r.where(liquid).mean(axis=1)
F = {
    "漲跌 1 日": r, "漲跌 3 日": adj / adj.shift(3) - 1, "漲跌 5 日": adj / adj.shift(5) - 1,
    "漲跌 20 日": adj / adj.shift(20) - 1, "漲跌 60 日": adj / adj.shift(60) - 1,
    "今天振幅": rng, "5 日平均振幅": rng.rolling(5).mean(), "20 日平均振幅": rng.rolling(20).mean(),
    "收在當天高低的位置": (C - L) / (H - L).replace(0, np.nan), "上影線": (H - np.maximum(O, C)) / pc, "下影線": (np.minimum(O, C) - L) / pc,
    "今天開盤到收盤": oc, "今天跳空": O / pc - 1,
    "量比（今天／20 日）": V / v20, "量比（5 日／20 日）": V.rolling(5).mean() / v20, "成交金額（對數）": np.log((V * C).rolling(20).mean()),
    "股價（對數）": np.log(C), "離 20 日高點": C / H.rolling(20).max() - 1, "離 60 日高點": C / H.rolling(60).max() - 1,
    "離 20 日低點": C / L.rolling(20).min() - 1, "離 5 日線": C / C.rolling(5).mean() - 1, "離 20 日線": C / C.rolling(20).mean() - 1,
    "外資今天買超／均量": FN / v20, "外資 5 日買超／均量": FN.rolling(5).sum() / v20, "投信 5 日買超／均量": TN.rolling(5).sum() / v20,
    "融資 5 日增減／均量": (MB - MB.shift(5)) / v20, "20 日波動": r.rolling(20).std(),
    "過去 20 天盤中平均": oc.rolling(20).mean(), "過去 20 天隔夜平均": (O / pc - 1).rolling(20).mean(),
    "過去 20 天上漲空間平均": (H / O - 1).rolling(20).mean(), "過去 20 天下跌空間平均": (O / L - 1).rolling(20).mean(),
    "今天漲停": (r >= 0.095).astype(float).where(r.notna()),
}
# 隔天
O1, H1, L1, C1 = O.shift(-1), H.shift(-1), L.shift(-1), C.shift(-1)
up, dn, oc1 = H1 / O1 - 1, O1 / L1 - 1, C1 / O1 - 1
buyable = liquid & (O1 / C - 1 < 0.095) & O1.notna()
Y = {"上漲空間": up, "空間差（上減下）": up - dn, "開盤到收盤": oc1}


def stack(d):
    return d.where(buyable).stack(future_stack=True)


X = pd.DataFrame({k: stack(v.replace([np.inf, -np.inf], np.nan)) for k, v in F.items()})
T = pd.DataFrame({k: stack(v) for k, v in Y.items()})
T["下跌空間"], T["隔夜"] = stack(dn), stack(O1 / C - 1)
ok = buyable.stack(future_stack=True).fillna(False) & T["開盤到收盤"].notna() & X.notna().mean(axis=1).ge(0.8)
X, T = X[ok], T[ok]
day = X.index.get_level_values(0)
Xr = X.groupby(day).rank(pct=True).sub(0.5).fillna(0.0)          # 每天橫向排名，-0.5 ~ 0.5
yr = pd.Series(day.year, index=X.index)
years = sorted(yr.unique())

score = pd.DataFrame(index=X.index, columns=list(Y), dtype=float)
coef = {}
for y in years[1:]:
    tr, te = (yr < y).to_numpy(), (yr == y).to_numpy()
    A = Xr[tr].to_numpy()
    G = A.T @ A + LAM * np.eye(A.shape[1])
    for k in Y:
        t = T.loc[tr, k]
        t = t.groupby(day[tr]).rank(pct=True).sub(0.5).to_numpy()   # 學相對名次，不學絕對大小
        w = np.linalg.solve(G, A.T @ t)
        score.loc[te, k] = Xr[te].to_numpy() @ w
        coef[(y, k)] = w
te_all = score["上漲空間"].notna()
S, TT, dd = score[te_all], T[te_all], day[te_all]


def plan(t, tp):
    """開盤買、掛停利 tp，沒到收盤賣。扣成本。"""
    return np.where(t["上漲空間"] >= tp, tp, t["開盤到收盤"]) - COST


pd.set_option("display.width", 250, "display.unicode.east_asian_width", True)
print(f"{p['date'].min().date()} ~ {p['date'].max().date()}；驗證年 {years[1:]}；每天買分數最高 {TOP} 檔；成本 {COST:.3%}\n")
rows = []
groups = {"全部有量股票（對照）": pd.Series(True, index=S.index)}
for k in Y:
    rk = S[k].groupby(dd).rank(ascending=False, method="first")
    groups[f"模型學「{k}」前 {TOP} 名"] = rk <= TOP
    groups[f"模型學「{k}」最後 {TOP} 名"] = rk > rk.groupby(dd).transform("max") - TOP
for g, m in groups.items():
    t = TT[m]
    d = dd[m]
    row = {"組別": g, "筆數": len(t), "上漲空間中位": t["上漲空間"].median(), "下跌空間中位": t["下跌空間"].median(),
           "盤中漲過 3% 的比例": (t["上漲空間"] >= 0.03).mean(), "盤中跌過 3% 的比例": (t["下跌空間"] >= 0.03).mean(),
           "隔夜跳空平均": t["隔夜"].mean(), "開盤到收盤（扣成本）": (t["開盤到收盤"] - COST).groupby(d).mean().mean()}
    for tp in (0.02, 0.03, 0.05):
        x = pd.Series(plan(t, tp), index=t.index).groupby(d).mean()
        yy = x.groupby(x.index.year).mean()
        row[f"掛停利 {tp:.0%}"] = x.mean()
        row[f"掛停利 {tp:.0%} 每年"] = " ".join(f"{a % 100}:{b:+.2%}" for a, b in yy.items())
    for dip in (0.02, 0.03):        # 不追開盤，掛在開盤價下面 dip 等它跌下來才買，收盤賣；沒跌到就沒成交
        hit = t["下跌空間"] >= dip
        x = ((1 + t["開盤到收盤"][hit]) / (1 - dip) - 1 - COST).groupby(d[hit]).mean()
        yy = x.groupby(x.index.year).mean()
        row[f"掛低接 -{dip:.0%}"] = x.mean()
        row[f"掛低接 -{dip:.0%} 成交比例"] = hit.mean()
        row[f"掛低接 -{dip:.0%} 每年"] = " ".join(f"{a % 100}:{b:+.2%}" for a, b in yy.items())
    rows.append(row)
R = pd.DataFrame(rows)
out = ROOT / "results" / "隔天空間"
out.mkdir(parents=True, exist_ok=True)
R.to_csv(out / "next_day_room.csv", index=False, encoding="utf-8-sig")
v = R.copy()
for c in v.columns[2:]:
    if not c.endswith("每年"):
        v[c] = v[c].map("{:+.2%}".format if ("扣成本" in c or "停利" in c or "跳空" in c or "低接" in c) and "比例" not in c else "{:.1%}".format)
print(v[[c for c in v.columns if not c.endswith("每年")]].to_string(index=False))
print("\n== 逐年（每筆扣成本）")
for _, x in R.iterrows():
    print(f"{x['組別']}｜停利 2%：{x['掛停利 2% 每年']}｜停利 3%：{x['掛停利 3% 每年']}｜停利 5%：{x['掛停利 5% 每年']}")
    print(f"　　低接 -2%：{x['掛低接 -2% 每年']}｜低接 -3%：{x['掛低接 -3% 每年']}")
print("\n== 模型看重什麼（最後一個驗證年的係數，正＝這個數字越高分數越高）")
for k in Y:
    w = pd.Series(coef[(years[-1], k)], index=X.columns).sort_values()
    print(f"{k}：最負 " + "、".join(f"{a}({b:+.2f})" for a, b in w.head(4).items()) + "｜最正 " + "、".join(f"{a}({b:+.2f})" for a, b in w.tail(5)[::-1].items()))
# 明天的名單（最新一天，用全部資料學）
last = F["漲跌 1 日"].index[-1]
m = liquid.loc[last]
xl = pd.DataFrame({k: v.loc[last] for k, v in F.items()})[m].replace([np.inf, -np.inf], np.nan)
xl = xl[xl.notna().mean(axis=1) >= 0.8].rank(pct=True).sub(0.5).fillna(0.0)
A = Xr.to_numpy()
G = A.T @ A + LAM * np.eye(A.shape[1])
names = p.drop_duplicates("stock_id", keep="last").set_index("stock_id")["name"]
for k in Y:
    w = np.linalg.solve(G, A.T @ T[k].groupby(day).rank(pct=True).sub(0.5).to_numpy())
    top = (xl @ w).sort_values(ascending=False).head(TOP)
    print(f"\n{last.date()} 收盤後，「{k}」分數最高：" + "、".join(f"{s} {names.get(s, '')}" for s in top.index))
