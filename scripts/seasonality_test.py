"""固定月份買賣（季節性、景氣循環）有沒有用（2026-11-26）。

問題：有沒有「每年某月買、抱幾個月賣就會賺」的股票或產業？
資料：Yahoo 月線還原收盤（含配息），目前有流動性的上市櫃股票（20 日均量 >= 500 張、股價 >= 10），最長到 2000 年。
做法：
  - 一組規則＝（股票或產業, 買進月份 m, 抱 k 個月, k=1~6）。上月最後一個交易日收盤買，抱 k 個月後月底收盤賣，扣 0.6%。
  - 用 2020 年（含）以前找規律：至少 8 年資料、賺錢年數 >= 80%、報酬中位數 >= 5%、贏全體平均的年數 >= 70%。
  - 用 2021 年以後驗證：同一組規則在沒看過的年份還賺不賺、還贏不贏全體平均。
  - 對照：所有規則（不挑）在 2021 年以後的表現。挑出來的要明顯比不挑的好才算有東西。
注意：名單是「現在還活著而且有量」的股票，下市的、變冷門的不在裡面，所以早年的絕對報酬偏高；看「贏全體平均」比較公平。
    python scripts/seasonality_test.py fetch    # 抓月線（約 10 分鐘）
    python scripts/seasonality_test.py          # 回測
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MON = ROOT / "data" / "monthly"
OUT = ROOT / "results" / "季節性"
COST = 0.006
SPLIT = 2020          # 這一年（含）以前找規律，之後驗證
KS = (1, 2, 3, 4, 5, 6)


def universe():
    files = sorted((ROOT / "data" / "daily").glob("*.csv"))[-20:]
    d = pd.concat(pd.read_csv(f, dtype={"stock_id": str}, usecols=["stock_id", "market", "close", "vol"]) for f in files)
    d = d[d["stock_id"].str.fullmatch(r"\d{4}")]
    g = d.groupby(["stock_id", "market"]).agg(vol=("vol", "mean"), close=("close", "last")).reset_index()
    g = g[(g["vol"] >= 500) & (g["close"] >= 10)]
    return [(r.stock_id, ".TW" if r.market == "上市" else ".TWO") for r in g.itertuples()]


def fetch():
    MON.mkdir(parents=True, exist_ok=True)
    uni = universe()
    for k, (sid, suf) in enumerate(uni, 1):
        f = MON / f"{sid}.csv"
        if f.exists():
            continue
        url = f"https://query2.finance.yahoo.com/v8/finance/chart/{sid}{suf}?range=max&interval=1mo"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                res = json.load(r)["chart"]["result"][0]
            df = pd.DataFrame({"ts": res["timestamp"], "adj": res["indicators"]["adjclose"][0]["adjclose"]}).dropna()
            df["month"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert("Asia/Taipei").dt.strftime("%Y-%m")
            df.drop_duplicates("month", keep="last")[["month", "adj"]].to_csv(f, index=False)
        except Exception as e:
            print(f"{sid} 失敗（{e.__class__.__name__}）", flush=True)
        if k % 100 == 0:
            print(f"{k}/{len(uni)}", flush=True)
        time.sleep(0.4)
    print(f"完成，{len(list(MON.glob('*.csv')))} 檔")


def load():
    px = {}
    for f in MON.glob("*.csv"):
        s = pd.read_csv(f).set_index("month")["adj"]
        px[f.stem] = s[s > 0]
    px = pd.DataFrame(px).sort_index()
    bad = px.pct_change().abs() > 3      # 單月漲跌超過 300%：Yahoo 壞點，連同下個月一起拿掉
    px = px.mask(bad | bad.shift(-1, fill_value=False))
    this = pd.Timestamp.now().strftime("%Y-%m")
    px = px[px.index < this]          # 這個月還沒走完
    px.index = pd.PeriodIndex(px.index, freq="M")
    return px


def rules(px, label):
    """回傳每組規則每年的報酬與超額。px：月底還原收盤（欄＝股票或產業）。"""
    mkt = px.pct_change().clip(-0.7, 2.0).mean(axis=1)      # 全體等權平均月報酬（Yahoo 偶有壞點，先截掉離譜值）
    mkt_px = (1 + mkt.fillna(0)).cumprod()
    rows = []
    for k in KS:
        r = px.shift(-k) / px - 1 - COST                    # 月底買、k 個月後月底賣
        rm = mkt_px.shift(-k) / mkt_px - 1 - COST
        for m in range(1, 13):
            sel = r.index.month == (m - 2) % 12 + 1         # 上個月月底買＝m 月初進場
            rr, bb = r[sel], rm[sel]
            yrs = rr.index.year + (rr.index.month == 12)    # 12 月底買的算隔年 1 月進場
            rr = rr.set_axis(yrs)
            bb = bb.set_axis(yrs)
            tr, te = rr[rr.index <= SPLIT], rr[rr.index > SPLIT]
            xtr, xte = tr.sub(bb[bb.index <= SPLIT], axis=0), te.sub(bb[bb.index > SPLIT], axis=0)
            t = pd.DataFrame({"n_tr": tr.notna().sum(), "win_tr": (tr > 0).sum() / tr.notna().sum(), "med_tr": tr.median(),
                              "beat_tr": (xtr > 0).sum() / xtr.notna().sum(),
                              "n_te": te.notna().sum(), "win_te": (te > 0).sum() / te.notna().sum(), "avg_te": te.mean(),
                              "med_te": te.median(), "beat_te": (xte > 0).sum() / xte.notna().sum(), "x_te": xte.mean()})
            t["買進月"], t["抱幾個月"] = m, k
            rows.append(t.reset_index(names=label))
    t = pd.concat(rows, ignore_index=True)
    # 同一檔、同樣抱 k 個月，12 個買進月份在驗證期的平均：用來分辨「這個月份特別好」還是「這檔本來就一直漲」
    t["self_te"] = t.groupby([label, "抱幾個月"])["avg_te"].transform("mean")
    t["vs_self"] = t["avg_te"] - t["self_te"]
    return t


def report(t, label, names=None):
    ok = t[(t["n_tr"] >= 8) & (t["n_te"] >= 4)]
    pick = ok[(ok["win_tr"] >= 0.8) & (ok["med_tr"] >= 0.05) & (ok["beat_tr"] >= 0.7)]
    lines = [f"規則總數 {len(ok):,}（至少 8 年找規律、4 年驗證），{SPLIT} 年以前符合條件的 {len(pick):,} 組（{len(pick) / max(len(ok), 1):.1%}）"]
    for nm, g in (("不挑（全部規則）", ok), ("挑出來的", pick)):
        if len(g):
            lines.append(f"{nm}：{SPLIT + 1} 年以後 賺錢年數比例 {g['win_te'].mean():.0%}、平均報酬 {g['avg_te'].mean():+.1%}、"
                         f"贏全體平均的年數比例 {g['beat_te'].mean():.0%}、平均超額 {g['x_te'].mean():+.1%}、"
                         f"比同一檔其他月份多 {g['vs_self'].mean():+.1%}（{(g['vs_self'] > 0).mean():.0%} 的規則為正）")
    keep = pick[(pick["win_te"] >= 0.8) & (pick["beat_te"] >= 0.6) & (pick["x_te"] > 0) & (pick["vs_self"] > 0)].copy()
    lines.append(f"兩段都成立（驗證期賺錢年數 >= 80%、贏全體 >= 60%、超額為正、也比同一檔其他月份好）：{len(keep):,} 組，占挑出來的 {len(keep) / max(len(pick), 1):.0%}")
    if names is not None:
        keep.insert(1, "名稱", keep[label].map(names))
    keep = keep.sort_values(["x_te", "win_te"], ascending=False)
    return lines, pick, keep


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "fetch":
        return fetch()
    OUT.mkdir(parents=True, exist_ok=True)
    px = load()
    ind = pd.read_csv(ROOT / "data" / "industry.csv", dtype={"stock_id": str}).set_index("stock_id")["industry"]
    latest = sorted((ROOT / "data" / "daily").glob("*.csv"))[-1]
    names = pd.read_csv(latest, dtype={"stock_id": str}).set_index("stock_id")["name"]
    text = [f"月線 {px.index[0]} ~ {px.index[-1]}，{px.shape[1]} 檔；{SPLIT} 年（含）以前找規律、之後驗證；成本 {COST:.1%}", ""]

    t = rules(px, "stock_id")
    lines, pick, keep = report(t, "stock_id", names)
    text += ["== 個股"] + lines
    keep.to_csv(OUT / "stock_rules_kept.csv", index=False, encoding="utf-8-sig")
    # 隨機對照：把驗證期的年份打亂沒有意義（同一年相關），改看「挑出來的」在驗證期的分布和「不挑」差多少
    fmt = lambda v: f"{v:+.1%}"  # noqa: E731
    show = keep.head(25)
    text += ["", "兩段都成立的前 25 組（依驗證期超額排序）："]
    for r in show.itertuples():
        text.append(f"{r.stock_id} {r.名稱}：{r.買進月} 月初買、抱 {r.抱幾個月} 個月｜{SPLIT} 前 {r.n_tr} 年賺 {r.win_tr:.0%}、中位數 {fmt(r.med_tr)}"
                    f"｜之後 {r.n_te} 年賺 {r.win_te:.0%}、平均 {fmt(r.avg_te)}、超額 {fmt(r.x_te)}、比自己其他月份 {fmt(r.vs_self)}")

    # 產業：成分股等權
    cols = [c for c in px.columns if c in ind.index]
    ret = px[cols].pct_change()
    ipx = (1 + ret.T.groupby(ind.reindex(cols).values).mean().T.fillna(0)).cumprod()
    cnt = ind.reindex(cols).value_counts()
    ipx = ipx[[c for c in ipx.columns if cnt.get(c, 0) >= 5]]
    ti = rules(ipx, "產業")
    # 產業的全體平均要用個股的全體，不是產業的平均；差異小，這裡沿用
    lines, pick_i, keep_i = report(ti, "產業")
    text += ["", "== 產業（成分股等權，至少 5 檔）"] + lines
    keep_i.to_csv(OUT / "industry_rules_kept.csv", index=False, encoding="utf-8-sig")
    for r in keep_i.head(25).itertuples():
        text.append(f"{r.產業}：{r.買進月} 月初買、抱 {r.抱幾個月} 個月｜{SPLIT} 前 {r.n_tr} 年賺 {r.win_tr:.0%}、中位數 {fmt(r.med_tr)}"
                    f"｜之後 {r.n_te} 年賺 {r.win_te:.0%}、平均 {fmt(r.avg_te)}、超額 {fmt(r.x_te)}、比自己其他月份 {fmt(r.vs_self)}")

    # 全市場月份效應：哪幾個月買、抱 1 個月，全體平均
    m1 = px.pct_change().clip(-0.7, 2.0).mean(axis=1)
    by = pd.DataFrame({"前段平均": m1[m1.index.year <= SPLIT].groupby(m1[m1.index.year <= SPLIT].index.month).mean(),
                       "前段賺的年數": m1[m1.index.year <= SPLIT].groupby(m1[m1.index.year <= SPLIT].index.month).apply(lambda s: f"{(s > 0).sum()}/{len(s)}"),
                       "後段平均": m1[m1.index.year > SPLIT].groupby(m1[m1.index.year > SPLIT].index.month).mean(),
                       "後段賺的年數": m1[m1.index.year > SPLIT].groupby(m1[m1.index.year > SPLIT].index.month).apply(lambda s: f"{(s > 0).sum()}/{len(s)}")})
    text += ["", "== 全體股票等權平均，各月份的月報酬（未扣成本）"]
    for m, r in by.iterrows():
        text.append(f"{m:>2} 月：{SPLIT} 前 {r['前段平均']:+.1%}（{r['前段賺的年數']}）｜之後 {r['後段平均']:+.1%}（{r['後段賺的年數']}）")
    s = "\n".join(text)
    print(s)
    (OUT / "seasonality.txt").write_text(s, encoding="utf-8")


if __name__ == "__main__":
    main()
