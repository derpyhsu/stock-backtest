"""六套隔日當沖策略回測（2026-11-26，Derpy 貼來的規格）。

規格要 5 年 1 分 K；我們實際只有 data/intraday/ 的 5 分 K（成交最大約 300 檔、約 60 個交易日，每晚累積）。
所以這份是「照規格做、資料縮水」的版本：前半段日子當研究區間、後半段當樣本外，逐月當 walk-forward。
結論只能當初步線索。資料累積更久之後原封不動重跑即可。

共同規則
- 訊號只用已經收完的 5 分 K，下一根的開盤價進場；進場最晚 12:30，13:20 那根收盤一律平倉。
- 停損停利照 K 棒先後；同一根都碰到算停損；跳空越過停損／停利價用該根開盤價成交。
- 成本 0.535%：手續費 0.1425%×2（未折扣）＋當沖證交稅 0.15%＋滑價單邊 0.05%。
- 排除：處置中的股票、開盤就漲停或跌停的。每檔每天每個策略每個方向只做第一個訊號。
- 做空沒有檢查能不能先賣（平盤下、券源），實際會更差。

六套（多／空分開）
 1 ORB：開盤 5／10／15 分鐘區間，之後第一根收盤突破高點做多、跌破低點做空；可加量能確認（>= 前 5 根均量 1.5 倍）。結構停損＝區間中點。
 2 VWAP Reclaim：9:30 後前一根收在均價線下、這根收上去做多（空反過來）；變化＝還要高過前 3 根高點。結構停損＝當日目前最低（最高）。
 3 Gap Fade：開高 2%／3% 以上且第一根收黑→做空；開低同幅度且第一根收紅→做多。結構停損＝第一根高（低）點。
 4 Pullback Continuation：盤中曾比昨收漲 3% 以上，回到均價線附近（0.3% 內），再出現收盤高過前一根高點→做多（空反過來）。結構停損＝回檔低點。
 5 Relative Strength：9:30（另測 10:00）當下，比昨收漲幅在全部股票裡最高 10 檔且在均價線上→做多；最低 10 檔且在均價線下→做空。結構停損＝均價線。
 6 Opening Flush Reversal：開盤 30 分鐘內比開盤價殺 2%／3% 以上，之後第一根收盤高過前一根高點→做多（開盤急拉後反轉做空）。結構停損＝殺低的低點。
出場參數（每個進場都算 100 種）：停損 1%／1.5%／2%／2.5%／結構 × 停利 1R／1.5R／2R／3R × 時間 15／30／60／120 分鐘／抱到收盤。
ATR 停損沒做：60 天資料下參數已經太多。
對照：A 前一晚候選（昨天漲 5% 以上、收最高、量 2 倍）開盤買收盤賣；B 隨機（同一批股票 9:15 進、收盤出，多空各一）；C 只看量（前 15 分鐘量 >= 自己過去平均 2 倍，9:15 做多）。
    python scripts/six_intraday_strategies_test.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BARS = ROOT / "data" / "intraday"
OUT = ROOT / "results" / "當沖六策略"
COST = 0.001425 * 2 + 0.0015 + 0.0005 * 2
STOPS = [0.01, 0.015, 0.02, 0.025, None]          # None = 結構停損
RS = [1.0, 1.5, 2.0, 3.0]
TIMES = [3, 6, 12, 24, 999]                        # 幾根 5 分 K；999 = 收盤
CFG = [(s, r, t) for s in STOPS for r in RS for t in TIMES]
DEFAULT = CFG.index((0.015, 2.0, 999))
NB = 53                                            # 09:00 ~ 13:20 共 53 根
LAST_ENTRY = 42                                    # 12:30 那根


def cfg_name(k):
    s, r, t = CFG[k]
    return f"停損{'結構' if s is None else f'{s:.1%}'}／停利{r:g}R／{'收盤' if t == 999 else f'{t * 5}分'}"


def eval_exits(o, h, l, c, i, side, struct):
    """第 i 根開盤進場。回傳 (報酬[100], 持有根數[100], MFE[100], MAE[100])，報酬未扣成本。"""
    e = o[i]
    oo, hh, ll, cc = o[i:], h[i:], l[i:], c[i:]
    n = len(cc)
    if side == 1:
        adv, fav = 1 - ll / e, hh / e - 1          # 不利、有利幅度（正數）
        gap_o = oo / e - 1
    else:
        adv, fav = hh / e - 1, 1 - ll / e
        gap_o = 1 - oo / e
    cmfe, cmae = np.maximum.accumulate(fav), np.maximum.accumulate(adv)
    cret = side * (cc / e - 1)
    ds_struct = max(abs(e - struct) / e, 0.003) if struct is not None and np.isfinite(struct) else 0.015
    out = np.empty((4, len(CFG)))
    for k, (s, r, t) in enumerate(CFG):
        ds = ds_struct if s is None else s
        dt = r * ds
        js = np.argmax(adv >= ds) if (adv >= ds).any() else n
        jt = np.argmax(fav >= dt) if (fav >= dt).any() else n
        jT = min(t - 1, n - 1)
        if js <= jt and js <= jT:                 # 同一根都碰到算停損
            j, ret = js, -max(ds, -gap_o[js])
        elif jt <= jT:
            j, ret = jt, max(dt, gap_o[jt])
        else:
            j, ret = jT, cret[jT]
        out[:, k] = ret, j + 1, cmfe[j], cmae[j]
    return out


def load():
    files = sorted((ROOT / "data" / "daily").glob("*.csv"))[-110:]
    d = pd.concat(pd.read_csv(f, dtype={"stock_id": str}, usecols=["date", "stock_id", "high", "close", "vol"]) for f in files)
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values(["stock_id", "date"])
    g = d.groupby("stock_id")
    d["pc"] = g["close"].shift(1)
    d["vol20"] = g["vol"].transform(lambda s: s.shift(1).rolling(20).mean())
    d["cand"] = (d["close"] / d["pc"] - 1 >= 0.05) & (d["close"] >= d["high"] * 0.99) & (d["vol"] >= 2 * d["vol20"])
    d["next"] = g["date"].shift(-1)
    info = d.dropna(subset=["next"]).set_index(["stock_id", "next"])[["close", "cand"]]
    dis = pd.read_csv(ROOT / "data" / "disposition.csv", dtype={"stock_id": str}, parse_dates=["start", "end"])
    dis = dis[dis["end"] >= d["date"].min()]
    indis = {(r.stock_id, x) for r in dis.itertuples() for x in pd.date_range(r.start, r.end)}
    days = {}
    for f in sorted(BARS.glob("*.csv")):
        sid = f.stem
        b = pd.read_csv(f, parse_dates=["time"])
        b["m"] = b["time"].dt.hour * 60 + b["time"].dt.minute
        b = b[(b["m"] >= 540) & (b["m"] <= 800)]
        opn15 = []                                 # 這檔過去每天前 15 分鐘的量
        for day, x in b.groupby(b["time"].dt.normalize()):
            if len(x) != NB or x["m"].iat[0] != 540 or (sid, day) in indis or (sid, day) not in info.index:
                continue
            pc, cand = info.loc[(sid, day)]
            a = {k: x[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume")}
            if not (0.905 < a["open"][0] / pc < 1.095):
                continue
            v15 = a["volume"][:3].sum()
            base = np.mean(opn15[-20:]) if len(opn15) >= 5 else np.nan
            opn15.append(v15)
            cv = np.cumsum(a["volume"])
            vw = np.where(cv > 0, np.cumsum((a["high"] + a["low"] + a["close"]) / 3 * a["volume"]) / np.where(cv > 0, cv, 1), a["close"])
            days.setdefault(day, {})[sid] = (a["open"], a["high"], a["low"], a["close"], a["volume"], vw, pc, bool(cand), v15 / base if base and base > 0 else np.nan)
    return days


def signals(o, h, l, c, v, vw, pc):
    """回傳 [(策略, 進場變化, 方向, 訊號那根 j, 結構停損價)]，每個 (策略,變化,方向) 只取第一個。"""
    out = []
    n = len(c)

    def first(mask, lo, hi=LAST_ENTRY - 1):
        idx = np.flatnonzero(mask[lo:hi + 1])
        return lo + idx[0] if len(idx) else None

    vavg5 = np.array([v[max(j - 5, 0):j].mean() if j else np.nan for j in range(n)])
    volok = v >= 1.5 * vavg5
    for nb, lab in ((1, "5分"), (2, "10分"), (3, "15分")):
        hi, lo = h[:nb].max(), l[:nb].min()
        mid = (hi + lo) / 2
        for vc, vl in ((False, ""), (True, "＋量")):
            ok = volok if vc else np.ones(n, bool)
            j = first((c > hi) & ok, nb)
            if j is not None:
                out.append(("1 ORB", lab + vl, 1, j, mid))
            j = first((c < lo) & ok, nb)
            if j is not None:
                out.append(("1 ORB", lab + vl, -1, j, mid))
    pcl, pvw = np.r_[np.nan, c[:-1]], np.r_[np.nan, vw[:-1]]
    h3 = np.array([h[max(j - 3, 0):j].max() if j else np.nan for j in range(n)])
    l3 = np.array([l[max(j - 3, 0):j].min() if j else np.nan for j in range(n)])
    up, dn = (pcl < pvw) & (c > vw), (pcl > pvw) & (c < vw)
    for lab, mu, md in (("站上", up, dn), ("站上＋過前高", up & (c > h3), dn & (c < l3))):
        j = first(mu, 6)
        if j is not None:
            out.append(("2 VWAP Reclaim", lab, 1, j, l[:j + 1].min()))
        j = first(md, 6)
        if j is not None:
            out.append(("2 VWAP Reclaim", lab, -1, j, h[:j + 1].max()))
    gap = o[0] / pc - 1
    for g in (0.02, 0.03):
        if gap >= g and c[0] < o[0]:
            out.append(("3 Gap Fade", f"跳空{g:.0%}", -1, 0, h[0]))
        if gap <= -g and c[0] > o[0]:
            out.append(("3 Gap Fade", f"跳空{g:.0%}", 1, 0, l[0]))
    ph = np.r_[np.nan, h[:-1]]
    pl = np.r_[np.nan, l[:-1]]
    runhi, runlo = np.maximum.accumulate(h), np.minimum.accumulate(l)
    for j in range(3, LAST_ENTRY):                 # 4 回檔續強
        if runhi[j - 1] / pc - 1 >= 0.03:
            k = int(np.argmax(h[:j]))
            if j - k >= 2 and l[k + 1:j + 1].min() <= vw[j] * 1.003 and c[j] > ph[j] and c[j] > vw[j]:
                out.append(("4 Pullback", "漲3%回均價", 1, j, l[k + 1:j + 1].min()))
                break
    for j in range(3, LAST_ENTRY):
        if runlo[j - 1] / pc - 1 <= -0.03:
            k = int(np.argmin(l[:j]))
            if j - k >= 2 and h[k + 1:j + 1].max() >= vw[j] * 0.997 and c[j] < pl[j] and c[j] < vw[j]:
                out.append(("4 Pullback", "跌3%彈均價", -1, j, h[k + 1:j + 1].max()))
                break
    for f in (0.02, 0.03):                         # 6 開盤急殺反轉
        k = int(np.argmin(l[:6]))
        if l[k] <= o[0] * (1 - f):
            j = first(c > ph, k + 1, 11)
            if j is not None:
                out.append(("6 Flush Reversal", f"殺{f:.0%}", 1, j, l[:j + 1].min()))
        k = int(np.argmax(h[:6]))
        if h[k] >= o[0] * (1 + f):
            j = first(c < pl, k + 1, 11)
            if j is not None:
                out.append(("6 Flush Reversal", f"拉{f:.0%}", -1, j, h[:j + 1].max()))
    return out


def main():
    days = load()
    meta, res = [], []
    bench = []
    for day in sorted(days):
        S = days[day]
        sigs = {sid: signals(*a[:7]) for sid, a in S.items()}
        for j, lab in ((5, "9:30"), (11, "10:00")):      # 5 相對強弱
            r = pd.Series({sid: a[3][j] / a[6] - 1 for sid, a in S.items()})
            top, bot = r.nlargest(10).index, r.nsmallest(10).index
            for sid in top:
                if S[sid][3][j] > S[sid][5][j]:
                    sigs[sid].append(("5 Relative Strength", lab, 1, j, S[sid][5][j]))
            for sid in bot:
                if S[sid][3][j] < S[sid][5][j]:
                    sigs[sid].append(("5 Relative Strength", lab, -1, j, S[sid][5][j]))
        for sid, (o, h, l, c, v, vw, pc, cand, vr) in S.items():
            seen = set()
            for strat, var, side, j, struct in sigs[sid]:
                if (strat, var, side) in seen or j + 1 > LAST_ENTRY:
                    continue
                seen.add((strat, var, side))
                meta.append((day, sid, strat, var, side, j + 1))
                res.append(eval_exits(o, h, l, c, j + 1, side, struct))
            b = c[-1] / o[3] - 1
            bench.append((day, sid, "B 隨機", 1, b))
            bench.append((day, sid, "B 隨機", -1, -b))
            if cand:
                bench.append((day, sid, "A 前一晚候選開盤買", 1, c[-1] / o[0] - 1))
            if np.isfinite(vr) and vr >= 2:
                bench.append((day, sid, "C 只看量（前15分量2倍）", 1, b))
    M = pd.DataFrame(meta, columns=["day", "sid", "strat", "var", "side", "bar"])
    R = np.stack(res)                                     # (筆, 4, 100)
    net = R[:, 0, :] - COST
    alld = sorted(days)
    half = alld[len(alld) // 2]
    is_r = (M["day"] < half).to_numpy()
    OUT.mkdir(parents=True, exist_ok=True)
    lines = [f"{alld[0].date()} ~ {alld[-1].date()}，{len(alld)} 天，{len({s for d in days.values() for s in d})} 檔，5 分 K；成本 {COST:.3%}",
             f"研究區間 {alld[0].date()} ~ {alld[alld.index(half) - 1].date()}；樣本外 {half.date()} ~ {alld[-1].date()}", ""]

    def stats(x, d, hold=None, mfe=None, mae=None):
        if len(x) == 0:
            return None
        w, ls = x[x > 0], x[x <= 0]
        dm = pd.Series(x).groupby(np.asarray(d)).mean().reindex([q for q in alld if q >= pd.Series(d).min() and q <= pd.Series(d).max()], fill_value=0.0)
        eq = dm.cumsum()
        mo = pd.Series(x).groupby(pd.DatetimeIndex(d).strftime("%m")).mean()
        s = {"筆數": len(x), "勝率": f"{(x > 0).mean():.0%}", "平均獲利": f"{w.mean():+.2%}" if len(w) else "", "平均虧損": f"{ls.mean():+.2%}" if len(ls) else "",
             "期望值": f"{x.mean():+.3%}", "PF": f"{w.sum() / -ls.sum():.2f}" if len(ls) and ls.sum() < 0 else "", "MDD": f"{(eq - eq.cummax()).min():.1%}",
             "Sharpe": f"{dm.mean() / dm.std() * np.sqrt(250):.1f}" if dm.std() > 0 else "", "逐月": " ".join(f"{k}月{v:+.2%}" for k, v in mo.items())}
        if hold is not None:
            s.update({"平均持有": f"{hold.mean() * 5:.0f}分", "MFE": f"{mfe.mean():+.2%}", "MAE": f"{-mae.mean():+.2%}"})
        return s

    rows, sens = [], []
    for (strat, side), g in M.groupby(["strat", "side"]):
        best = None
        for var, gv in g.groupby("var"):
            ix = gv.index.to_numpy()
            r_ix, o_ix = ix[is_r[ix]], ix[~is_r[ix]]
            if len(r_ix) < 40 or len(o_ix) < 20:
                continue
            er, eo = net[r_ix].mean(axis=0), net[o_ix].mean(axis=0)
            sens.append({"策略": strat, "方向": "多" if side == 1 else "空", "進場": var, "研究筆數": len(r_ix), "樣本外筆數": len(o_ix),
                         "研究：100 組出場中為正的": f"{(er > 0).mean():.0%}", "樣本外：為正的": f"{(eo > 0).mean():.0%}",
                         "研究中位": f"{np.median(er):+.2%}", "樣本外中位": f"{np.median(eo):+.2%}", "樣本外最好": f"{eo.max():+.2%}", "樣本外最差": f"{eo.min():+.2%}",
                         "研究與樣本外的相關": f"{np.corrcoef(er, eo)[0, 1]:+.2f}"})
            k = int(np.argmax(er))
            if best is None or er[k] > best[0]:
                best = (er[k], var, k, r_ix, o_ix)
        if best is None:
            continue
        _, var, k, r_ix, o_ix = best
        for lab, ix, kk in (("研究選出→樣本外", o_ix, k), ("（同一組在研究區間）", r_ix, k), ("預設參數→樣本外", o_ix, DEFAULT), ("預設參數→全期", np.r_[r_ix, o_ix], DEFAULT)):
            s = stats(net[ix, kk], M["day"].to_numpy()[ix], R[ix, 1, kk], R[ix, 2, kk], R[ix, 3, kk])
            rows.append({"策略": strat, "方向": "多" if side == 1 else "空", "看哪段": lab, "進場": var, "出場": cfg_name(kk), **s})
    T = pd.DataFrame(rows)
    B = pd.DataFrame(bench, columns=["day", "sid", "name", "side", "gross"])
    brow = []
    for (name, side), g in B.groupby(["name", "side"]):
        for lab, gg in (("樣本外", g[g["day"] >= half]), ("全期", g)):
            s = stats(gg["gross"].to_numpy() - COST, gg["day"].to_numpy())
            brow.append({"對照": name, "方向": "多" if side == 1 else "空", "看哪段": lab, **s})
    BT = pd.DataFrame(brow)
    SE = pd.DataFrame(sens)
    pd.set_option("display.width", 320, "display.unicode.east_asian_width", True, "display.max_rows", 300, "display.max_colwidth", 60)
    lines += ["== 六套策略（每筆扣成本後）", T.to_string(index=False), "", "== 對照組", BT.to_string(index=False), "", "== 參數敏感度（每個進場變化 × 100 組出場）", SE.to_string(index=False)]
    text = "\n".join(lines)
    print(text)
    (OUT / "six_strategies.txt").write_text(text, encoding="utf-8")
    T.to_csv(OUT / "six_strategies.csv", index=False, encoding="utf-8-sig")
    SE.to_csv(OUT / "sensitivity.csv", index=False, encoding="utf-8-sig")
    BT.to_csv(OUT / "benchmarks.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    sys.exit(main())
