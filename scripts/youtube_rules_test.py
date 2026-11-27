"""YouTube 當沖影片裡講得夠明確的兩條規則（2026-11-27）。影片整理在專案外的 strategy/ 筆記。

資料同 six_intraday_strategies_test.py：data/intraday 的 5 分 K（成交最大約 300 檔、約 60 天），成本 0.535%。
訊號只用收完的 K，下一根開盤進場；最晚 12:30 進場，13:20 收盤平倉；每檔每天每條規則只做一次。

A 長K順勢（影片是做空版，做多版是對照）
  5 分 K 出現「長黑 K」→ 下一根開盤放空；停損＝那根長黑 K 的開盤價往上一檔；之後每出現新的長黑 K，停損移到新那根的開盤價；
  沒碰到就抱到 13:20。長黑 K 的定義影片只說「實體明顯比今天和昨天其他 K 長」，這裡測兩種：
  嚴格＝實體比昨天全部和今天到目前為止每一根都長；寬鬆＝實體 >= 那些 K 平均實體的 3 倍且 >= 0.5%。
B 開盤 5 分鐘順勢
  昨天盤中明顯上漲（收盤比開盤高 2% 以上且收在當天高低區間上 1/3）＋今天開高＋第一根收紅；
  10:00 前有一根收盤高過第一根收盤 → 下一根開盤買；收盤跌破 5 根均線就出場；停損＝今天開盤價；做空全部反過來。
子集合：全部／昨天強勢爆量收高（漲 5% 以上、收最高、量 2 倍）／昨天漲停。後兩個是「隔日沖分點大買」的替代品，真正的分點資料要付費。
    python scripts/youtube_rules_test.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from six_intraday_strategies_test import COST, LAST_ENTRY, ROOT, load  # noqa: E402

OUT = ROOT / "results" / "當沖影片規則"


def tick(p):
    return 0.01 if p < 10 else 0.05 if p < 50 else 0.1 if p < 100 else 0.5 if p < 500 else 1 if p < 1000 else 5


def long_bars(o, c, ref, side, strict):
    """回傳每根是不是長 K（side=-1 長黑、1 長紅）。ref＝昨天每根的實體絕對值。"""
    body = (c - o) * side
    ab = np.abs(c - o)
    out = np.zeros(len(o), bool)
    hist = list(ref)
    for j in range(len(o)):
        if body[j] > 0 and len(hist) >= 20:
            if strict:
                out[j] = body[j] > max(hist)
            else:
                out[j] = body[j] >= 3 * np.mean(hist) and body[j] / o[j] >= 0.005
        hist.append(ab[j])
    return out


def rule_a(o, h, l, c, ref, side, strict):
    lb = long_bars(o, c, ref, side, strict)
    idx = np.flatnonzero(lb)
    if len(idx) == 0 or idx[0] + 1 > LAST_ENTRY:
        return None
    j = idx[0]
    e = o[j + 1]
    stop = o[j] + side * -1 * tick(o[j])           # 做空：長黑開盤價上一檔；做多：長紅開盤價下一檔
    if (e - stop) * side <= 0:
        return None
    for k in range(j + 1, len(o)):
        if side == -1 and h[k] >= stop:
            return (max(stop, o[k]) / e - 1) * -1, "停損"
        if side == 1 and l[k] <= stop:
            return min(stop, o[k]) / e - 1, "停損"
        if lb[k]:
            stop = o[k] - side * tick(o[k])
    return (c[-1] / e - 1) * side, "收盤"


def rule_b(o, h, l, c, pc, yo, yh, yl, yc, ytail, side):
    yr = (yc / yo - 1) * side
    pos = (yc - yl) / (yh - yl) if yh > yl else 0.5
    if yr < 0.02 or (pos < 2 / 3 if side == 1 else pos > 1 / 3):
        return None
    if (o[0] / pc - 1) * side <= 0 or (c[0] - o[0]) * side <= 0:
        return None
    hit = [j for j in range(1, 12) if (c[j] - c[0]) * side > 0]
    if not hit:
        return None
    j = hit[0]
    e = o[j + 1]
    stop = o[0]
    if (e - stop) * side <= 0:
        return None
    cc = np.concatenate([ytail, c])
    ma5 = np.convolve(cc, np.ones(5) / 5, "valid")[len(ytail) - 4:]      # 和 c 對齊
    for k in range(j + 1, len(o)):
        if side == 1 and l[k] <= stop:
            return min(stop, o[k]) / e - 1, "停損"
        if side == -1 and h[k] >= stop:
            return (max(stop, o[k]) / e - 1) * -1, "停損"
        if (c[k] - ma5[k]) * side < 0:
            return (c[k] / e - 1) * side, "破5MA"
    return (c[-1] / e - 1) * side, "收盤"


def main():
    days = load()
    alld = sorted(days)
    files = sorted((ROOT / "data" / "daily").glob("*.csv"))[-110:]
    d = pd.concat(pd.read_csv(f, dtype={"stock_id": str}, usecols=["date", "stock_id", "close"]) for f in files)
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values(["stock_id", "date"])
    g = d.groupby("stock_id")
    d["lim"] = d["close"] / g["close"].shift(1) - 1 >= 0.095
    d["next"] = g["date"].shift(-1)
    lim = d.dropna(subset=["next"]).set_index(["stock_id", "next"])["lim"]
    rows = []
    for n, day in enumerate(alld[1:], 1):
        prev = days[alld[n - 1]]
        for sid, (o, h, l, c, v, vw, pc, cand, vr) in days[day].items():
            if sid not in prev:
                continue
            yo, yh, yl, yc = prev[sid][:4]
            ref = np.abs(yc - yo)
            was_lim = bool(lim.get((sid, day), False))
            base = (day, sid, cand, was_lim)
            for side in (-1, 1):
                for strict in (True, False):
                    r = rule_a(o, h, l, c, ref, side, strict)
                    if r:
                        rows.append(base + (f"A 長{'黑' if side == -1 else '紅'}K（{'嚴格' if strict else '寬鬆'}）", side) + r)
                r = rule_b(o, h, l, c, pc, yo[0], yh.max(), yl.min(), yc[-1], yc[-4:], side)
                if r:
                    rows.append(base + ("B 開盤5分鐘順勢", side) + r)
            b = c[-1] / o[1] - 1
            rows.append(base + ("對照 9:05 進收盤出", 1, b, "收盤"))
            rows.append(base + ("對照 9:05 進收盤出", -1, -b, "收盤"))
    T = pd.DataFrame(rows, columns=["day", "sid", "cand", "lim", "rule", "side", "gross", "exit"])
    T["net"] = T["gross"] - COST
    half = alld[len(alld) // 2]
    T["oos"] = T["day"] >= half

    def stat(x):
        if len(x) == 0:
            return dict(筆數=0)
        dm = x.groupby("day")["net"].mean()
        return dict(筆數=len(x), 勝率=(x["net"] > 0).mean(), 平均=x["net"].mean(), 中位數=x["net"].median(),
                    未扣成本=x["gross"].mean(), 前半=x.loc[~x["oos"], "net"].mean(), 後半=x.loc[x["oos"], "net"].mean(),
                    賺錢天數比=(dm > 0).mean(), 停損比例=(x["exit"] == "停損").mean())

    out = []
    for sub, m in (("全部", T["sid"].notna()), ("昨天強勢爆量收高", T["cand"]), ("昨天漲停", T["lim"])):
        for (rule, side), x in T[m].groupby(["rule", "side"]):
            out.append(dict(子集合=sub, 規則=rule, 方向="多" if side == 1 else "空", **stat(x)))
    R = pd.DataFrame(out)
    OUT.mkdir(parents=True, exist_ok=True)
    R.to_csv(OUT / "youtube_rules.csv", index=False, encoding="utf-8-sig")
    pct = ["勝率", "平均", "中位數", "未扣成本", "前半", "後半", "賺錢天數比", "停損比例"]
    S = R.copy()
    for k in pct:
        S[k] = S[k].map(lambda x: "" if pd.isna(x) else f"{x:+.2%}" if k not in ("勝率", "賺錢天數比", "停損比例") else f"{x:.0%}")
    head = (f"{alld[1].date()} ~ {alld[-1].date()}，{len(alld) - 1} 天，5 分 K；成本 {COST:.3%}；"
            f"前半 ~ {alld[alld.index(half) - 1].date()}，後半 {half.date()} ~")
    txt = head + "\n\n" + S.to_string(index=False)
    (OUT / "youtube_rules.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
