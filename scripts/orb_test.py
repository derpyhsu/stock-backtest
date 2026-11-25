"""台股版開盤區間突破（ORB，參考 day-trader-skill 的 trader-entry / trader-exit）當沖回測。

資料：Yahoo 5 分 K（免費、非官方，只有最近約 60 個交易日），存在 data/intraday/<代號>.csv。
股票：最新一天 20 日成交金額最大的 N 檔（流動性好、能當沖）。
規則：
- 開盤區間：9:00、9:05、9:10 三根 5 分 K 的最高、最低。
- 進場：9:15 之後、12:30 之前，第一根「收盤」站上區間高點的 5 分 K，用那根收盤價買（可加：量 >= 區間平均量 × 1.5）。
- 停損：區間中點（或區間低點）；停利：2R（或 1R）。同一根 K 同時碰到停損和停利算停損。
- 沒碰到就在 13:20 那根收盤平倉（13:25 之後是收盤集合競價）。
- 成本：當沖證交稅 0.15% ＋ 手續費 0.1425% × 2 = 0.435%（未折扣）。
對照：同一天同一批股票，9:15 收盤買、13:20 收盤賣（隨機當沖）。
    python scripts/orb_test.py fetch     # 抓 5 分 K（約 0.5 秒一檔），和舊檔合併累積；不存今天還沒收完的資料
    python scripts/orb_test.py           # 回測
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BARS = ROOT / "data" / "intraday"
OUT = ROOT / "results" / "當沖工具"
N = 300
COST = 0.0015 + 0.001425 * 2


def universe():
    files = sorted((ROOT / "data" / "daily").glob("*.csv"))[-20:]
    d = pd.concat(pd.read_csv(f, dtype={"stock_id": str}) for f in files)
    d = d[d["stock_id"].str.fullmatch(r"\d{4}")]
    d["amt"] = d["close"] * d["vol"]
    top = d.groupby(["stock_id", "market"])["amt"].mean().sort_values(ascending=False).head(N)
    return [(s, ".TW" if m == "上市" else ".TWO") for s, m in top.index]


def fetch():
    BARS.mkdir(parents=True, exist_ok=True)
    for k, (sid, suf) in enumerate(universe(), 1):
        url = f"https://query2.finance.yahoo.com/v8/finance/chart/{sid}{suf}?range=60d&interval=5m"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                res = json.load(r)["chart"]["result"][0]
            q = res["indicators"]["quote"][0]
            df = pd.DataFrame({"ts": res["timestamp"], **{c: q[c] for c in ["open", "high", "low", "close", "volume"]}})
            df["time"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert("Asia/Taipei").dt.tz_localize(None)
            df = df.drop(columns="ts").dropna()
            f = BARS / f"{sid}.csv"
            if f.exists():  # Yahoo 只給最近 60 天，每天抓一次累積起來
                df = pd.concat([pd.read_csv(f, parse_dates=["time"]), df]).drop_duplicates("time", keep="last")
            # 只留完整的 5 分 K（盤中抓會有還沒走完的那根）
            now = pd.Timestamp.now(tz="Asia/Taipei").tz_localize(None)
            if now.strftime("%H:%M") < "13:35":
                df = df[df["time"].dt.date < now.date()]
            df.sort_values("time").to_csv(f, index=False)
        except Exception as e:
            print(sid, "失敗", e)
        if k % 50 == 0:
            print(k, flush=True)
        time.sleep(0.5)


def day_trades(bars, stop_rule, target_r, vol_x):
    out = []
    for day, g in bars.groupby(bars["time"].dt.date):
        g = g.set_index("time").sort_index()
        hm = g.index.strftime("%H:%M")
        rng = g[hm.isin(["09:00", "09:05", "09:10"])]
        if len(rng) < 3:
            continue
        hi, lo = rng["high"].max(), rng["low"].min()
        stop = (hi + lo) / 2 if stop_rule == "mid" else lo
        after = g[(hm >= "09:15") & (hm <= "13:20")]
        if len(after) < 40 or hi <= lo:
            continue
        base = after["close"].iat[-1] / after["close"].iat[0] - 1 - COST  # 9:15 收盤買、13:20 收盤賣
        trig = after[(after["close"] > hi) & (after.index.strftime("%H:%M") <= "12:30")]
        if vol_x:
            trig = trig[trig["volume"] >= rng["volume"].mean() * vol_x]
        row = {"day": day, "base": base, "traded": False}
        if len(trig):
            t0 = trig.index[0]
            entry = trig["close"].iat[0]
            risk = entry - stop
            if risk > entry * 0.001:
                target = entry + target_r * risk
                rest = after[after.index > t0]
                exit_p, how = (rest["close"].iat[-1] if len(rest) else entry), "收盤"
                for _, b in rest.iterrows():
                    if b["low"] <= stop:
                        exit_p, how = min(stop, b["open"]), "停損"
                        break
                    if b["high"] >= target:
                        exit_p, how = max(target, b["open"]), "停利"
                        break
                row.update(traded=True, ret=exit_p / entry - 1 - COST, gross=exit_p / entry - 1, how=how,
                           r=(exit_p - entry) / risk)
        out.append(row)
    return out


def backtest():
    files = sorted(BARS.glob("*.csv"))
    variants = {"區間中點停損・2R": ("mid", 2, 0), "區間中點停損・2R・量1.5倍": ("mid", 2, 1.5),
                "區間低點停損・2R": ("low", 2, 0), "區間中點停損・1R": ("mid", 1, 0)}
    lines = [f"股票 {len(files)} 檔，成本 {COST:.3%}"]
    for name, (sr, tr, vx) in variants.items():
        rows = []
        for f in files:
            b = pd.read_csv(f, parse_dates=["time"])
            for r in day_trades(b, sr, tr, vx):
                r["stock_id"] = f.stem
                rows.append(r)
        t = pd.DataFrame(rows)
        tr_ = t[t["traded"]]
        base = t.groupby("day")["base"].mean()
        x = tr_["ret"] - tr_["day"].map(base)
        lines.append(f"\n== {name}（{t['day'].min()} ~ {t['day'].max()}，{t['day'].nunique()} 天）")
        lines.append(f"出手 {len(tr_)} 筆（佔股票日 {len(tr_) / len(t):.0%}）・勝率 {(tr_['ret'] > 0).mean():.0%}・"
                     f"平均每筆 {tr_['ret'].mean():+.3%}（未扣成本 {tr_['gross'].mean():+.3%}）・平均 {tr_['r'].mean():+.2f}R")
        lines.append(f"出場：{tr_['how'].value_counts(normalize=True).map('{:.0%}'.format).to_dict()}")
        lines.append(f"對照隨機當沖每筆 {t['base'].mean():+.3%}・ORB 比同日隨機 {x.mean():+.3%}")
        daily = tr_.groupby("day")["ret"].mean()
        lines.append(f"每天平均報酬為正的天數 {(daily > 0).mean():.0%}・前半段 {daily.iloc[:len(daily) // 2].mean():+.3%}・"
                     f"後半段 {daily.iloc[len(daily) // 2:].mean():+.3%}")
    text = "\n".join(lines)
    print(text)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "orb.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    fetch() if "fetch" in sys.argv else backtest()
