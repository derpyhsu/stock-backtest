"""模擬帳戶：完全照規則買賣，不經人手。每晚跑一次，也可以從過去某天開始回測整個帳戶。

    python -m twscreener.portfolio daily                  # 每晚：執行今天開盤的委託、檢查出場、排明天的委託
    python -m twscreener.portfolio backtest 2024-12-01    # 同一套規則從某天跑到今天，看淨值曲線與最大回撤

規則（全部寫死，中途不改；換版規則見 專案總覽.md「什麼時候校正規則」）：
- 選股 v4（2026-11-26 起，Derpy 決定：不求贏大盤，要正報酬、年化兩成左右、常有進出）：資金分三份，各 1/3，各自像獨立帳戶。
  1. v2 月營收加速（twscreener/pead.py，每月 11 號後第一個交易日出訊號），最多 10 檔，停損 -15%、60 天。
  2. S1 一年新高（s1_pick，天天選），最多 10 檔，停損 -15%、60 天。
  3. 處置股（disp_pick，data/disposition.csv）：公告處置的隔天（處置第一天）開盤買，每筆佔那一份的 5%，最多 20 檔，
     不設停損，處置最後一天收盤賣（收盤跌停就隔天開盤賣）。
  回測 2023-04～2026-11：v4 +137%／回撤 -17.7%，v3 +84%／-21.8%，v2 +115%／-27.6%，加權 +142%／-28.7%（results/持有天數/v3_v4_backtest.txt）。
  持股遇到分割、減資（開盤和前一個收盤差超過 11%）會自動換算股數和成本。
  v3（只在 2026-11-26 白天，沒有實際跑過一晚）：前兩份各半。v2（2026-11-24～11-26）：只有月營收加速。
- 進場：訊號日隔天開盤買；開盤已經低於停損、或開盤漲停（買不到）就不買。開盤跌停的賣單留到隔天。
- 部位（前兩份）：打到停損最多虧「那一份淨值」的 1%，單檔最多那一份的 20%，現金不夠就少買。
- 出場：收盤跌破停損 → 隔天開盤賣；持有滿 60 個交易日 → 隔天開盤賣。不設固定停利（回測顯示會砍掉大漲股）。
  續抱（2026-11-26 晚上起，Derpy 決定）：滿 60 天那天收盤（還原價）還在 60 日線上就續抱，之後哪天收盤跌破 60 日線、
  或滿 120 天，隔天開盤賣。回測 2023-04～2026-11：+168.8%／回撤 -18.1%（不續抱 +137.0%／-17.7%），四年都不輸不續抱；上限改 100、121、140 天是 +169.0%、+158.2%、+154.3%，都贏不續抱，但差一天就差 10 個百分點，數字本身別當真
  （scripts/exit_rule_variants_test.py 的 D）。每晚把每檔持股續抱或賣出的依據寫到 results/paper/hold_review.csv。
- 成本：買手續費 0.1425%，賣手續費 0.1425% + 證交稅 0.3%。
- 比較：同期間加權指數。
狀態存在 results/paper/：positions.csv、orders.csv、trades.csv、nav.csv。
"""
import math
import sys

import numpy as np
import pandas as pd

from .fundamentals import load_industry
from .market import RESULTS, load_market_index, load_panel
from .review import market_context
from . import pead
from .simulate import pick, prices

PAPER = RESULTS / "paper"
CAPITAL = 1_000_000
RISK = 0.01
MAX_POS = 0.20
MAX_HOLD = 10
HOLD_DAYS = 60
EXT_MA, EXT_DAYS = 60, 120   # 滿 HOLD_DAYS 時收盤還在 EXT_MA 日線上就續抱，跌破或滿 EXT_DAYS 天才賣
FEE, TAX = 0.001425, 0.003
LIMIT = 0.095   # 開盤離昨收 ±9.5% 以上視為漲跌停鎖住，成交不了


class Account:
    def __init__(self, cash=CAPITAL):
        self.cash = cash
        self.pos = pd.DataFrame(columns=["stock_id", "name", "strategy", "entry_date", "entry", "shares", "stop", "days", "exit_date"])
        self.orders = pd.DataFrame(columns=["date", "side", "stock_id", "name", "strategy", "shares", "stop", "reason", "exit_date"])
        self.sleeve_w = SLEEVE_W["v3"]    # 各策略佔的資金比例；run() 依版本設定
        self.trades, self.nav = [], []
        self.review = []                  # 最近一次 plan() 每檔持股續抱或賣出的依據

    def value(self, close):
        px = self.pos["stock_id"].map(close).fillna(self.pos["entry"])
        return self.cash + float((px * self.pos["shares"]).sum())

    def execute(self, day, opn, prev=None):
        """前一晚排的委託在今天開盤成交。"""
        keep = []
        for _, o in self.orders.iterrows():
            p = opn.get(o["stock_id"], np.nan)
            if not np.isfinite(p):
                continue
            pc = prev.get(o["stock_id"], np.nan) if prev is not None else np.nan
            if o["side"] == "賣" and np.isfinite(pc) and p <= pc * (1 - LIMIT):
                keep.append(o)  # 開盤跌停：實際賣不掉，留到隔天再賣
                continue
            if o["side"] == "買" and np.isfinite(pc) and p >= pc * (1 + LIMIT):
                continue  # 開盤漲停：實際買不到，不進場
            if o["side"] == "賣":
                h = self.pos[self.pos["stock_id"] == o["stock_id"]]
                if h.empty:
                    continue
                h = h.iloc[0]
                self.cash += h["shares"] * p * (1 - FEE - TAX)
                self.trades.append({"賣出日": day.date(), "stock_id": h["stock_id"], "name": h["name"], "strategy": h["strategy"],
                                    "買進日": h["entry_date"], "買價": h["entry"], "賣價": p, "股數": h["shares"],
                                    "報酬": p * (1 - FEE - TAX) / (h["entry"] * (1 + FEE)) - 1, "原因": o["reason"]})
                self.pos = self.pos[self.pos["stock_id"] != o["stock_id"]]
            else:
                if p <= o["stop"] or (self.pos["stock_id"] == o["stock_id"]).any():
                    continue  # 開盤就低於停損：不進場
                shares = min(o["shares"], math.floor(self.cash / (p * (1 + FEE))))
                if shares <= 0:
                    continue
                self.cash -= shares * p * (1 + FEE)
                self.pos.loc[len(self.pos) + 1000] = [o["stock_id"], o["name"], o["strategy"], day.date(), p, shares, o["stop"], 0, o.get("exit_date", np.nan)]
        self.pos = self.pos.reset_index(drop=True)
        self.orders = pd.DataFrame(keep, columns=self.orders.columns) if keep else self.orders.iloc[0:0]

    def plan(self, day, close, cands, prev=None, trend=None):
        """收盤後：檢查出場、排明天的買進。trend＝(還原收盤, 60 日線) 兩個 Series，有給才套用續抱規則。"""
        self.pos["days"] = self.pos["days"] + 1
        sells = []
        # 處置股：處置最後一天收盤賣（前一晚就知道，當天收盤前下單）；收盤跌停賣不掉就隔天開盤賣
        late = set()
        for idx, h in self.pos.iterrows():
            if pd.isna(h["exit_date"]) or str(day.date()) < str(h["exit_date"])[:10]:
                continue
            c = close.get(h["stock_id"], np.nan)
            pc = prev.get(h["stock_id"], np.nan) if prev is not None else np.nan
            if not np.isfinite(c):
                continue
            if np.isfinite(pc) and c <= pc * (1 - LIMIT):
                late.add(h["stock_id"])
                continue
            self.cash += h["shares"] * c * (1 - FEE - TAX)
            self.trades.append({"賣出日": day.date(), "stock_id": h["stock_id"], "name": h["name"], "strategy": h["strategy"],
                                "買進日": h["entry_date"], "買價": h["entry"], "賣價": c, "股數": h["shares"],
                                "報酬": c * (1 - FEE - TAX) / (h["entry"] * (1 + FEE)) - 1, "原因": "處置結束"})
            self.pos = self.pos.drop(idx)
        self.pos = self.pos.reset_index(drop=True)
        self.review = []
        for _, h in self.pos.iterrows():
            c = close.get(h["stock_id"], np.nan)
            why, basis, gap = None, "", np.nan
            if h["stock_id"] in late:
                why = "處置結束（跌停順延）"
            elif pd.notna(h["exit_date"]):
                basis = f"處置股，{str(h['exit_date'])[:10]} 收盤賣"
            elif np.isfinite(c) and c <= h["stop"]:
                why = "跌破停損"
            else:
                if trend is not None:
                    a_, m_ = trend[0].get(h["stock_id"], np.nan), trend[1].get(h["stock_id"], np.nan)
                    gap = a_ / m_ - 1 if np.isfinite(a_) and np.isfinite(m_) and m_ > 0 else np.nan
                line = f"收盤比 {EXT_MA} 日線{'高' if gap >= 0 else '低'} {abs(gap):.1%}" if np.isfinite(gap) else f"沒有 {EXT_MA} 日線資料"
                if h["days"] < HOLD_DAYS:
                    basis = f"未滿 {HOLD_DAYS} 天（第 {h['days']} 天），只看停損；{line}"
                elif trend is None:
                    why = f"持有滿 {HOLD_DAYS} 天"
                elif h["days"] >= EXT_DAYS:
                    why = f"持有滿 {EXT_DAYS} 天"
                elif np.isfinite(gap) and gap > 0:
                    basis = f"滿 {HOLD_DAYS} 天（第 {h['days']} 天）續抱：{line}，跌破就賣，最多抱到第 {EXT_DAYS} 天"
                else:
                    why = f"持有滿 {HOLD_DAYS} 天且{line}"
            if why:
                sells.append((h, why))
            ret = c / h["entry"] - 1 if np.isfinite(c) and h["entry"] else np.nan
            self.review.append({"date": day.date(), "stock_id": h["stock_id"], "name": h["name"], "strategy": h["strategy"],
                                "買進日": h["entry_date"], "持有天數": h["days"], "目前報酬": round(ret, 4) if np.isfinite(ret) else np.nan,
                                "決定": "明天開盤賣" if why else "續抱", "依據": why or basis,
                                "停損價": h["stop"] if pd.isna(h["exit_date"]) else np.nan,
                                "離停損": round(c / h["stop"] - 1, 4) if np.isfinite(c) and pd.isna(h["exit_date"]) and h["stop"] > 0 else np.nan,
                                "離60日線": round(gap, 4) if np.isfinite(gap) else np.nan,
                                "60日線價": round(c / (1 + gap), 2) if np.isfinite(gap) and np.isfinite(c) else np.nan})
        rows = [{"date": day.date(), "side": "賣", "stock_id": h["stock_id"], "name": h["name"], "strategy": h["strategy"],
                 "shares": h["shares"], "stop": h["stop"], "reason": why, "exit_date": np.nan} for h, why in sells]
        nav = self.value(close)
        sleeved = "sleeve" in cands.columns
        W = self.sleeve_w
        cap = {k: SLEEVE_MAX.get(k, MAX_HOLD) for k in W}
        slots = (sum(cap.values()) if sleeved else MAX_HOLD) - (len(self.pos) - len(sells))
        held = set(self.pos["stock_id"])
        budget = self.cash + sum(h["shares"] * close.get(h["stock_id"], 0) for h, _ in sells)
        # v3：資金分兩半（SLEEVES），每半像一個獨立帳戶：各自最多 MAX_HOLD 檔、風險與單檔上限照那一半的淨值算
        sold = {h["stock_id"] for h, _ in sells}
        rest = self.pos[~self.pos["stock_id"].isin(sold)]
        sleeve_of = lambda st: next((k for k in W if st in SLEEVES[k]), next(iter(W)))
        used = {k: 0.0 for k in W}
        cnt = {k: 0 for k in W}
        for _, h in rest.iterrows():
            k = sleeve_of(h["strategy"])
            used[k] += h["shares"] * close.get(h["stock_id"], h["entry"])
            cnt[k] += 1
        for _, c in cands.iterrows():
            if slots <= 0 or c["stock_id"] in held:
                continue
            px, stop = c["close_at_pick"], c["close_at_pick"] * (1 + c["stop"])
            if not (np.isfinite(px) and px > stop > 0):
                continue
            if "sleeve" in cands.columns and pd.notna(c.get("sleeve")):
                k, share = c["sleeve"], W[c["sleeve"]]
                if cnt[k] >= cap[k]:
                    continue
                room = nav * share - used[k]
                if pd.notna(c.get("weight")):   # 固定比例（處置股：不設停損，每筆佔那一份資金的固定比例）
                    shares = math.floor(min(nav * share * c["weight"] / px, budget / (px * 1.01), room / px))
                else:
                    shares = math.floor(min(nav * share * RISK / (px - stop), nav * share * MAX_POS / px, budget / (px * 1.01), room / px))
            else:
                k = None
                shares = math.floor(min(nav * RISK / (px - stop), nav * MAX_POS / px, budget / (px * 1.01)))
            if shares <= 0:
                continue
            budget -= shares * px
            slots -= 1
            held.add(c["stock_id"])
            if k is not None:
                used[k] += shares * px
                cnt[k] += 1
            rows.append({"date": day.date(), "side": "買", "stock_id": c["stock_id"], "name": c["name"], "strategy": c["strategy"],
                         "shares": shares, "stop": round(stop, 2), "reason": REASON.get(c["strategy"], "分數 {:+.1%}").format(c["score"]),
                         "exit_date": c.get("exit_date", np.nan)})
        self.orders = pd.DataFrame(rows, columns=self.orders.columns)
        self.nav.append({"date": day.date(), "淨值": nav, "現金": self.cash, "持股數": len(self.pos)})

    def save(self):
        PAPER.mkdir(parents=True, exist_ok=True)
        self.pos.to_csv(PAPER / "positions.csv", index=False, encoding="utf-8-sig")
        self.orders.to_csv(PAPER / "orders.csv", index=False, encoding="utf-8-sig")
        old = PAPER / "trades.csv"
        t = pd.DataFrame(self.trades)
        if old.exists() and len(t) == 0:
            pass
        elif old.exists():
            t = pd.concat([pd.read_csv(old, dtype={"stock_id": str}), t])
        if len(t):
            t.to_csv(old, index=False, encoding="utf-8-sig")
        n = PAPER / "nav.csv"
        nv = pd.DataFrame(self.nav)
        if n.exists():
            nv = pd.concat([pd.read_csv(n), nv]).drop_duplicates("date", keep="last")
        nv.to_csv(n, index=False, encoding="utf-8-sig")
        (PAPER / "cash.txt").write_text(f"{self.cash:.2f}")
        pd.DataFrame(self.review, columns=["date", "stock_id", "name", "strategy", "買進日", "持有天數", "目前報酬", "決定", "依據",
                                           "停損價", "離停損", "離60日線", "60日線價"]).to_csv(
            PAPER / "hold_review.csv", index=False, encoding="utf-8-sig")

    @classmethod
    def load(cls):
        a = cls()
        if (PAPER / "cash.txt").exists():
            a.cash = float((PAPER / "cash.txt").read_text())
            a.pos = pd.read_csv(PAPER / "positions.csv", dtype={"stock_id": str})
            if "exit_date" not in a.pos.columns:
                a.pos["exit_date"] = np.nan
            o = PAPER / "orders.csv"
            a.orders = pd.read_csv(o, dtype={"stock_id": str}) if o.exists() and o.stat().st_size > 0 else a.orders
            if "exit_date" not in a.orders.columns:
                a.orders["exit_date"] = np.nan
        return a


def _inputs():
    panel = load_panel()
    raw, adj, opn, vol = prices(panel)
    raw_open = panel.pivot_table(index="date", columns="stock_id", values="open")
    sig = pd.read_csv(RESULTS / "market_signals.csv", dtype={"stock_id": str})
    sig["date"] = pd.to_datetime(sig["signal_date"])
    today = RESULTS / "picks_today.csv"
    if today.exists():  # 今天的訊號還沒有隔天開盤價，不在 market_signals.csv 裡，從今日選股補上
        t = pd.read_csv(today, dtype={"stock_id": str})
        if len(t):
            t["date"] = pd.to_datetime(t["date"])
            t = t[~t["date"].isin(sig["date"])]
            sig = pd.concat([sig, t[["date", "stock_id", "name", "strategy"]]], ignore_index=True)
    ctx = market_context()[["date", "大盤狀態"]]
    return raw, adj, opn, vol, raw_open, sig, ctx, load_industry()


REASON = {"營收加速": "月營收年增比前 3 個月平均多 {:.0%}，創 12 個月新高", "一年新高": "收盤創一年新高，近 20 日漲 {:+.0%}"}
REASON["處置股"] = "公告處置（累計第 {:.0f} 次）；不設停損，處置最後一天收盤賣"
SLEEVES = {"v2": {"營收加速", "逆勢籌碼"}, "s1": {"一年新高"}, "disp": {"處置股"}}  # 各份資金管哪些策略；v1 留下的持股算 v2 那份
SLEEVE_W = {"v3": {"v2": 0.5, "s1": 0.5}, "v4": {"v2": 1 / 3, "s1": 1 / 3, "disp": 1 / 3}}
SLEEVE_MAX = {"disp": 20}            # 處置股那份最多 20 檔（其他 MAX_HOLD）
DISP_W = 0.05                        # 處置股每筆佔那一份資金的 5%（＝總資金約 1.7%）
_disp = {}


def disp_pick(day, raw, vol, ind, names):
    """v4 的第三份：當天收盤後公告處置的股票，隔天（處置第一天）開盤買，處置最後一天收盤賣。
    資料 data/disposition.csv（scripts/fetch_disposition.py）。回測見 results/大盤以外/README.md。"""
    if "d" not in _disp:
        f = RESULTS.parent / "data" / "disposition.csv"
        d = pd.read_csv(f, dtype={"stock_id": str}, parse_dates=["announce", "end"]) if f.exists() else pd.DataFrame(columns=["announce"])
        _disp["d"] = d[d["stock_id"].str.fullmatch(r"\d{4}")] if len(d) else d
    d = _disp["d"]
    g = d[d["announce"] == day].drop_duplicates("stock_id", keep="last")
    if g.empty or day not in raw.index:
        return pd.DataFrame()
    i = raw.index.get_loc(day)
    liquid = (vol.iloc[max(0, i - 19):i + 1].mean() >= 500) & (raw.iloc[i] >= 10)
    g = g[[x in raw.columns and bool(liquid.get(x, False)) for x in g["stock_id"]]]
    if g.empty:
        return pd.DataFrame()
    return pd.DataFrame({"stock_id": g["stock_id"].values, "name": [names.get(x, x) for x in g["stock_id"]], "strategy": "處置股",
                         "industry": [ind.get(x, "其他") for x in g["stock_id"]], "score": g["count"].astype(float).values, "stop": -0.999,
                         "close_at_pick": raw.loc[day, g["stock_id"]].values, "sleeve": "disp", "weight": DISP_W,
                         "exit_date": g["end"].dt.strftime("%Y-%m-%d").values})
S1_STOP = -0.15


def s1_pick(day, sig, raw, adj, vol, ind, names):
    """v3 的另一半：S1 一年新高（market_signals／今日選股），同一天多檔時 20 日漲幅小的先買（離突破點近）。
    回測見 scripts/s1_portfolio_test.py、results/持有天數/README.md。"""
    g = sig[(sig["date"] == day) & (sig["strategy"] == "S1_一年新高")]
    if g.empty or day not in adj.index:
        return pd.DataFrame()
    i = adj.index.get_loc(day)
    liquid = (vol.iloc[max(0, i - 19):i + 1].mean() >= 500) & (raw.iloc[i] >= 10)
    ids = [x for x in g["stock_id"].unique() if x in adj.columns and liquid.get(x, False)]
    if not ids or i < 20:
        return pd.DataFrame()
    r20 = (adj.iloc[i][ids] / adj.iloc[i - 20][ids] - 1).dropna().sort_values()
    return pd.DataFrame({"stock_id": r20.index, "name": [names.get(x, x) for x in r20.index], "strategy": "一年新高",
                         "industry": [ind.get(x, "其他") for x in r20.index], "score": r20.values, "stop": S1_STOP,
                         "close_at_pick": raw.loc[day, r20.index].values, "sleeve": "s1"})


def _names(sig):
    latest = sorted((RESULTS.parent / "data" / "daily").glob("*.csv"))[-1]
    n = pd.read_csv(latest, dtype={"stock_id": str}, usecols=["stock_id", "name"])
    return {**dict(zip(sig["stock_id"], sig["name"])), **dict(zip(n["stock_id"], n["name"]))}


def run(days, acct, inputs, version=None):
    raw, adj, opn, vol, raw_open, sig, ctx, ind = inputs
    ver = version or VERSION
    v2 = ver in ("v2", "v3", "v4")
    acct.sleeve_w = SLEEVE_W.get(ver, SLEEVE_W["v3"])
    if v2:
        taiex, names = load_market_index()["taiex"], _names(sig)
    last_close = raw.ffill()   # 停牌幾天後才分割的，要和停牌前最後一個收盤比
    ma_ext = adj.rolling(EXT_MA).mean() if ver == "v4" and EXTEND else None
    for d in days:
        i = raw.index.get_loc(d)
        if i > 0 and len(acct.pos):  # 分割、減資：開盤價和昨收差超過漲跌幅限制，不是真的漲跌，股數和成本跟著換算
            for ix, h in acct.pos.iterrows():
                o_, pc = raw_open.loc[d].get(h["stock_id"], np.nan), last_close.iloc[i - 1].get(h["stock_id"], np.nan)
                if np.isfinite(o_) and np.isfinite(pc) and pc > 0 and abs(o_ / pc - 1) > 0.11:
                    f = o_ / pc
                    acct.pos.loc[ix, ["entry", "stop"]] = [h["entry"] * f, h["stop"] * f]
                    acct.pos.loc[ix, "shares"] = int(round(h["shares"] / f))
        acct.execute(d, raw_open.loc[d], raw.iloc[i - 1] if i > 0 else None)
        if ver in ("v3", "v4"):
            c, _ = pead.pick(d, sig, ctx, raw, adj, vol, ind, taiex, names)
            parts = [c.assign(sleeve="v2") if len(c) else c, s1_pick(d, sig, raw, adj, vol, ind, names)]
            if ver == "v4":
                parts.append(disp_pick(d, raw, vol, ind, names))
            parts = [x for x in parts if len(x)]
            c = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        elif v2:
            c, _ = pead.pick(d, sig, ctx, raw, adj, vol, ind, taiex, names)
        else:
            c, _ = pick(d, sig, ctx, raw, adj, vol, ind)
        acct.plan(d, raw.loc[d], c if len(c) else pd.DataFrame(columns=["stock_id"]), raw.iloc[i - 1] if i > 0 else None,
                  trend=(adj.loc[d], ma_ext.loc[d]) if ma_ext is not None else None)
    return acct


def stats(nav, taiex):
    nv = nav.set_index(pd.to_datetime(nav["date"]))["淨值"]
    tx = taiex.reindex(nv.index).ffill()
    dd = nv / nv.cummax() - 1
    m = nv.resample("ME").last().pct_change().dropna()
    return {"期間": f"{nv.index[0]:%Y-%m-%d} ~ {nv.index[-1]:%Y-%m-%d}", "帳戶報酬": f"{nv.iat[-1] / nv.iat[0] - 1:+.1%}",
            "加權指數": f"{tx.iat[-1] / tx.iat[0] - 1:+.1%}", "最大回撤": f"{dd.min():.1%}",
            "賺錢月份": f"{(m > 0).sum()}/{len(m)}", "最差月": f"{m.min():+.1%}" if len(m) else "-"}


# 校正規則（2026-11-23 定）。v4 門檻來自 2023-01～2026-11 的 v4 回測（scripts/v3_backtest.py，1,381 筆，results/持有天數/v4_trades.csv）
VERSION = "v4"                         # v1 用到 2026-11-23；v2 11-24～11-26；v3 只在 11-26 白天；v4（營收加速、一年新高、處置股各 1/3）從 2026-11-26 晚上起
EXTEND = True                          # v4 的續抱規則（2026-11-26 晚上起），回測最差回撤 -19.0%
DD_WATCH, DD_STOP = -0.14, -0.21      # v4 回測最差回撤 -17.7%（加續抱 -19.0%，門檻不變）
STREAK_WATCH, STREAK_STOP = 20, 30     # 連虧筆數：重抽 99 百分位 14，但處置股會在急跌時一起賠，回測實際最長 26，所以注意 20、暫停 30
ROLL_N, ROLL_MIN = 30, -0.020          # 最近 30 筆平均報酬低於 v4 回測重抽第 5 百分位


def check():
    """每晚檢查模擬帳戶是否超出回測預期。回傳 (等級, 訊息列表)；等級：正常／注意／暫停。
    暫停＝不再排新的買進（已持有的照規則出場），並且要重測規則。"""
    level, notes = "正常", []
    nav_p, tr_p = PAPER / "nav.csv", PAPER / "trades.csv"
    if nav_p.exists():
        nv = pd.read_csv(nav_p)["淨值"]
        dd = nv.iat[-1] / nv.cummax().iat[-1] - 1
        if dd <= DD_STOP:
            level = "暫停"
            notes.append(f"回撤 {dd:.0%}，超過回測最差 {DD_STOP:.0%}")
        elif dd <= DD_WATCH:
            level = "注意"
            notes.append(f"回撤 {dd:.0%}，接近回測最差 {DD_STOP:.0%}")
    if tr_p.exists():
        t = pd.read_csv(tr_p)
        streak = 0
        for x in t["報酬"][::-1]:
            if x > 0:
                break
            streak += 1
        if streak >= STREAK_STOP:
            level = "暫停"
            notes.append(f"連虧 {streak} 筆，超過回測 99% 的情況")
        elif streak >= STREAK_WATCH:
            level = "注意" if level == "正常" else level
            notes.append(f"連虧 {streak} 筆，超過回測 95% 的情況")
        if len(t) >= ROLL_N and t["報酬"].tail(ROLL_N).mean() < ROLL_MIN:
            level = "注意" if level == "正常" else level
            notes.append(f"最近 {ROLL_N} 筆平均 {t['報酬'].tail(ROLL_N).mean():+.1%}，低於回測的 5% 最差情況")
    PAPER.mkdir(parents=True, exist_ok=True)
    (PAPER / "check.txt").write_text(f"{VERSION}\t{level}\t{'；'.join(notes)}", encoding="utf-8")
    return level, notes


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "daily"
    inputs = _inputs()
    raw = inputs[0]
    taiex = load_market_index()["taiex"]
    if cmd == "backtest":
        start = pd.Timestamp(sys.argv[2] if len(sys.argv) > 2 else "2024-12-01")
        days = raw.index[raw.index >= start]
        acct = run(days, Account(), inputs)
        nav = pd.DataFrame(acct.nav)
        tr = pd.DataFrame(acct.trades)
        out = RESULTS / "paper_backtest"
        out.mkdir(parents=True, exist_ok=True)
        nav.to_csv(out / "nav.csv", index=False, encoding="utf-8-sig")
        tr.to_csv(out / "trades.csv", index=False, encoding="utf-8-sig")
        print(stats(nav, taiex))
        if len(tr):
            print(f"交易 {len(tr)} 筆，勝率 {(tr['報酬'] > 0).mean():.0%}，平均 {tr['報酬'].mean():+.1%}，"
                  f"出場原因 {tr['原因'].value_counts().to_dict()}")
        return
    acct = Account.load()
    last = pd.read_csv(PAPER / "nav.csv")["date"].max() if (PAPER / "nav.csv").exists() else None
    days = raw.index[raw.index > pd.Timestamp(last)] if last else raw.index[-1:]
    run(days, acct, inputs)
    acct.save()
    level, notes = check()
    if level == "暫停":  # 超出回測預期：取消新買進，只保留賣出
        acct.orders = acct.orders[acct.orders["side"] == "賣"]
        acct.orders.to_csv(PAPER / "orders.csv", index=False, encoding="utf-8-sig")
    print(f"校正檢查：{level}" + (f"（{'；'.join(notes)}）" if notes else ""))
    print(stats(pd.read_csv(PAPER / "nav.csv"), taiex))
    print(acct.orders.to_string(index=False) if len(acct.orders) else "明天沒有委託")


if __name__ == "__main__":
    main()
