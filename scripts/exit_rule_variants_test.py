"""模擬帳戶 v4 持有中「每天重新檢查要不要走」的幾種規則，和現在的固定規則比（2026-11-26）。
現在：收盤跌破買進價 -15% 賣、抱滿 60 個交易日賣（處置股那份另計，這裡不動）。
  A 現在的 v4
  B 再加「破線」：收盤跌破前 20 日最低收盤、而且在 20 日線下 → 隔天開盤賣
  C 再加「外資鬆動」：外資過去 250 日買超占量 > 2%，但近 20 日賣掉超過其中 1/3 → 隔天開盤賣
  D 滿 60 天時收盤還在 60 日線上就續抱，跌破 60 日線或滿 120 天才賣
  E B＋D
  F B＋C＋D
判斷都只用當天收盤以前的資料。
    python scripts/exit_rule_variants_test.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import twscreener.portfolio as P  # noqa: E402
from twscreener.market import load_market_index, load_panel  # noqa: E402

inputs = P._inputs()
raw, adj, opn, vol = inputs[:4]
days = raw.index
p = load_panel()
fn = p.pivot_table(index="date", columns="stock_id", values="foreign_net").reindex(index=days, columns=raw.columns).fillna(0)
c = adj
ma20, ma60 = c.rolling(20).mean(), c.rolling(60).mean()
brk = (c < c.shift(1).rolling(20).min()) & (c < ma20)
b250, s20 = fn.rolling(250).sum(), fn.rolling(20).sum()
loose = (b250 / vol.rolling(250).sum() > 0.02) & (s20 < -b250 / 3)
above60 = c > ma60
pos_of = {d: i for i, d in enumerate(days)}
MODE = {"m": "A"}
MAXD = 120
orig_plan = P.Account.plan


def flag(tbl, day, sid):
    try:
        return bool(tbl.at[day, sid])
    except KeyError:
        return False


def plan(self, day, close, cands, prev=None):
    m = MODE["m"]
    if m != "A" and len(self.pos):
        for idx, h in self.pos.iterrows():
            if pd.notna(h["exit_date"]):
                continue
            sid = h["stock_id"]
            out = (m in "BEF" and flag(brk, day, sid)) or (m in "CF" and flag(loose, day, sid))
            if out:
                self.pos.at[idx, "stop"] = 1e9   # 今天收盤一定低於它 → 照停損流程隔天開盤賣
            elif m in "DEF" and h["days"] + 1 >= P.HOLD_DAYS:
                held = pos_of[day] - pos_of.get(pd.Timestamp(h["entry_date"]), pos_of[day])
                if held < MAXD and flag(above60, day, sid):
                    self.pos.at[idx, "days"] = P.HOLD_DAYS - 2   # 續抱
    return orig_plan(self, day, close, cands, prev)


P.Account.plan = plan
taiex = load_market_index()["taiex"].reindex(days).ffill()
navs, info = {}, {}
for m in "ABCDEF":
    MODE["m"] = m
    a = P.run(days[days >= "2023-01-01"], P.Account(), inputs, version="v4")
    n = pd.DataFrame(a.nav)
    navs[m] = n.set_index(pd.to_datetime(n["date"]))["淨值"]
    t = pd.DataFrame(a.trades)
    s = t[t["strategy"] != "處置股"]
    hold = (pd.to_datetime(s["賣出日"]) - pd.to_datetime(s["買進日"])).dt.days.mean()
    info[m] = f"非處置股 {len(s)} 筆、勝率 {(s['報酬'] > 0).mean():.0%}、平均 {s['報酬'].mean():+.1%}、平均抱 {hold:.0f} 天"
navs = pd.DataFrame(navs)
navs["加權指數"] = taiex.reindex(navs.index)
seg = navs[navs.index >= "2023-04-01"]


def summ(s):
    s = s.dropna() / s.dropna().iat[0]
    return f"{s.iat[-1] - 1:+.1%}／{(s / s.cummax() - 1).min():.1%}"


lines = [f"期間 {seg.index[0].date()} ~ {seg.index[-1].date()}（報酬／最大回撤）"]
for k in seg.columns:
    yr = "  ".join(f"{y} {summ(g[k])}" for y, g in seg.groupby(seg.index.year))
    lines.append(f"{k:5s} 全期 {summ(seg[k])}  {yr}  {info.get(k, '')}")
text = "\n".join(lines)
print(text)
out = ROOT / "results" / "手冊未測型態"
(out / "exit_rule_variants.txt").write_text(text, encoding="utf-8")
