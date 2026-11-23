"""滾動式校正驗證：假裝每季初只用「當時以前」的資料挑規則版本，用到下一季，再往後對答案。

驗證的是「定期校正」這個做法本身有沒有用，而不是某一組參數。
候選版本：停損（歷史統計/波動×3、固定 -15%、固定 -20%）× 空頭時買不買。
每個版本先完整跑一次模擬帳戶，得到每日淨值；每季初用「之前全部期間」的報酬÷最大回撤挑最好的版本，
下一季接上那個版本的每日報酬（近似：換版時假設持股跟著換）。
和「一直用 v1」「事後看最好的版本（偷看答案）」「加權指數」比較。
    python scripts/walk_forward_test.py 2024-12-01
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import twscreener.portfolio as P  # noqa: E402
from twscreener.market import load_market_index  # noqa: E402

start = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else "2024-12-01")
inputs = P._inputs()
days = inputs[0].index[inputs[0].index >= start]
orig = P.pick


def variant(stop=None, skip_bear=False):
    def pk(d, *a):
        c, reg = orig(d, *a)
        if skip_bear and reg == "空頭":
            return pd.DataFrame(), reg
        if stop is not None and len(c):
            c = c.assign(stop=stop)
        return c, reg
    P.pick = pk
    nav = pd.DataFrame(P.run(days, P.Account(), inputs).nav)
    return nav.set_index(pd.to_datetime(nav["date"]))["淨值"]


V = {"v1": {}, "v1+空頭不買": dict(skip_bear=True), "停損-15%": dict(stop=-0.15), "停損-15%+空頭不買": dict(stop=-0.15, skip_bear=True),
     "停損-20%": dict(stop=-0.20), "停損-20%+空頭不買": dict(stop=-0.20, skip_bear=True)}
navs = pd.DataFrame({k: variant(**kw) for k, kw in V.items()})
ret = navs.pct_change().fillna(0)


def score(n):
    dd = (n / n.cummax() - 1).min()
    total = n.iat[-1] / n.iat[0] - 1
    return total / abs(dd) if dd < 0 else total


quarters = pd.date_range(start, days[-1], freq="QS")
first_q = quarters[quarters >= start + pd.DateOffset(months=6)]  # 至少先有半年資料才開始挑
chosen, wf = [], []
for i, q in enumerate(first_q):
    end = first_q[i + 1] if i + 1 < len(first_q) else days[-1] + pd.Timedelta(days=1)
    past = navs[navs.index < q]
    best = max(V, key=lambda k: score(past[k]))
    chosen.append((q.date(), best))
    wf.append(ret.loc[(ret.index >= q) & (ret.index < end), best])
wf = pd.concat(wf)
wf_nav = (1 + wf).cumprod()
seg = navs[navs.index >= first_q[0]]
taiex = load_market_index()["taiex"].reindex(seg.index).ffill()


def summary(n):
    n = n / n.iat[0]
    return f"報酬 {n.iat[-1] - 1:+.1%}，最大回撤 {(n / n.cummax() - 1).min():.1%}"


print(f"對答案期間 {first_q[0].date()} ~ {days[-1].date()}（之前的資料只用來挑版本）")
print("每季挑的版本：", chosen)
print("滾動校正：", summary(wf_nav))
for k in V:
    print(f"一直用 {k}：", summary(seg[k]))
print("加權指數：", summary(taiex))

# 分年（每年各自從 1 起算）
out = pd.DataFrame({"滾動校正": wf_nav, "一直 v1": seg["v1"], "加權指數": taiex}).dropna()
print("\n分年報酬／最大回撤：")
for y, g in out.groupby(out.index.year):
    print(y, "  ".join(f"{k} {summary(g[k])}" for k in out.columns))
if len(sys.argv) > 2:
    navs.assign(滾動校正=wf_nav, 加權指數=load_market_index()["taiex"].reindex(navs.index)).to_csv(sys.argv[2])
