"""讀取本機 CSV，算出 M哥策略需要的指標。

所有欄位都只用「當天收盤後已知」的資料，避免偷看未來。
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MAS = [5, 10, 20, 60, 120, 240]


def _read(folder, name):
    path = DATA / folder / f"{name}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype={"stock_id": str})
    if df.empty:
        return None
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_stock(sid):
    """回傳一檔股票的日資料表（index=日期），沒有價格資料就回傳 None。"""
    px = _read("price", sid)
    if px is None:
        return None
    px = px[px["Trading_Volume"] > 0]
    df = pd.DataFrame({
        "open": px["open"].values,
        "high": px["max"].values,
        "low": px["min"].values,
        "close": px["close"].values,
        "vol": px["Trading_Volume"].values / 1000.0,  # 張
    }, index=px["date"].values).sort_index()
    df = df[~df.index.duplicated()]

    inst = _read("inst", sid)
    if inst is not None:
        inst["net"] = (inst["buy"] - inst["sell"]) / 1000.0
        piv = inst.pivot_table(index="date", columns="name", values="net", aggfunc="sum")
        df["foreign_net"] = piv.get("Foreign_Investor", pd.Series(dtype=float)).reindex(df.index)
        df["trust_net"] = piv.get("Investment_Trust", pd.Series(dtype=float)).reindex(df.index)

    fh = _read("foreign_hold", sid)
    if fh is not None:
        s = fh.drop_duplicates("date").set_index("date")["ForeignInvestmentShares"] / 1000.0
        df["foreign_hold"] = s.reindex(df.index).ffill()

    mg = _read("margin", sid)
    if mg is not None:
        s = mg.drop_duplicates("date").set_index("date")["MarginPurchaseTodayBalance"]
        df["margin_bal"] = s.reindex(df.index).ffill()

    for c in ["foreign_net", "trust_net"]:
        if c in df:
            df[c] = df[c].fillna(0.0)
        else:
            df[c] = 0.0
    for c in ["foreign_hold", "margin_bal"]:
        if c not in df:
            df[c] = np.nan
    df["trust_hold"] = df["trust_net"].cumsum()  # 投信沒有持股資料，用累計淨買超近似
    df["stock_id"] = sid
    return add_features(df)


def swing_low(low, n=10):
    """最近 n 日內的最低價（當作「前低」的簡化版）。"""
    return low.rolling(n).min()


def add_features(df):
    c, v = df["close"], df["vol"]
    for n in MAS:
        df[f"ma{n}"] = c.rolling(n).mean()
    df["above_all"] = np.logical_and.reduce([c > df[f"ma{n}"] for n in MAS])
    df["above_all_cnt10"] = df["above_all"].rolling(10).sum()

    # A：趨勢
    df["ma240_slope20"] = df["ma240"] - df["ma240"].shift(20)
    # 接下來 60 天會被扣掉的 240 日扣抵價（t-239 ~ t-180 的收盤），全部低於現價 → MA240 會往上
    df["deduct240_ahead_max"] = c.shift(180).rolling(60).max()
    df["A_up"] = df["ma240_slope20"] > 0
    df["A_turning"] = (c > df["deduct240_ahead_max"]) & (c > df["ma240"])
    df["A"] = df["A_up"] | df["A_turning"]

    # 量能
    df["vma5"] = v.rolling(5).mean()
    df["vma20"] = v.rolling(20).mean()
    df["vma120"] = v.rolling(120).mean()
    df["vol_q10_250"] = v.rolling(250).quantile(0.10)

    # B2：賣壓消失（量縮到相對極小 + 股價不再下跌）
    df["vol_dry"] = (df["vma5"] <= 0.5 * df["vma120"]) | (v <= df["vol_q10_250"])
    low10 = df["low"].rolling(10).min()
    low_prev = df["low"].shift(10).rolling(20).min()
    rng10 = c.rolling(10).max() / c.rolling(10).min() - 1
    df["price_stable"] = (low10 >= low_prev * 0.99) & (rng10 <= 0.10)
    df["B2_today"] = df["vol_dry"] & df["price_stable"]
    df["B2_recent"] = df["B2_today"].rolling(20).max().fillna(0).astype(bool)

    # C：量能增溫 + 站上全部均線（今天才站上）
    df["vol_warm"] = v > 1.5 * df["vma20"]
    df["C"] = df["above_all"] & ~df["above_all"].shift(1, fill_value=False) & df["vol_warm"]

    # 一年新高
    df["high250"] = c >= c.rolling(250).max()
    df["days_since_high250"] = _days_since(df["high250"].shift(1, fill_value=False))

    # B1：主要力量（外資持股、投信累計、融資餘額）
    for name, col in [("foreign", "foreign_hold"), ("trust", "trust_hold"), ("margin", "margin_bal")]:
        s = df[col]
        df[f"{name}_chg120"] = s - s.shift(120)
        df[f"{name}_chg20"] = s - s.shift(20)
        df[f"{name}_chg60"] = s - s.shift(60)
        df[f"{name}_peak250"] = s.rolling(250, min_periods=60).max()
        df[f"{name}_gain250"] = s - s.rolling(250, min_periods=60).min()
    # 正規化：用 120 日均量當分母，不同大小的股票才能比較
    for name in ["foreign", "trust", "margin"]:
        df[f"{name}_chg60_n"] = df[f"{name}_chg60"] / (df["vma120"] * 60)

    # 「沒有真的鬆動」：近 20 日減少量 < 過去一年累積增加量的 1/3
    def holds(name):
        loss = (-df[f"{name}_chg20"]).clip(lower=0)
        return (df[f"{name}_chg120"] > 0) & (loss < df[f"{name}_gain250"] / 3)

    df["B1_foreign"] = holds("foreign")
    df["B1_trust"] = holds("trust")
    df["B1_margin"] = holds("margin")
    df["B1"] = df["B1_foreign"] | df["B1_trust"] | df["B1_margin"]

    # 三破用：近 60 日最大量那根 K 棒的最低價
    idx = v.rolling(60).apply(np.argmax, raw=True)
    lows = df["low"].values
    bigbar_low = np.full(len(df), np.nan)
    for i, k in enumerate(idx.values):
        if not np.isnan(k):
            bigbar_low[i] = lows[i - 59 + int(k)]
    df["bigbar_low"] = bigbar_low
    df["swing_low_prev"] = swing_low(df["low"].shift(1), 20)
    df["ret5"] = c.pct_change(5)
    df["dd_from_high60"] = c / c.rolling(60).max() - 1
    return df


def _days_since(flag):
    out = np.zeros(len(flag))
    cnt = 9999
    for i, f in enumerate(flag.values):
        cnt = 0 if f else cnt + 1
        out[i] = cnt
    return pd.Series(out, index=flag.index)


def load_market():
    """加權指數 + 全市場外資買賣超，回傳 index=日期 的表。"""
    tx = _read("market", "taiex")
    out = pd.DataFrame()
    if tx is not None:
        out = tx.drop_duplicates("date").set_index("date")[["close"]].rename(columns={"close": "taiex"})
        out["taiex_ret5"] = out["taiex"].pct_change(5)
    mi = _read("market", "market_inst")
    if mi is not None:
        mi = mi[mi["name"].str.contains("Foreign_Investor")]
        net = (mi["buy"] - mi["sell"]).groupby(mi["date"]).sum() / 1e8  # 億元
        out = out.join(net.rename("mkt_foreign_net"), how="outer")
        out["mkt_foreign_net5"] = out["mkt_foreign_net"].rolling(5).sum()
        out["mkt_foreign_net5_q05"] = out["mkt_foreign_net5"].rolling(500, min_periods=120).quantile(0.05)
    return out.sort_index()
