"""四種只需要日資料的策略訊號（對應《M哥策略手冊》S1、S2、S5、S6）。

每個函式吃一檔股票的特徵表（features.load_stock 的輸出）與市場表，
回傳布林 Series：當天收盤後出現訊號，隔天開盤進場。
"""
import numpy as np
import pandas as pd

FORCES = ["foreign", "trust", "margin"]


def main_force(df):
    """近 60 日哪一方加碼最多（用 120 日均量正規化）。"""
    cols = [f"{f}_chg60_n" for f in FORCES]
    x = df[cols].fillna(-np.inf)
    return x.idxmax(axis=1).str.replace("_chg60_n", "", regex=False)


def s1_new_high(df, mkt=None):
    """S1 一年新高：沉寂 60 天以上後首次創 250 日新高，240MA 走平或轉上，且有一方力量明顯加碼。"""
    force_in = df[[f"{f}_chg60_n" for f in FORCES]].max(axis=1) > 0.02
    return df["high250"] & (df["days_since_high250"] >= 60) & df["A"] & force_in


def s2_abc(df, mkt=None):
    """S2 A+B1+B2+C：長多 + 大錢沒跑 + 近 20 日出現過賣壓消失 + 今天出量站上全部均線。"""
    return df["A"] & df["B1"] & df["B2_recent"] & df["C"]


def s5_contrarian(df, mkt):
    """S5 逆勢籌碼：大盤 5 日跌超過 5% 或外資 5 日賣超處於歷史最低 5%，
    個股的外資或融資 5 日卻是增加，且個股 5 日表現強於大盤。"""
    if mkt is None or mkt.empty:
        return pd.Series(False, index=df.index)
    m = mkt.reindex(df.index)
    stress = (m["taiex_ret5"] <= -0.05)
    if "mkt_foreign_net5" in m:
        stress = stress | (m["mkt_foreign_net5"] <= m["mkt_foreign_net5_q05"])
    f5 = df["foreign_net"].rolling(5).sum() > 0
    mg5 = (df["margin_bal"] - df["margin_bal"].shift(5)) > 0
    stronger = df["ret5"] > m["taiex_ret5"]
    sig = stress.fillna(False) & (f5 | mg5) & stronger.fillna(False)
    # 同一波壓力只取第一天
    return sig & ~sig.rolling(20, min_periods=1).max().shift(1, fill_value=0).astype(bool)


def s6_shakeout(df, mkt=None):
    """S6 心理黑洞後回補：近 60 日從高點跌超過 15%，跌到 120~240MA 附近，
    長期力量沒有鬆動，之後出量站上全部均線（C）。"""
    dipped = (df["dd_from_high60"] <= -0.15)
    near_long_ma = (df["low"] <= df["ma120"] * 1.03) & (df["low"] >= df["ma240"] * 0.95)
    setup = (dipped & near_long_ma).rolling(20).max().fillna(0).astype(bool)
    return setup & df["B1"] & df["C"]


def t1_trust_streak(df, mkt=None):
    """投信連買：投信連續 5 天買超，5 天合計至少是 20 日均量的 30%，只取剛成立的第一天。"""
    buy = df["trust_net"] > 0
    streak = buy.rolling(5).sum() == 5
    big = df["trust_net"].rolling(5).sum() >= 0.3 * df["vma20"]
    sig = streak & big
    return sig & ~sig.shift(1, fill_value=False)


STRATEGIES = {
    "S1_一年新高": s1_new_high,
    "S2_ABC拉回": s2_abc,
    "S5_逆勢籌碼": s5_contrarian,
    "S6_心理黑洞": s6_shakeout,
    "T1_投信連買": t1_trust_streak,
}

# 每個策略至少要多少天的歷史才開始算訊號（M哥的策略要看年線，需要一年）
MIN_HIST = {"T1_投信連買": 60}
MOMENTUM = "T2_動能"


def three_breaks(df):
    """三破：跌破 20MA、跌破近 60 日最大量 K 棒低點、跌破前 20 日低點，三者同時成立。"""
    c = df["close"]
    return (c < df["ma20"]) & (c < df["bigbar_low"]) & (c < df["swing_low_prev"])
