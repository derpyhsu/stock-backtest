"""每晚把今天的選股推播到 Telegram。

    python -m twscreener.notify          # 送出 results/picks_today.csv 的摘要
    python -m twscreener.notify --dry    # 只印出訊息，不送

需要環境變數 TELEGRAM_BOT_TOKEN（雲端環境設定裡填，不可以寫進檔案或 commit）。
收件人 TELEGRAM_CHAT_ID 可以用環境變數給；沒給的話，第一次會從機器人收到的訊息找出來，
存在 .env（不上傳 GitHub）。
"""
import html
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from .market import load_market_index
from .sizing import size

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
ENV = ROOT / ".env"
DASHBOARD_URL = "https://derpyhsu.github.io/stock-backtest/dashboard/"


def _api(method, **params):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("沒有 TELEGRAM_BOT_TOKEN，略過推播")
    data = urllib.parse.urlencode(params).encode() if params else None
    req = urllib.request.Request(f"https://api.telegram.org/bot{token}/{method}", data=data)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:  # 錯誤訊息裡不會有 token，只印 Telegram 的說明
        raise SystemExit(f"Telegram {method} 失敗：HTTP {e.code} {e.read().decode(errors='replace')[:200]}")


def _env_value(key):
    if ENV.exists():
        for line in ENV.read_text(encoding="utf-8").splitlines():
            if line.startswith(key + "="):
                return line.split("=", 1)[1].strip()
    return None


def chat_id():
    cid = os.environ.get("TELEGRAM_CHAT_ID") or _env_value("TELEGRAM_CHAT_ID")
    if cid:
        return cid
    chats = [u["message"]["chat"]["id"] for u in _api("getUpdates").get("result", [])
             if "message" in u and u["message"]["chat"].get("type") == "private"]
    if not chats:
        raise SystemExit("機器人還沒收到訊息：請先在 Telegram 打開機器人按「開始」並傳一句話")
    cid = str(chats[-1])
    with ENV.open("a", encoding="utf-8") as f:
        f.write(f"\nTELEGRAM_CHAT_ID={cid}\n")
    return cid


SEP = "━━━━━━━━━━━━"
CIRCLE = "①②③④⑤⑥⑦⑧⑨"
e = lambda x: html.escape(str(x))


def price(v):
    """依台股升降單位取到實際掛得出去的價位（例如 343.69 → 343.5、39.27 → 39.25），再去掉多餘的 0。"""
    if pd.isna(v):
        return "-"
    tick = 0.01 if v < 10 else 0.05 if v < 50 else 0.1 if v < 100 else 0.5 if v < 500 else 1 if v < 1000 else 5
    v = round(round(v / tick) * tick, 2)
    if v >= 1000:
        return f"{v:,.0f}"
    return f"{v:.2f}".rstrip("0").rstrip(".")


def short(text, n=2):
    """理由只留前幾段（用；分隔），手機上比較好讀。"""
    parts = [t for t in str(text).split("；") if t and t != "nan"]
    return "；".join(parts[:n])


WEEK = "一二三四五六日"
LIMIT = 4000  # Telegram 一則上限 4096 字


def short_qty(entry, stop):
    """每 100 萬買多少：'2 張'、'零股 110 股'。"""
    z = size(entry, stop)
    if not z or z["shares"] == 0:
        return "-", 0
    if z["lots"] >= 1:
        q = f"{z['lots']} 張" + (f" {z['odd']} 股" if z["odd"] >= 100 else "")
    else:
        q = f"{z['shares']} 股"
    return q, z["amount"]


def day_label(d):
    d = pd.Timestamp(d)
    return f"{d.month}/{d.day}（{WEEK[d.weekday()]}）"


def next_trading_day(d):
    d = pd.Timestamp(d) + pd.Timedelta(days=1)
    while d.weekday() >= 5:
        d += pd.Timedelta(days=1)
    return d


def market_info():
    path = RESULTS / "candidates_today.csv"
    regime, dd = None, None
    if path.exists():
        c = pd.read_csv(path)
        regime = c["大盤狀態"].iat[0] if "大盤狀態" in c and len(c) else None
    try:
        from .review import market_context
        dd = market_context()["大盤離高點"].dropna().iat[-1]
    except Exception:
        pass
    return regime, dd


def load_long():
    """今天的中長線推薦（同一天重選時用最新收盤那一批）與否決，附停損目標。"""
    path = RESULTS / "agent_picks_log.csv"
    if not path.exists():
        return pd.DataFrame(), pd.DataFrame(), {}
    log = pd.read_csv(path, dtype={"stock_id": str})
    if log.empty:
        return pd.DataFrame(), pd.DataFrame(), {}
    day = log["pick_date"].max()
    today = log[log["pick_date"] == day]
    today = today[today["signal_date"] == today["signal_date"].max()]
    lv = {}
    lpath = RESULTS / "agent_levels_today.csv"
    if lpath.exists():
        lt = pd.read_csv(lpath, dtype={"stock_id": str})
        lv = {r["stock_id"]: r for _, r in lt[lt["pick_date"] == day].iterrows()}
    return today[today["verdict"] == "推薦"], today[today["verdict"] == "否決"], lv


def load_short():
    path = RESULTS / "shortterm_picks_log.csv"
    if not path.exists():
        return pd.DataFrame(), pd.DataFrame(), {}
    log = pd.read_csv(path, dtype={"stock_id": str})
    if log.empty:
        return pd.DataFrame(), pd.DataFrame(), {}
    today = log[log["pick_date"] == log["pick_date"].max()]
    odds = {}
    tp = RESULTS / "shortterm_today.csv"
    if tp.exists():
        t = pd.read_csv(tp, dtype={"stock_id": str})
        if "先到停利" in t:
            odds = t.set_index("stock_id")[["先到停利", "先到停損"]].to_dict("index")
    return today[today["verdict"] == "推薦"], today[today["verdict"] == "否決"], odds


def pct(a, b):
    return f"{a / b - 1:+.0%}"


def rr_text(target, close, stop):
    rr = (target - close) / (close - stop) if close > stop else float("nan")
    return f"{rr:.1f}", round(rr, 1) < 1


def msg_overview(day, long_p, lv, short_p, picks):
    regime, dd = market_info()
    nxt = next_trading_day(day)
    lines = [f"📈 <b>台股日報｜{day_label(day)}收盤</b>"]
    if regime:
        lines.append(f"大盤 <b>{e(regime)}</b>" + (f"・距高點 {dd:+.0%}" if dd is not None else ""))
    lines.append(f"<i>適用 {day_label(nxt)}開盤；開盤就跌破停損的不要進場</i>")
    if len(long_p):
        lines += ["", "🏛 <b>中長線</b>・抱約 60 天"]
        for i, r in enumerate(long_p.itertuples()):
            x = lv.get(r.stock_id)
            tail = f"停損 {price(x['停損價'])}・目標 {price(x['60日目標價'])}" if x is not None else ""
            lines.append(f"{CIRCLE[i]} <b>{e(r.name)}</b> {price(r.close)}　{tail}")
    if len(short_p):
        lines += ["", "⚡ <b>短線</b>・抱 5～10 天"]
        for i, r in enumerate(short_p.itertuples()):
            lines.append(f"{CIRCLE[i]} <b>{e(r.name)}</b> {price(r.close)}　停損 {price(r.stop)}・停利 {price(r.tp10)}")
    if not len(long_p) and not len(short_p):
        lines += ["", "今天沒有推薦。" if not len(picks) else f"今天沒有 agent 推薦，規則訊號 {len(picks)} 檔次見下面。"]
    lines += ["", "👇 每檔的理由、風險和要買多少在下面"]
    return "\n".join(lines)


def card(i, r, close, stop, rows, reason, risk, qty_line):
    lines = [f"{CIRCLE[i]} <b>{e(r.name)}</b> {r.stock_id}　<i>{e(r.industry)}</i>"]
    body = [f"收盤 {price(close)}　🛑 停損 <b>{price(stop)}</b> {pct(stop, close)}"] + rows + [qty_line]
    lines.append("<blockquote>" + "\n".join(body) + "</blockquote>")
    lines.append(f"✅ {e(short(reason))}")
    if str(risk) not in ("", "nan"):
        lines.append(f"⚠️ {e(short(risk, 1))}")
    return "\n".join(lines)


def qty_text(close, stop, half):
    q, amt = short_qty(close, stop)
    note = "（報酬風險比不到 1，建議再減半）" if half else ""
    return f"💰 每 100 萬買 {q}，約 {amt / 1e4:.0f} 萬{note}"


def msg_long(long_p, lv):
    if not len(long_p):
        return None
    out = ["🏛 <b>中長線推薦</b>・抱約 60 天"]
    for i, r in enumerate(long_p.itertuples()):
        x = lv.get(r.stock_id)
        if x is None:
            out.append(f"{CIRCLE[i]} <b>{e(r.name)}</b> {r.stock_id}\n✅ {e(short(r.reason))}")
            continue
        rr, half = rr_text(x["60日停利價"], r.close, x["停損價"])
        rows = [f"🎯 60 天目標 <b>{price(x['60日目標價'])}</b>・樂觀 {price(x['60日停利價'])}・報酬風險比 {rr}",
                "📅 " + "・".join(f"{h}天 {price(x[f'{h}日目標價'])}" for h in (30, 90, 120))]
        out.append(card(i, r, r.close, x["停損價"], rows, r.reason, r.risk, qty_text(r.close, x["停損價"], half)))
    return "\n\n".join(out)


def msg_short(short_p, odds):
    if not len(short_p):
        return None
    out = ["⚡ <b>短線推薦</b>・抱 5～10 天，第 10 天收盤沒到停利就出場"]
    for i, r in enumerate(short_p.itertuples()):
        rr, half = rr_text(r.tp10, r.close, r.stop)
        rows = [f"🎯 停利 5 天 <b>{price(r.tp5)}</b>・10 天 <b>{price(r.tp10)}</b>・報酬風險比 {rr}"]
        o = odds.get(r.stock_id)
        if o:
            rows.append(f"🎲 過去同訊號：先到停利 {o['先到停利']:.0%}・先到停損 {o['先到停損']:.0%}")
        out.append(card(i, r, r.close, r.stop, rows, r.reason, r.risk, qty_text(r.close, r.stop, half)))
    return "\n\n".join(out)


def msg_paper():
    """模擬帳戶：完全照規則買賣（twscreener/portfolio.py）。"""
    d = RESULTS / "paper"
    if not (d / "nav.csv").exists():
        return None
    nav = pd.read_csv(d / "nav.csv")
    if nav.empty:
        return None
    first, last = nav.iloc[0], nav.iloc[-1]
    lines = ["🤖 <b>模擬帳戶</b>・完全照規則，不經人手"]
    body = [f"淨值 <b>{last['淨值'] / 1e4:,.1f} 萬</b>（起始 {first['淨值'] / 1e4:,.0f} 萬，{last['淨值'] / first['淨值'] - 1:+.1%}）",
            f"持股 {int(last['持股數'])} 檔・現金 {last['現金'] / 1e4:,.1f} 萬"]
    try:
        tx = load_market_index()["taiex"]
        t0, t1 = tx.asof(pd.Timestamp(first["date"])), tx.asof(pd.Timestamp(last["date"]))
        body.append(f"同期加權指數 {t1 / t0 - 1:+.1%}")
    except Exception:
        pass
    lines.append("<blockquote>" + "\n".join(body) + "</blockquote>")
    ck = d / "check.txt"
    if ck.exists():
        ver, level, why = (ck.read_text(encoding="utf-8").split("\t") + ["", "", ""])[:3]
        if level == "暫停":
            lines.append(f"⛔ <b>規則 {e(ver)} 暫停新買進</b>：{e(why)}。我會重測規則後再恢復。")
        elif level == "注意":
            lines.append(f"🟡 {e(why)}，還在回測預期內，照規則繼續。")
    o = d / "orders.csv"
    orders = pd.read_csv(o, dtype={"stock_id": str}) if o.exists() and o.stat().st_size > 0 else pd.DataFrame()
    if len(orders):
        lines.append("明天開盤：")
        for r in orders.itertuples():
            if r.side == "賣":
                lines.append(f"・🔴 賣 <b>{e(r.name)}</b> {int(r.shares):,} 股（{e(r.reason)}）")
            else:
                lines.append(f"・🟢 買 <b>{e(r.name)}</b> {int(r.shares):,} 股，停損 {price(r.stop)}（開盤低於停損就不買）")
    else:
        lines.append("明天沒有買賣。")
    return "\n".join(lines)


def msg_more(long_v, short_v, picks):
    """名詞說明，收在可展開的區塊裡。沒選的股票和規則訊號不推：收到的人本來就不知道它們，看了也沒用（看板上有）。"""
    lines = []
    lines.append("📖 <b>怎麼看</b>")
    lines.append("<blockquote expandable>"
                 "目標＝同組歷史報酬中位數，樂觀＝前 25%，當參考，不要掛單賣（回測發現固定停利會砍掉大漲股）。\n"
                 "停損＝八成賺錢的歷史訊號沒跌破的價位，盤勢亂時依波動放寬；短線停損＝近 20 日平均波動 × 3。\n"
                 "買多少＝打到停損最多虧總資金 1%、單檔最多 20%，依自己的資金等比例換算。\n"
                 "報酬風險比＝（樂觀停利 − 收盤）÷（收盤 − 停損），不到 1 部位再減半。"
                 "</blockquote>")
    lines.append(f'👉 <a href="{DASHBOARD_URL}">打開看板</a>・<i>歷史統計加新聞判斷，不保證未來</i>')
    return "\n".join(lines)


def build_messages():
    path = RESULTS / "picks_today.csv"
    picks = pd.read_csv(path, dtype={"stock_id": str}) if path.exists() else pd.DataFrame()
    day = picks["date"].iat[0] if len(picks) else pd.Timestamp.today().date()
    long_p, long_v, lv = load_long()
    short_p, short_v, odds = load_short()
    msgs = [m for m in (msg_overview(day, long_p, lv, short_p, picks), msg_long(long_p, lv), msg_short(short_p, odds),
                        msg_paper()) if m]
    msgs[-1] += "\n\n" + msg_more(long_v, short_v, picks)  # 說明接在最後一則後面，不另發
    out = []
    for m in msgs:
        if m and len(m) > LIMIT:  # 太長就從段落切開
            chunk = ""
            for part in m.split("\n\n"):
                if chunk and len(chunk) + len(part) + 2 > LIMIT:
                    out.append(chunk)
                    chunk = ""
                chunk = f"{chunk}\n\n{part}" if chunk else part
            m = chunk
        if m:
            out.append(m)
    return out


def main():
    msgs = build_messages()
    if "--preview" in sys.argv:
        msgs[0] = "🧪 <i>排版預覽（內容是上週五的推薦）</i>\n\n" + msgs[0]
    if "--dry" in sys.argv:
        print("\n\n========== 下一則 ==========\n\n".join(msgs))
        return
    cid = chat_id()
    for m in msgs:
        _api("sendMessage", chat_id=cid, text=m, parse_mode="HTML", disable_web_page_preview="true")
    print(f"已推播到 Telegram（{len(msgs)} 則）")


if __name__ == "__main__":
    main()
