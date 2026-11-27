"""每晚把今天的選股推播到 Telegram。

    python -m twscreener.notify          # 送出 results/picks_today.csv 的摘要
    python -m twscreener.notify --dry    # 只印出訊息，不送
    python -m twscreener.notify --file results/morning/2026-11-25.txt   # 推一則寫好的訊息（早上美股提醒）

需要環境變數 TELEGRAM_BOT_TOKEN（雲端環境設定裡填，不可以寫進檔案或 commit）。
收件人存在 .env 的 TELEGRAM_CHAT_ID（逗號分隔，不上傳 GitHub）。任何人私訊機器人就會在下次推播前
自動加入，傳 /stop 退出；也可以用環境變數 TELEGRAM_CHAT_ID 固定加人。
"""
import html
import json
import time
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from .market import load_market_index
from .sizing import size

REBOUND, REBOUND_HOLD = "處置跌深", 20   # 和 portfolio.py 的 REBOUND、REBOUND_HOLD 一致

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
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:  # 錯誤訊息裡不會有 token，只印 Telegram 的說明
            raise SystemExit(f"Telegram {method} 失敗：HTTP {e.code} {e.read().decode(errors='replace')[:200]}")
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:  # 連線被重設：等一下再試，不印網址（裡面有 token）
            if attempt == 3:
                raise SystemExit(f"Telegram {method} 失敗：連線錯誤 {type(e).__name__}")
            time.sleep(5 * (attempt + 1))


def _env_value(key):
    if ENV.exists():
        for line in ENV.read_text(encoding="utf-8").splitlines():
            if line.startswith(key + "="):
                return line.split("=", 1)[1].strip()
    return None


WELCOME = ("✅ 已加入每日推播：每晚約 22:00 送當天選股，早上 08:15 送美股提醒。\n"
           "不想收了傳 /stop。內容是歷史回測統計，不保證未來。")
BYE = "已停止推播。想再收的話隨時傳一句話給我。"


def _save_ids(ids):
    """收件人存在 .env 的 TELEGRAM_CHAT_ID（不上傳 GitHub）。"""
    lines = ENV.read_text(encoding="utf-8").splitlines() if ENV.exists() else []
    lines = [l for l in lines if not l.startswith("TELEGRAM_CHAT_ID=")]
    lines.append("TELEGRAM_CHAT_ID=" + ",".join(ids))
    ENV.write_text("\n".join(lines) + "\n", encoding="utf-8")


def chat_ids():
    """收件人名單。每次推播前看機器人最近 24 小時收到的私訊：
    傳過任何訊息的人自動加入（並回一則歡迎），傳 /stop 的人移除。
    早晚各推一次、間隔不到 24 小時，所以不會漏掉。"""
    fixed = [c.strip() for c in (os.environ.get("TELEGRAM_CHAT_ID") or "").split(",") if c.strip()]
    saved = [c.strip() for c in (_env_value("TELEGRAM_CHAT_ID") or "").split(",") if c.strip()]
    ids = list(dict.fromkeys(saved))
    joined, left = [], []
    for u in _api("getUpdates").get("result", []):
        m = u.get("message") or {}
        if m.get("chat", {}).get("type") != "private":
            continue
        cid, text = str(m["chat"]["id"]), (m.get("text") or "").strip().lower()
        if text == "/stop":
            if cid in ids:
                ids.remove(cid)
                left.append(cid)
        elif cid not in ids:
            ids.append(cid)
            joined.append(cid)
    if joined or left:
        _save_ids(ids)
    for cid in joined:
        _api("sendMessage", chat_id=cid, text=WELCOME)
    for cid in left:
        _api("sendMessage", chat_id=cid, text=BYE)
    if joined or left:
        print(f"推播名單：新加入 {len(joined)} 人、退出 {len(left)} 人")
    ids = list(dict.fromkeys(fixed + ids))
    if not ids:
        raise SystemExit("機器人還沒收到訊息：請先在 Telegram 打開機器人按「開始」並傳一句話")
    return ids


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


def load_v2_buys():
    """模擬帳戶（v3：營收加速、一年新高各半）明天要買的股票＝帳戶的買進委託，推播和帳戶同一套規則。"""
    o = RESULTS / "paper" / "orders.csv"
    if not o.exists() or o.stat().st_size == 0:
        return pd.DataFrame()
    orders = pd.read_csv(o, dtype={"stock_id": str})
    return orders[orders["side"] == "買"]


def next_v2_day(day):
    """下一個 v2 訊號日（約略）：月營收 10 號公布完，11 號後第一個交易日。"""
    d = pd.Timestamp(day)
    t = pd.Timestamp(d.year, d.month, 11)
    if d >= t:
        t = t + pd.offsets.MonthBegin(1) + pd.Timedelta(days=10)
    while t.weekday() >= 5:
        t += pd.Timedelta(days=1)
    return t


_CHIPS = {}


def chips_line(sid):
    """籌碼面一行：外資、投信近 5 日買賣超和融資近 5 日增減（張），直接從 data/daily 最近 6 天算，不經人手。"""
    if "d" not in _CHIPS:
        files = sorted((RESULTS.parent / "data" / "daily").glob("*.csv"))[-6:]
        _CHIPS["d"] = pd.concat([pd.read_csv(f, dtype={"stock_id": str}) for f in files]) if files else pd.DataFrame()
    d = _CHIPS["d"]
    x = d[d["stock_id"] == sid].sort_values("date") if len(d) else d
    if len(x) < 2:
        return ""
    w = x.tail(5)
    f5, t5, mg = w["foreign_net"].sum(), w["trust_net"].sum(), x["margin_bal"].iat[-1] - x["margin_bal"].iat[-len(w) - 1 if len(x) > len(w) else 0]
    word = lambda v: f"買超 {v:,.0f} 張" if v > 0 else f"賣超 {-v:,.0f} 張" if v < 0 else "沒有進出"  # noqa: E731
    parts = [f"外資 5 日{word(f5)}", f"投信 5 日{word(t5)}"]
    if mg == mg:
        parts.append(f"融資 5 日{'增' if mg >= 0 else '減'} {abs(mg):,.0f} 張")
    return "📊 " + "、".join(parts)


def news_map(day):
    """消息面：每晚 agent 查的新聞（results/paper/news.csv：date, stock_id, name, kind, news, link）。只當佐證，不影響買賣。"""
    f = RESULTS / "paper" / "news.csv"
    if not f.exists() or f.stat().st_size == 0:
        return {}
    n = pd.read_csv(f, dtype={"stock_id": str})
    n = n[n["date"].astype(str) == str(pd.Timestamp(day).date())]
    return {r.stock_id: (str(r.news), str(getattr(r, "link", "") or "")) for r in n.itertuples()}


def news_line(nm, sid):
    if sid not in nm:
        return ""
    text, link = nm[sid]
    tail = f' <a href="{html.escape(link, quote=True)}">來源</a>' if link.startswith("http") else ""
    return f"📰 {e(text)}{tail}"


HORIZON = (("⚡ 短期・約 5～10 天", lambda g: g == "處置股"), ("📆 中期・20 個交易日", lambda g: g == REBOUND),
           ("🌲 長期・60～120 天", lambda g: g not in ("處置股", REBOUND)))


def horizon_groups(v2):
    """買進訊號依預計抱多久分三區：短期＝處置股、中期＝處置跌深、長期＝營收加速和一年新高。"""
    rows = list(v2.itertuples())
    return [(t, [r for r in rows if f(r.strategy)]) for t, f in HORIZON if any(f(r.strategy) for r in rows)]


def paper_nav():
    f = RESULTS / "paper" / "nav.csv"
    try:
        return float(pd.read_csv(f)["淨值"].iat[-1])
    except Exception:
        return float("nan")


def msg_overview(day, v2, long_p, lv, short_p, picks):
    regime, dd = market_info()
    nxt = next_trading_day(day)
    lines = [f"📈 <b>台股日報｜{day_label(day)}收盤</b>"]
    if regime:
        lines.append(f"大盤 <b>{e(regime)}</b>" + (f"・距高點 {dd:+.0%}" if dd is not None else ""))
    lines.append(f"<i>適用 {day_label(nxt)}開盤；開盤就跌破停損的不要進場</i>")
    lines += ["", "🎯 <b>明天開盤的訊號</b>・照固定規則選出"]
    if len(v2):
        n = 0
        for title, rows in horizon_groups(v2):
            lines.append(f"<b>{title}</b>")
            for r in rows:
                xd = getattr(r, "exit_date", None)
                tail = (f"抱 {REBOUND_HOLD} 個交易日收盤賣" if r.strategy == REBOUND
                        else f"{md(xd)} 收盤賣" if isinstance(xd, str) and xd else f"停損 {price(r.stop)}")
                lines.append(f"{CIRCLE[n % len(CIRCLE)]} <b>{e(r.name)}</b>　{e(r.strategy)}・{tail}")
                n += 1
    else:
        lines.append(f"明天沒有新訊號（今天沒有處置第 2 天、處置跌深和一年新高訊號，或那一類的位子已滿；營收加速下次約 {day_label(next_v2_day(day))}）。")
    ann = announced_today(day)
    if ann:
        lines.append(f"📌 今天公告處置（處置第 2 天開盤才買，明晚排單）：{e('、'.join(ann[:8]))}" + (f" 等 {len(ann)} 檔" if len(ann) > 8 else ""))
    if len(short_p):
        lines += ["", "⚡ <b>短線</b>・抱 5～10 天"]
        for i, r in enumerate(short_p.itertuples()):
            lines.append(f"{CIRCLE[i]} <b>{e(r.name)}</b> {price(r.close)}　停損 {price(r.stop)}・停利 {price(r.tp10)}")
    if len(long_p):
        lines += ["", "👀 <b>觀察名單</b>・其他規則加新聞挑選，不在模擬帳戶裡，當參考"]
        for r in long_p.itertuples():
            x = lv.get(r.stock_id)
            tail = f"　停損 {price(x['停損價'])}" if x is not None else ""
            lines.append(f"・{e(r.name)} {price(r.close)}{tail}")
    if len(v2) or len(short_p):
        lines += ["", "👇 理由、風險和要買多少在下面"]
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


def msg_v2(v2):
    """帳戶有買進委託才有：每檔的策略、理由和要買多少（停損 -15%、滿 60 天賣、不設停利）。"""
    if not len(v2):
        return None
    files = sorted((RESULTS.parent / "data" / "daily").glob("*.csv"))
    close = pd.read_csv(files[-1], dtype={"stock_id": str}).set_index("stock_id")["close"] if files else pd.Series(dtype=float)
    nav = paper_nav()
    nm = news_map(v2["date"].iat[0])
    out = ["🎯 <b>明天開盤的訊號・細節</b>\n<i>都是開盤買；開盤漲停或開盤就低於停損的不買</i>"]
    n = 0
    for title, rows in horizon_groups(v2):
        rule = ("不設停損，處置最後一天收盤賣" if "短期" in title else f"不設停損，抱 {REBOUND_HOLD} 個交易日收盤賣" if "中期" in title
                else "收盤跌破停損隔天賣，不設停利；滿 60 天還在 60 日線上續抱，最多 120 天")
        out.append(f"<b>{title}</b>\n<i>{rule}</i>")
        for r in rows:
            c = close.get(r.stock_id, float("nan"))
            xd = getattr(r, "exit_date", None)
            head = (f"收盤 {price(c)}　⏱ 抱 {REBOUND_HOLD} 個交易日收盤賣" if r.strategy == REBOUND
                    else f"收盤 {price(c)}　⏱ {md(xd)} 收盤賣" if isinstance(xd, str) and xd
                    else f"收盤 {price(c)}　🛑 停損 <b>{price(r.stop)}</b>" + (f" {pct(r.stop, c)}" if c == c else ""))
            w = r.shares * c / nav if c == c and nav == nav and nav > 0 else float("nan")
            body = [head, f"💰 資金的 {w:.1%}（每 100 萬約 {w * 100:.1f} 萬）" if w == w else ""]
            ev = [f"✅ {e(short(r.reason))}", chips_line(r.stock_id), news_line(nm, r.stock_id)]
            out.append(f"{CIRCLE[n % len(CIRCLE)]} <b>{e(r.name)}</b> {r.stock_id}　<i>{e(r.strategy)}</i>\n<blockquote>" + "\n".join(x for x in body if x)
                       + "</blockquote>\n" + "\n".join(x for x in ev if x))
            n += 1
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
    lines = ["🤖 <b>模擬帳戶 v6</b>・100 萬起，完全照上面的訊號買賣，不經人手"]
    body = [f"淨值 <b>{last['淨值'] / 1e4:,.1f} 萬</b>（起始 {first['淨值'] / 1e4:,.0f} 萬，{last['淨值'] / first['淨值'] - 1:+.1%}）",
            f"持股 {int(last['持股數'])} 檔・現金 {last['現金'] / 1e4:,.1f} 萬"]
    try:
        tx = load_market_index()["taiex"]
        t0, t1 = tx.asof(pd.Timestamp(first["date"])), tx.asof(pd.Timestamp(last["date"]))
        body.append(f"同期加權指數 {t1 / t0 - 1:+.1%}")
    except Exception:
        pass
    lines.append("<blockquote>" + "\n".join(body) + "</blockquote>")
    lines += paper_holdings(d, last["date"])
    ck = d / "check.txt"
    if ck.exists():
        ver, level, why = (ck.read_text(encoding="utf-8").split("\t") + ["", "", ""])[:3]
        if level == "暫停":
            lines.append(f"⛔ <b>規則 {e(ver)} 暫停新買進</b>：{e(why)}。我會重測規則後再恢復。")
        elif level == "注意":
            lines.append(f"🟡 {e(why)}，還在回測預期內，照規則繼續。")
    o = d / "orders.csv"
    orders = pd.read_csv(o, dtype={"stock_id": str}) if o.exists() and o.stat().st_size > 0 else pd.DataFrame()
    files = sorted((RESULTS.parent / "data" / "daily").glob("*.csv"))
    pclose = pd.read_csv(files[-1], dtype={"stock_id": str}).set_index("stock_id")["close"] if files else pd.Series(dtype=float)
    if len(orders):
        lines.append("🛒 <b>明天出手</b>")
        for r in orders.itertuples():
            if r.side == "賣":
                lines.append(f"・🔴 賣 <b>{e(r.name)}</b> {int(r.shares):,} 股（{e(r.reason)}）")
            elif r.strategy == REBOUND:
                lines.append(f"・🟢 買 <b>{e(r.name)}</b> {int(r.shares):,} 股（處置跌深，不設停損，抱 {REBOUND_HOLD} 個交易日收盤賣；開盤漲停就不買）")
            elif isinstance(getattr(r, "exit_date", None), str) and r.exit_date:
                lines.append(f"・🟢 買 <b>{e(r.name)}</b> {int(r.shares):,} 股（處置股，不設停損，{md(r.exit_date)} 收盤賣；開盤漲停就不買）")
            else:
                lines.append(f"・🟢 買 <b>{e(r.name)}</b> {int(r.shares):,} 股，停損 {price(r.stop)}（開盤低於停損就不買）")
            if r.side == "買":
                c = pclose.get(r.stock_id, float("nan"))
                if c == c:
                    lines.append(f"　　開盤市價買，不挑價；今天收盤 {price(c)}，約 {r.shares * c / 1e4:.1f} 萬；開在漲停 {price(c * 1.1)} 附近就放棄")
    pos_p = d / "positions.csv"
    pos = pd.read_csv(pos_p, dtype={"stock_id": str}) if pos_p.exists() and pos_p.stat().st_size > 0 else pd.DataFrame()
    due = []
    if len(pos) and "exit_date" in pos.columns:
        nxt = next_trading_day(pd.Timestamp(last["date"]))
        due = [r for r in pos.itertuples() if (r.strategy == REBOUND and int(r.days) + 1 >= REBOUND_HOLD)
               or (r.strategy != REBOUND and isinstance(r.exit_date, str) and r.exit_date and pd.Timestamp(r.exit_date) <= nxt)]
    if due:
        lines.append(f"明天收盤前賣（處置最後一天，或處置跌深滿 {REBOUND_HOLD} 天）：")
        for r in due:
            lines.append(f"・🔴 賣 <b>{e(r.name)}</b> {int(r.shares):,} 股")
    if not len(orders) and not due:
        lines.append("🛒 <b>明天不出手</b>，沒有買賣。")
    return "\n".join(lines)


def announced_today(day):
    """今天收盤後公告處置的普通股名稱（v5 第 2 天才買，先預告）。"""
    try:
        f = RESULTS.parent / "data" / "disposition.csv"
        d = pd.read_csv(f, dtype={"stock_id": str}, parse_dates=["announce"])
        d = d[(d["announce"] == pd.Timestamp(day).normalize()) & d["stock_id"].str.fullmatch(r"\d{4}")]
        return list(dict.fromkeys(d["name"].astype(str)))
    except Exception:
        return []


def md(s):
    t = pd.Timestamp(s)
    return f"{t.month}/{t.day}"


def paper_holdings(d, day):
    """模擬帳戶持股明細（最多 10 檔，長度固定）和今天出場的交易。"""
    out = []
    pos_p = d / "positions.csv"
    pos = pd.read_csv(pos_p, dtype={"stock_id": str}) if pos_p.exists() and pos_p.stat().st_size > 0 else pd.DataFrame()
    if len(pos):
        files = sorted((RESULTS.parent / "data" / "daily").glob("*.csv"))
        close = pd.read_csv(files[-1], dtype={"stock_id": str}).set_index("stock_id")["close"] if files else pd.Series(dtype=float)
        rows = []
        rv_p = d / "hold_review.csv"   # 每檔續抱或賣出的依據（portfolio.plan 寫的）
        rv = pd.read_csv(rv_p, dtype={"stock_id": str}) if rv_p.exists() and rv_p.stat().st_size > 0 else pd.DataFrame()
        basis = dict(zip(rv["stock_id"], rv["依據"].astype(str))) if len(rv) else {}
        for r in pos.itertuples():
            c = close.get(r.stock_id, float("nan"))
            ret = c / r.entry - 1 if c == c else float("nan")
            gap = c / r.stop - 1 if c == c else float("nan")
            flag = "⚠️" if gap == gap and gap < 0.05 else ("🟢" if ret == ret and ret > 0 else "🔻")
            xd = getattr(r, "exit_date", None)
            if r.strategy == REBOUND:        # 處置跌深：不設停損，抱滿 REBOUND_HOLD 個交易日收盤賣
                flag = "🟢" if ret == ret and ret > 0 else "🔻"
                rows.append(f"{flag} <b>{e(r.name)}</b> {price(r.entry)}→{price(c)} <b>{ret:+.1%}</b>・處置跌深，第 {int(r.days)}/{REBOUND_HOLD} 天，滿 {REBOUND_HOLD} 天收盤賣")
            elif isinstance(xd, str) and xd:   # 處置股：不設停損，處置最後一天收盤賣
                flag = "🟢" if ret == ret and ret > 0 else "🔻"
                rows.append(f"{flag} <b>{e(r.name)}</b> {price(r.entry)}→{price(c)} <b>{ret:+.1%}</b>・處置股，{md(xd)} 收盤賣")
            else:
                why = basis.get(r.stock_id, "")
                tail = f"第 {int(r.days)} 天，{e(why.split('：', 1)[1])}" if int(r.days) >= 60 and "續抱：" in why else f"第 {int(r.days)}/60 天"
                rows.append(f"{flag} <b>{e(r.name)}</b> {price(r.entry)}→{price(c)} <b>{ret:+.1%}</b>・停損 {price(r.stop)}・{tail}")
        out.append("📍 <b>持股</b>（⚠️＝離停損不到 5%）")
        out.append("<blockquote>" + "\n".join(rows) + "</blockquote>")
        nm = news_map(day)
        hn = [f"・<b>{e(r.name)}</b>：{news_line(nm, r.stock_id)[2:]}" for r in pos.itertuples() if r.stock_id in nm]
        if hn:
            out += ["📰 <b>持股消息</b>"] + hn
    out += paper_watch(d)
    tr_p = d / "trades.csv"
    if tr_p.exists():
        t = pd.read_csv(tr_p, dtype={"stock_id": str})
        t = t[t["賣出日"].astype(str) == str(day)]
        for r in t.itertuples():
            out.append(f"✔️ 今天賣出 <b>{e(r.name)}</b> {r.報酬:+.1%}（{e(r.原因)}）")
    return out


NEAR_STOP, NEAR_MA, NEAR_DAYS, BIG_GAIN = 0.05, 0.03, 5, 0.30


def paper_watch(d):
    """明天要留意的持股：離停損近、快滿 60／120 天、續抱中快碰 60 日線、已經大賺。只提醒，不改規則。
    資料來自 portfolio.plan 每晚寫的 hold_review.csv。"""
    p = d / "hold_review.csv"
    if not p.exists() or p.stat().st_size == 0:
        return []
    rv = pd.read_csv(p, dtype={"stock_id": str})
    if rv.empty or "離停損" not in rv.columns:
        return []
    rows = []
    for r in rv[rv["決定"] == "續抱"].itertuples():
        if r.strategy in ("處置股", REBOUND):
            continue
        days, gs, gm, ret = int(r.持有天數), r.離停損, r.離60日線, r.目前報酬
        ma = f"目前比 60 日線（約 {price(r._13)}）{'高' if gm >= 0 else '低'} {abs(gm):.1%}" if gm == gm else "沒有 60 日線資料"
        notes = []
        if gs == gs and gs < NEAR_STOP:
            notes.append(f"離停損只剩 {gs:.1%}，明天收盤低於 {price(r.停損價)} 就會在後天開盤賣")
        if 60 - NEAR_DAYS <= days < 60:
            notes.append(f"再 {60 - days} 天滿 60 天，{ma}；到時在線上續抱、在線下賣")
        elif days >= 60:
            if days >= 120 - NEAR_DAYS:
                notes.append(f"續抱第 {days} 天，再 {120 - days} 天滿 120 天就賣")
            if gm == gm and gm < NEAR_MA:
                notes.append(f"續抱中，{ma}，收盤跌破就賣")
        if ret == ret and ret >= BIG_GAIN:
            seen = any("60 日線" in x for x in notes)
            notes.append(f"已賺 {ret:+.0%}，規則沒有停利價，出場看 60 日線" + ("" if seen else f"（{ma}）") + ("，未滿 60 天前只看停損" if days < 60 else ""))
        if notes:
            rows.append(f"・<b>{e(r.name)}</b>：" + "；".join(notes))
    if not rows:
        return []
    return ["👀 <b>明天留意</b>（只是提醒，帳戶照規則走）", "<blockquote>" + "\n".join(rows) + "</blockquote>"]


def msg_more(long_v, short_v, picks):
    """名詞說明，收在可展開的區塊裡。沒選的股票和規則訊號不推：收到的人本來就不知道它們，看了也沒用（看板上有）。"""
    lines = []
    lines.append("📖 <b>怎麼看</b>")
    lines.append("<blockquote expandable>"
                 "v6＝資金分三份：營收加速 30%、一年新高 30%、處置股 40%。營收加速（月營收年增 30% 以上且在加速、創 12 個月新高、股價已比大盤多漲 5%，每月一次）和一年新高（收盤創一年新高，天天選）：各最多 10 檔，停損 -15%，不設停利；滿 60 天時收盤還在 60 日線上就續抱，跌破 60 日線或滿 120 天才賣。"
                 "處置股（被公告處置的股票，處置第 2 天開盤買、最後一天收盤賣，約 5～10 天；處置第一天平均是跌的，所以等一天）：每筆只放總資金約 2%，不設停損，最多 20 檔；處置股是分盤撮合，下單可能要預收款券。"
                 "處置跌深（處置期間跌 15% 以上的，解除當天開盤買，抱 20 個交易日收盤賣）：每筆一樣約 2%、不設停損，和處置股共用 20 個位子。\n"
                 "觀察名單的目標＝同組歷史報酬中位數，樂觀＝前 25%，當參考，不要掛單賣（回測發現固定停利會砍掉大漲股）。\n"
                 "停損＝八成賺錢的歷史訊號沒跌破的價位，盤勢亂時依波動放寬；短線停損＝近 20 日平均波動 × 3。\n"
                 "買多少＝營收加速、一年新高每檔打到停損最多虧總資金約 0.33%、單檔最多約 6.7%（資金分三份各算）；處置股每筆約 2%；短線每檔最多虧 1%、單檔 20%。依自己的資金等比例換算。\n"
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
    v2 = load_v2_buys()
    msgs = [m for m in (msg_overview(day, v2, long_p, lv, short_p, picks), msg_v2(v2), msg_short(short_p, odds)) if m]
    msgs[-1] += "\n\n" + msg_more(long_v, short_v, picks)  # 說明接在訊號的最後一則後面，不另發
    paper = msg_paper()  # 模擬帳戶獨立一則，放最後
    if paper:
        msgs.append(paper)
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
    unknown = [x for x in sys.argv[1:] if x.startswith("--") and x not in ("--file", "--preview", "--dry")]
    if unknown:  # 打錯參數（例如 --dry-run）不可以變成真的推播
        sys.exit(f"不認得的參數 {unknown}，可用：--dry（只印不送）、--preview、--file 路徑")
    if "--file" in sys.argv:  # 早上美股消息提醒：推一個寫好的 HTML 文字檔（results/morning/日期.txt）
        path = sys.argv[sys.argv.index("--file") + 1]
        msgs = [open(path, encoding="utf-8").read().strip()[:4000]]
    else:
        msgs = build_messages()
    if "--preview" in sys.argv:
        msgs[0] = "🧪 <i>排版預覽（內容是上週五的推薦）</i>\n\n" + msgs[0]
    if "--dry" in sys.argv:
        print("\n\n========== 下一則 ==========\n\n".join(msgs))
        return
    cids = chat_ids()
    for cid in cids:
        for m in msgs:
            _api("sendMessage", chat_id=cid, text=m, parse_mode="HTML", disable_web_page_preview="true")
    print(f"已推播到 Telegram（{len(msgs)} 則 × {len(cids)} 人）")


if __name__ == "__main__":
    main()
