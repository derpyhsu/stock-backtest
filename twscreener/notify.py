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
    """台股價位：100 元以上取整數，以下取一位小數。"""
    if pd.isna(v):
        return "-"
    return f"{v:,.0f}" if v >= 100 else f"{v:.1f}"


def short(text, n=2):
    """理由只留前幾段（用；分隔），手機上比較好讀。"""
    parts = [t for t in str(text).split("；") if t and t != "nan"]
    return "；".join(parts[:n])


def market_line():
    path = RESULTS / "candidates_today.csv"
    if not path.exists():
        return None
    c = pd.read_csv(path)
    regime = c["大盤狀態"].iat[0] if "大盤狀態" in c and len(c) else None
    try:
        from .review import market_context
        dd = market_context()["大盤離高點"].dropna().iat[-1]
        return f"大盤：<b>{e(regime)}</b>（距 120 日高點 {dd:+.0%}）"
    except Exception:
        return f"大盤：<b>{e(regime)}</b>" if regime else None


def agent_section():
    """今天的 agent 推薦（results/agent_picks_log.csv），附停損與目標價。"""
    path = RESULTS / "agent_picks_log.csv"
    if not path.exists():
        return []
    log = pd.read_csv(path, dtype={"stock_id": str})
    if log.empty:
        return []
    day = log["pick_date"].max()
    today = log[log["pick_date"] == day]
    today = today[today["signal_date"] == today["signal_date"].max()]  # 同一天重選時用最新收盤那一批
    picks = today[today["verdict"] == "推薦"]
    if picks.empty:
        return []
    lv = {}
    lpath = RESULTS / "agent_levels_today.csv"
    if lpath.exists():
        lt = pd.read_csv(lpath, dtype={"stock_id": str})
        lv = {r["stock_id"]: r for _, r in lt[lt["pick_date"] == day].iterrows()}
    lines = [SEP, f"🤖 <b>今日推薦 {len(picks)} 檔</b>"]
    for i, r in enumerate(picks.itertuples()):
        lines += ["", f"{CIRCLE[i]} <b>{e(r.name)} {r.stock_id}</b>｜{e(r.industry)}", f"收盤 {price(r.close)}"]
        x = lv.get(r.stock_id)
        if x is not None:
            stop = x["停損價"]
            lines.append(f"🎯 60日目標 <b>{price(x['60日目標價'])}</b>（樂觀 {price(x['60日停利價'])}）")
            how = "，依近期波動放寬" if str(x.get("停損依據", "")) == "依近期波動放寬" else ""
            lines.append(f"🛑 停損 <b>{price(stop)}</b>（{stop / r.close - 1:+.0%}{how}）")
            lines.append("📅 " + "｜".join(f"{h}日 {price(x[f'{h}日目標價'])}" for h in (30, 90, 120)))
        lines.append(f"✅ {e(short(r.reason))}")
        lines.append(f"⚠️ {e(short(r.risk, 1))}")
    vetoed = today[today["verdict"] == "否決"]
    if len(vetoed):
        lines += ["", f"❌ <b>否決 {len(vetoed)} 檔</b>"]
        for reason, g in vetoed.groupby("reason", sort=False):
            lines.append(f"・{e('、'.join(g['name']))}：{e(short(reason, 1))}")
    if lv:
        lines += ["", "<i>目標＝同組歷史報酬中位數，樂觀＝前 25%；停損＝八成賺錢的歷史訊號都沒跌破的價位，盤勢亂時依波動放寬；開盤就跳空跌破停損的，當天不要進場</i>"]
    return lines


def shortterm_section():
    """今天的 agent 短線推薦（results/shortterm_picks_log.csv），持有 5～10 天。"""
    path = RESULTS / "shortterm_picks_log.csv"
    if not path.exists():
        return []
    log = pd.read_csv(path, dtype={"stock_id": str})
    if log.empty:
        return []
    today = log[log["pick_date"] == log["pick_date"].max()]
    picks = today[today["verdict"] == "推薦"]
    if picks.empty:
        return []
    lines = [SEP, f"⚡ <b>短線 {len(picks)} 檔</b>（持有 5～10 天）"]
    for i, r in enumerate(picks.itertuples()):
        lines += ["", f"{CIRCLE[i]} <b>{e(r.name)} {r.stock_id}</b>｜{e(r.industry)}｜{e(r.rule)}", f"收盤 {price(r.close)}"]
        lines.append(f"🎯 停利 5日 <b>{price(r.tp5)}</b>｜10日 <b>{price(r.tp10)}</b>")
        lines.append(f"🛑 停損 <b>{price(r.stop)}</b>（{r.stop / r.close - 1:+.0%}）")
        lines.append(f"📊 歷史 10 日勝率 {r.hist_win10:.0%}，平均 {r.hist_mean10:+.1%}")
        lines.append(f"✅ {e(short(r.reason))}")
        lines.append(f"⚠️ {e(short(r.risk, 1))}")
    lines += ["", "<i>停利＝同規則歷史 5/10 日報酬前 25%（依個股波動換算）；停損＝近 20 日平均波動 × 3；第 10 天收盤沒到停利就出場</i>"]
    return lines


def signals_section(picks):
    lines = [SEP, f"📊 <b>各策略訊號</b>（{len(picks)} 檔次）"]
    for strategy, g in picks.groupby("strategy", sort=False):
        label = strategy.split("_", 1)[-1]
        win = g["win60"].dropna()
        tag = f"（歷史勝率 {win.mean():.0%}）" if len(win) else ""
        names = "、".join(g["name"].astype(str))
        lines.append(f"・<b>{e(label)}</b>{tag}：{e(names)}")
    return lines


def build_message():
    path = RESULTS / "picks_today.csv"
    picks = pd.read_csv(path, dtype={"stock_id": str}) if path.exists() else pd.DataFrame()
    day = picks["date"].iat[0] if len(picks) else ""
    lines = [f"📈 <b>台股選股日報</b>　{day} 收盤，適用隔天開盤"]
    m = market_line()
    if m:
        lines.append(m)
    lines += agent_section()
    lines += shortterm_section()
    if picks.empty:
        lines += [SEP, "今天沒有股票符合條件。"]
    else:
        lines += signals_section(picks)
    lines += [SEP, f'👉 <a href="{DASHBOARD_URL}">打開看板</a>：走勢圖、完整理由與新聞連結',
              "<i>歷史統計加新聞判斷，不保證未來，請自行評估部位。</i>"]
    return "\n".join(lines)


def main():
    msg = build_message()
    if "--dry" in sys.argv:
        print(msg)
        return
    _api("sendMessage", chat_id=chat_id(), text=msg, parse_mode="HTML", disable_web_page_preview="true")
    print("已推播到 Telegram")


if __name__ == "__main__":
    main()
