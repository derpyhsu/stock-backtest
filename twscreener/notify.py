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


def agent_section():
    """有今天的 agent 推薦就放在最前面（results/agent_picks_log.csv）。"""
    path = RESULTS / "agent_picks_log.csv"
    if not path.exists():
        return []
    log = pd.read_csv(path, dtype={"stock_id": str})
    if log.empty:
        return []
    day = log["pick_date"].max()
    picks = log[(log["pick_date"] == day) & (log["verdict"] == "推薦")]
    if picks.empty:
        return []
    lines = [f"🤖 <b>今日推薦</b>（{day}，數據＋新聞綜合判斷）"]
    for i, r in enumerate(picks.itertuples(), 1):
        lines.append(f"{i}. <b>{html.escape(r.name)} {r.stock_id}</b>［{html.escape(str(r.industry))}］收 {r.close:g}")
        lines.append(f"   依據：{html.escape(str(r.reason))}")
        lines.append(f"   風險：{html.escape(str(r.risk))}")
    vetoed = log[(log["pick_date"] == day) & (log["verdict"] == "否決")]
    if len(vetoed):
        lines.append(f"否決 {len(vetoed)} 檔（詳見看板）：" + "、".join(vetoed["name"].head(8)))
    lines.append("")
    return lines


def build_message():
    path = RESULTS / "picks_today.csv"
    picks = pd.read_csv(path, dtype={"stock_id": str}) if path.exists() else pd.DataFrame()
    if picks.empty:
        return "📊 <b>台股選股看板</b>\n今天沒有股票符合條件。\n\n" + f'<a href="{DASHBOARD_URL}">打開看板</a>'
    day = picks["date"].iat[0]
    lines = agent_section() + [f"📊 <b>各策略訊號 {day}</b>（共 {len(picks)} 檔次）"]
    for strategy, g in picks.groupby("strategy", sort=False):
        label = strategy.split("_", 1)[-1]
        win = g["win60"].dropna()
        head = f"\n<b>{html.escape(label)}</b>"
        if len(win):
            head += f"　歷史 60 日勝率 {win.mean():.0%}"
        lines.append(head)
        for _, r in g.iterrows():
            ind = r.get("industry")
            ind = f"［{html.escape(str(ind))}］" if isinstance(ind, str) and ind else ""
            lines.append(f"• {html.escape(str(r['name']))} {r['stock_id']}{ind}　收 {r['close']:g}")
    lines.append("\n勝率是過去的統計，不保證未來。")
    lines.append(f'<a href="{DASHBOARD_URL}">打開看板看走勢圖和說明</a>')
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
