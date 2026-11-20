---
name: commit
description: 這個 repo 的 commit 規範（Conventional Commits 標題、說明為什麼的內文、AI 協作 trailer）。Use before every git commit in this repo, or when the user asks how to write a commit message.
---

# commit

## 標題

`type(scope): summary`，一行，英文小寫開頭、祈使句、不加句點，72 字元內。

- `type`：`feat` 新功能、`fix` 修錯、`refactor` 不改行為的整理、`perf`、`docs`、`test`、`chore`（設定、依賴）、`data`（只有資料或看板更新）。
- `scope`：動到的部分，例如 `twse`、`fundamentals`、`market`、`strategies`、`dashboard`、`skill`。跨很多部分就省略 scope。
- 一個 commit 做一件事。好幾件不相關的事就拆開。

例：`fix(twse): refetch recent days fetched before margin data was published`

## 內文

空一行後寫，每行約 72 字元換行。寫「為什麼」和「看到什麼」，不是重述 diff：

- 碰到的問題或觀察（附數字更好）。
- 這次怎麼處理，還有刻意沒做的事。
- 怎麼驗證的；沒驗證到的也要寫。

純資料更新可以只有標題，例如 `data: daily update 2026-11-20`。

## Trailer

AI 參與的 commit 在最後加上：

```
Co-Authored-By: <模型名稱> <noreply@anthropic.com>
```

用其他工具（Codex、Cursor）時，改成該工具的署名。

## 不要

- 不要 commit `.env`、token、URL 裡的 session 參數。
- 不要 commit `__pycache__/`、暫存檔、heap snapshot。
- 不要改寫已經推上去的歷史（不 force push `main`）。
