#!/bin/bash
# 把共用資料夾同步到 GitHub（derpyhsu/stock-backtest，公開 repo）。
# .env（FinMind token）和 __pycache__ 一律不上傳。
set -e
SRC=/mnt/project-files/stock-backtest
REPO=/home/user/stock-backtest
if ! git -C "$REPO" rev-parse HEAD >/dev/null 2>&1; then
  rm -rf "$REPO"
  git clone --depth 1 https://github.com/derpyhsu/stock-backtest "$REPO"
fi
cd "$REPO"
tar -C "$SRC" --exclude=.env --exclude=__pycache__ --exclude='*.pyc' -cf - . | tar -xf -
chown -R "$(id -u):$(id -g)" .
git add -A
if git ls-files | grep -qE '(^|/)\.env$'; then echo ".env 不可以上傳" >&2; exit 1; fi
if git diff --cached --quiet; then echo "沒有變更"; exit 0; fi
git -c user.name="derpyhsu" -c user.email="derpyhsu@users.noreply.github.com" commit -q -m "每日更新 $(date +%F)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01DWj1XdHt74qkxRVdp3mjBD"
git -c push.negotiate=true push -u origin main
