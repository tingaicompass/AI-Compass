#!/bin/zsh
set -euo pipefail

ROUTER_HOST="${AI_COMPASS_ROUTER_HOST:-ws-ozv555vq5mlbiepv.cn-beijing.maas.aliyuncs.com}"
ROUTER_TIMEOUT="${AI_COMPASS_ROUTER_TIMEOUT:-10}"
SYNC_SCRIPT="${AI_COMPASS_SYNC_SCRIPT:-/Users/ting/.codex/skills/ai-compass-weekly-highlight-sync/scripts/sync_weekly_highlight.py}"

if [[ ! -f "$SYNC_SCRIPT" ]]; then
  print -u2 "sync script not found: $SYNC_SCRIPT"
  exit 1
fi

if python3 - "$ROUTER_HOST" 2>/dev/null <<'PY'
import socket
import sys

socket.getaddrinfo(sys.argv[1], 443, type=socket.SOCK_STREAM)
PY
then
  router_mode="${AI_COMPASS_ROUTER_MODE:-hybrid}"
else
  router_mode="rules"
  print -u2 "[ai-compass] DNS preflight failed for ${ROUTER_HOST}; using deterministic rules routing"
fi

exec python3 "$SYNC_SCRIPT" \
  --router-mode "$router_mode" \
  --router-timeout "$ROUTER_TIMEOUT" \
  "$@"
