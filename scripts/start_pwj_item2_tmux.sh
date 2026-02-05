#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SESSION="${1:-pwj-pilot}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  SESSION="${SESSION}-$(date -u +%Y%m%d-%H%M%S)"
fi

RUN_CMD="python scripts/pwj_pipeline.py --items 2 --enable-worker-network --reset-state"

tmux new-session -d -s "$SESSION" -c "$REPO_ROOT" -n run "bash -lc '$RUN_CMD; ec=\$?; echo; echo \"[${SESSION}] pipeline exited with code \$ec\"; exec zsh'"

tmux new-window -t "$SESSION" -c "$REPO_ROOT" -n logs "bash -lc 'cd \"$REPO_ROOT\" && echo \"Waiting for worker logs...\" && while true; do f=\$(ls -1td outputs/pwj_pipeline/item_2/attempt_*/worker_logs/stdout.jsonl 2>/dev/null | head -n1 || true); if [ -n \"\$f\" ]; then echo \"Tailing \$f\"; tail -n 30 -f \"\$f\"; fi; sleep 2; done'"

tmux new-window -t "$SESSION" -c "$REPO_ROOT" -n inspect "bash -lc 'cd \"$REPO_ROOT\" && while true; do clear; echo \"Newest attempt dirs:\"; ls -1td outputs/pwj_pipeline/item_2/attempt_* 2>/dev/null | head -n 3 || true; echo; echo \"Attempt contents (newest):\"; d=\$(ls -1td outputs/pwj_pipeline/item_2/attempt_* 2>/dev/null | head -n1 || true); if [ -n \"\$d\" ]; then find \"\$d\" -maxdepth 2 -type f | sed \"s|^| - |\"; else echo \"(none yet)\"; fi; echo; echo \"State:\"; cat outputs/pwj_pipeline/state.json 2>/dev/null || echo \"(state not written yet)\"; sleep 2; done'"

tmux select-window -t "$SESSION":run

echo "tmux session created: $SESSION"
echo "Attach with: tmux attach -t $SESSION"
