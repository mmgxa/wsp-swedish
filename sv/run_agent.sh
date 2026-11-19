#!/usr/bin/env bash
# Launch the editing agent (Kiro CLI) on a prepared run.
#   sv/run_agent.sh <run-id>            e.g. sv/run_agent.sh b2-test
# Environment: MODEL (default claude-opus-5.5), EFFORT (default high), TRUST (default "fs_read,fs_write,execute_bash"; "all" = -a).
set -euo pipefail
RUN="${1:?usage: sv/run_agent.sh <run-id>}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RD="$ROOT/sv/runs/$RUN"
[ -f "$RD/PROMPT.md" ] || { echo "no run $RUN (create it with: uv run --offline python sv/init_run.py --budget N)"; exit 1; }
MODEL="${MODEL:-claude-opus-5.5}"
EFFORT="${EFFORT:-high}"
TRUST="${TRUST:-fs_read,fs_write,execute_bash}"
if [ "$TRUST" = all ]; then TRUSTARG=(--trust-all-tools); else TRUSTARG=(--trust-tools="$TRUST"); fi

cd "$ROOT"
export WSP_RUN_DIR="$RD" HF_HUB_OFFLINE=1
echo "$RD" > sv/runs/CURRENT
MSG="Read $RD/PROMPT.md and carry out the task end to end. Follow every rule in it. Run all Python with 'uv run --offline python' from $ROOT, with WSP_RUN_DIR=$RD exported."
echo "run $RUN  model=$MODEL effort=$EFFORT trust=$TRUST  log=$RD/agent.log"
kiro-cli chat --no-interactive --model "$MODEL" --effort "$EFFORT" "${TRUSTARG[@]}" "$MSG" 2>&1 | tee "$RD/agent.log"
