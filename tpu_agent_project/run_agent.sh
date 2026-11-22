#!/usr/bin/env bash
# Headless Claude Code session in this workspace with the TPU agent suite
# installed: `tpu-agent` routes the prompt to a specialist (tpu-knowledge-base,
# tpu-diagnostics, ...). Usage: ./run_agent.sh prompts/knowledge_base.md
# The suite's agents run `python3 ~/.tpu-agent-suite/.../runtime.py ...`, so
# the allowlist covers python3 plus file tools; everything else is denied.
set -euo pipefail
cd "$(dirname "$0")"
PROMPT_FILE=${1:?prompt file}
NAME=$(basename "${PROMPT_FILE%.md}")
mkdir -p results
source .venv/bin/activate            # hooks/runtime call `python3`: must be 3.12
source ../scripts/env.sh 2>/dev/null || true
env -u CLAUDECODE -u CLAUDE_CODE_SESSION_ID -u CLAUDE_CODE_ENTRYPOINT \
  claude -p "$(cat "$PROMPT_FILE")" \
    --allowedTools "Bash(python3 *)" "Read" "Write" "Edit" "Glob" "Grep" "Agent" "Skill" \
    --max-turns 60 --output-format json > "results/agent_${NAME}.json" 2> "results/agent_${NAME}.err"
python3 - "results/agent_${NAME}.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print({k: d.get(k) for k in ("is_error", "num_turns", "duration_ms", "total_cost_usd")})
print(d.get("result", "")[:3000])
PY
