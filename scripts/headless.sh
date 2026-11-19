#!/usr/bin/env bash
# Software OpenGL fallback for the first Linux smoke/benchmark run.
# Use a configured hardware display/VirtualGL directly for faster production runs.
set -euo pipefail
if [[ $# -eq 0 ]]; then
  echo "Usage: bash scripts/headless.sh python -m combat_rl.smoke" >&2
  exit 2
fi
exec xvfb-run -a -s "-screen 0 640x480x24" env LIBGL_ALWAYS_SOFTWARE=1 "$@"
