#!/usr/bin/env bash
# Sequential runs avoid oversubscribing Minecraft instances. Start with one pilot first.
set -euo pipefail
ROOT="${RUN_ROOT:-runs}"
CONFIG="${CONFIG:-configs/default.json}"
STEPS="${STEPS:-1000000}"
DEVICE="${DEVICE:-cpu}"
N_ENVS="${N_ENVS:-1}"
PORT="${PORT:-8000}"
for seed in 0 1 2; do
  for condition in A B C; do
    run="$ROOT/${condition}_seed${seed}"
    python -m combat_rl.train --config "$CONFIG" --condition "$condition" --seed "$seed" \
      --steps "$STEPS" --n-envs "$N_ENVS" --device "$DEVICE" --port "$PORT" --run-dir "$run"
    python -m combat_rl.evaluate --model "$run/final.zip" --split validation \
      --device "$DEVICE" --port "$PORT" --output "$run/validation.csv"
  done
done
# Run the test split only after choosing hyperparameters on validation scenarios.
