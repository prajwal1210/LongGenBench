#!/bin/bash

# ——————————————————————————————————————————
# You can override these from the environment if you like:
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
# ——————————————————————————————————————————

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <model_path>"
  exit 1
fi

MODEL_TYPE=$1
SCRIPT="./run.sh"   # ← replace with the path to your existing script

# the arrays to loop over
BACKENDS=(flash thresh)
WARMUPS=(64 128 256 512)
PERCENTILES=(0.5 0.75 0.875)

# Run the baselines
echo "→ $MODEL_TYPE | backend=flash | warmup=0 | pct=0"
sbatch $SCRIPT "$MODEL_TYPE" "flash" 0 0

for WARMUP in "${WARMUPS[@]}"; do
  for PERCENTILE in "${PERCENTILES[@]}"; do
    echo "→ $MODEL_TYPE | backend=thresh | warmup=$WARMUP | pct=$PERCENTILE"
    sbatch "$SCRIPT" \
      "$MODEL_TYPE" \
      "thresh" \
      "$WARMUP" \
      "$PERCENTILE"
  done
done
