#!/usr/bin/env bash
set -euo pipefail
CONFIG=$1
CHECKPOINT=$2
GPUS=$3
shift 3
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec torchrun --standalone --nproc_per_node="$GPUS" "$ROOT/tools/test.py" "$CONFIG" "$CHECKPOINT" --launcher pytorch "$@"
