#!/usr/bin/env bash
set -euo pipefail
CONFIG=$1
GPUS=$2
shift 2
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec torchrun --standalone --nproc_per_node="$GPUS" "$ROOT/tools/train.py" "$CONFIG" --launcher pytorch "$@"
