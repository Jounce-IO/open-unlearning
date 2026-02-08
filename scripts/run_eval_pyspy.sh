#!/usr/bin/env bash
# Run trajectory evaluation under py-spy with ~100 samples for benchmarking.
#
# Usage:
#   ./scripts/run_eval_pyspy.sh [EVAL_COMMAND...]
#
# If no command is given, prints the recommended py-spy + eval command.
# Example with dllm eval:
#   ./scripts/run_eval_pyspy.sh python -m eval experiment=eval/tofu model=... eval.tofu_trajectory.samples=100
#
# Output: py-spy raw file under /tmp/pyspy_eval_*.raw and wall-clock time.

set -e

SAMPLES="${EVAL_SAMPLES:-100}"
PYSPY_OUT="${PYSPY_OUT:-/tmp/pyspy_eval_${SAMPLES}samples.raw}"

if [ $# -eq 0 ]; then
  echo "Usage: $0 <eval_command_with_samples>"
  echo "Example: $0 python -m eval experiment=eval/tofu model=your_model eval.tofu_trajectory.samples=${SAMPLES}"
  echo ""
  echo "Recommended py-spy command (run from repo root or open-unlearning):"
  echo "  uvx py-spy record -f raw -o ${PYSPY_OUT} --subprocesses --full-filenames -r 10 -- \\"
  echo "    python -m eval experiment=eval/tofu model=... eval.tofu_trajectory.samples=${SAMPLES}"
  exit 0
fi

echo "Running eval under py-spy (samples=${SAMPLES}, output=${PYSPY_OUT})..."
echo "Command: uvx py-spy record -f raw -o ${PYSPY_OUT} --subprocesses --full-filenames -r 10 -- $*"
START=$(date +%s)
uvx py-spy record -f raw -o "${PYSPY_OUT}" --subprocesses --full-filenames -r 10 -- "$@"
END=$(date +%s)
echo "Wall-clock time: $((END - START))s"
echo "Py-spy output: ${PYSPY_OUT}"
