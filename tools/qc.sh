#!/usr/bin/env bash
# Shared QC runner for Harbor tasks in this repository.
#
# Usage: bash tools/qc.sh <task-slug> [variant ...]
#
# For every directory under <task-slug>/solution/variants/ (or only the variants
# named on the command line), runs the task through Harbor with the oracle agent
# and SOLUTION=<variant>, then runs it once more with the nop agent. Each trial's
# reward.txt and details.json are copied to <task-slug>/qc/results/<name>.reward.txt
# and <name>.details.json, and a variant -> reward table is printed at the end.
# Harbor job directories go to <task-slug>/qc/jobs/ (ignored by git).
#
# Pass "nop" as a variant name to run only the nop agent. Harbor failures do not
# stop the loop; the table shows ERR for that row.
set -uo pipefail

TASK="${1:?usage: bash tools/qc.sh <task-slug> [variant ...]}"
shift || true

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_DIR="$ROOT/$TASK"
VARIANTS_DIR="$TASK_DIR/solution/variants"
JOBS_DIR="$TASK_DIR/qc/jobs"
RESULTS_DIR="$TASK_DIR/qc/results"

export PYTHONIOENCODING=utf-8
HARBOR=(uvx --from harbor harbor)

if [ ! -f "$TASK_DIR/task.toml" ]; then
    echo "qc.sh: $TASK_DIR has no task.toml" >&2
    exit 2
fi
mkdir -p "$JOBS_DIR" "$RESULTS_DIR"

# Variant list: command line, else every directory under solution/variants plus nop.
NAMES=()
if [ "$#" -gt 0 ]; then
    NAMES=("$@")
else
    if [ -d "$VARIANTS_DIR" ]; then
        for d in "$VARIANTS_DIR"/*/; do
            [ -d "$d" ] && NAMES+=("$(basename "$d")")
        done
    fi
    NAMES+=("nop")
fi

# newest_dir <parent>: the most recently modified direct subdirectory of <parent>.
newest_dir() {
    ls -1dt "$1"/*/ 2>/dev/null | head -n 1 | sed 's:/*$::'
}

# trial_dir <jobdir>: the single trial directory inside a Harbor job directory
# (<task>__<id>/), picked as the newest subdirectory that holds a result.json.
trial_dir() {
    local jobdir="$1" d
    for d in $(ls -1dt "$jobdir"/*/ 2>/dev/null); do
        d="${d%/}"
        if [ -f "$d/result.json" ]; then
            echo "$d"
            return 0
        fi
    done
    return 1
}

# reward_from_result <trial_dir>: fallback reward from result.json when the
# verifier left no reward.txt behind.
reward_from_result() {
    python - "$1/result.json" <<'PY'
import json, sys
try:
    r = json.load(open(sys.argv[1], encoding="utf-8"))
    v = (r.get("verifier_result") or {}).get("rewards") or {}
    print(v.get("reward", "ERR"))
except Exception:
    print("ERR")
PY
}

declare -A REWARD
declare -A NOTE

for name in "${NAMES[@]}"; do
    stamp="$(date +%Y%m%d-%H%M%S)"
    job="qc-${name}-${stamp}"
    if [ "$name" = "nop" ]; then
        agent_args=(-a nop)
    else
        if [ ! -f "$VARIANTS_DIR/$name/apply.sh" ]; then
            echo "qc.sh: no variant $VARIANTS_DIR/$name/apply.sh" >&2
            REWARD["$name"]="ERR"; NOTE["$name"]="variant missing"
            continue
        fi
        agent_args=(-a oracle --ae "SOLUTION=$name")
    fi

    echo "=== $TASK / $name  (job $job) ==="
    "${HARBOR[@]}" run -p "$TASK_DIR" "${agent_args[@]}" -y -q -n 1 \
        -o "$JOBS_DIR" --job-name "$job"
    status=$?
    [ "$status" -ne 0 ] && echo "qc.sh: harbor exited with $status for $name (continuing)" >&2

    jobdir="$JOBS_DIR/$job"
    [ -d "$jobdir" ] || jobdir="$(newest_dir "$JOBS_DIR")"
    trial=""
    [ -n "$jobdir" ] && trial="$(trial_dir "$jobdir" 2>/dev/null || true)"
    if [ -z "$trial" ]; then
        REWARD["$name"]="ERR"
        NOTE["$name"]="harbor produced no trial directory (exit $status); see $JOBS_DIR"
        continue
    fi

    reward_src="$(find "$trial" -name reward.txt -path '*verifier*' | head -n 1)"
    details_src="$(find "$trial" -name details.json -path '*verifier*' | head -n 1)"

    if [ -n "$reward_src" ]; then
        cp "$reward_src" "$RESULTS_DIR/$name.reward.txt"
    else
        reward_from_result "$trial" > "$RESULTS_DIR/$name.reward.txt"
    fi
    if [ -n "$details_src" ]; then
        cp "$details_src" "$RESULTS_DIR/$name.details.json"
    else
        printf '{"note": "verifier wrote no details.json", "trial": "%s"}\n' \
            "$(basename "$trial")" > "$RESULTS_DIR/$name.details.json"
    fi

    REWARD["$name"]="$(tr -d '[:space:]' < "$RESULTS_DIR/$name.reward.txt")"
    NOTE["$name"]="$(python - "$RESULTS_DIR/$name.details.json" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1], encoding="utf-8"))
    print(str(d.get("reason", "")).replace("\n", " ")[:90])
except Exception:
    print("")
PY
)"
done

echo
echo "QC results for $TASK  (files in $RESULTS_DIR)"
printf '%-20s %-8s %s\n' "variant" "reward" "reason"
printf '%-20s %-8s %s\n' "-------" "------" "------"
for name in "${NAMES[@]}"; do
    printf '%-20s %-8s %s\n' "$name" "${REWARD[$name]:-ERR}" "${NOTE[$name]:-}"
done
