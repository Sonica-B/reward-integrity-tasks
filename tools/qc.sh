#!/usr/bin/env bash
# Shared QC runner for Harbor tasks in this repository.
#
# Usage: bash tools/qc.sh <task-slug> [variant-filter ...]
#
# Runs `harbor lint` on the task (findings are printed, the run continues),
# then, for every directory under <task-slug>/solution/variants/, runs the task
# through Harbor with the oracle agent and SOLUTION=<variant>, then runs it once
# more with the nop agent. Each trial's reward.txt and details.json are copied
# to <task-slug>/qc/results/<name>.reward.txt and <name>.details.json, and a
# variant -> reward table is printed at the end. Harbor job directories go to
# <task-slug>/qc/jobs/ (ignored by git).
#
# The optional variant filter restricts the run to the named variants. It may
# be given as several arguments or as one space-separated argument, so both
#   bash tools/qc.sh grader-hardening alt nop
#   bash tools/qc.sh grader-hardening "alt nop"
# re-run only alt and the nop agent. "nop" names the nop-agent row.
#
# A result is attributed to a variant only if the Harbor job directory was
# created for that iteration (the --job-name directory, or failing that a job
# directory newer than a marker touched when the iteration started) AND the
# trial's recorded agent config matches the row (agent oracle with
# SOLUTION=<name>, or agent nop for the nop row). Anything else is an ERR row:
# stale <name>.reward.txt / <name>.details.json are deleted and <name>.ERR.txt
# records why, so no old evidence survives under a name it no longer belongs
# to. Harbor failures do not stop the loop.
set -uo pipefail

TASK="${1:?usage: bash tools/qc.sh <task-slug> [variant-filter ...]}"
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

MARKER="$JOBS_DIR/.qc-iteration-marker"
trap 'rm -f "$MARKER"' EXIT

# Variant list: the filter (each argument split on whitespace), else every
# directory under solution/variants plus nop.
NAMES=()
if [ "$#" -gt 0 ]; then
    for arg in "$@"; do
        # shellcheck disable=SC2086  # the filter is a whitespace-separated list
        for n in $arg; do NAMES+=("$n"); done
    done
else
    if [ -d "$VARIANTS_DIR" ]; then
        for d in "$VARIANTS_DIR"/*/; do
            [ -d "$d" ] && NAMES+=("$(basename "$d")")
        done
    fi
    NAMES+=("nop")
fi
if [ "${#NAMES[@]}" -eq 0 ]; then
    echo "qc.sh: no variants to run" >&2
    exit 2
fi

echo "=== harbor lint $TASK ==="
"${HARBOR[@]}" lint "$TASK_DIR"
lint_status=$?
echo "qc.sh: harbor lint exited with $lint_status (continuing)"
echo

# iteration_job_dir <job-name>: the job directory this iteration produced.
# Prefers the --job-name directory; otherwise the newest job directory created
# after the marker was touched. Prints nothing when neither exists.
iteration_job_dir() {
    local job="$1" d
    if [ -d "$JOBS_DIR/$job" ]; then
        echo "$JOBS_DIR/$job"
        return 0
    fi
    while IFS= read -r d; do
        d="${d%/}"
        if [ -d "$d" ] && [ "$d" -nt "$MARKER" ]; then
            echo "$d"
            return 0
        fi
    done < <(ls -1dt "$JOBS_DIR"/*/ 2>/dev/null)
    return 1
}

# trial_dir <jobdir>: the single trial directory inside a Harbor job directory
# (<task>__<id>/), picked as the newest subdirectory that holds a result.json.
trial_dir() {
    local jobdir="$1" d
    while IFS= read -r d; do
        d="${d%/}"
        if [ -f "$d/result.json" ]; then
            echo "$d"
            return 0
        fi
    done < <(ls -1dt "$jobdir"/*/ 2>/dev/null)
    return 1
}

# trial_matches <trial_dir> <name>: succeeds when the trial's recorded agent
# config is the one this row launched (oracle + SOLUTION=<name>, or nop).
# Prints the mismatch on failure.
trial_matches() {
    python - "$1" "$2" <<'PY'
import json, os, sys
trial, name = sys.argv[1], sys.argv[2]
agent = None
for fn, path in (("config.json", ("agent",)), ("result.json", ("config", "agent"))):
    try:
        node = json.load(open(os.path.join(trial, fn), encoding="utf-8"))
        for key in path:
            node = node.get(key) or {}
        if node.get("name"):
            agent = node
            break
    except Exception:
        continue
if not agent:
    print("trial has no readable agent config")
    sys.exit(1)
got_name = agent.get("name")
got_sol = (agent.get("env") or {}).get("SOLUTION")
if name == "nop":
    ok = got_name == "nop"
    want = "agent nop"
else:
    ok = got_name == "oracle" and got_sol == name
    want = f"agent oracle with SOLUTION={name}"
if ok:
    sys.exit(0)
print(f"trial ran agent {got_name!r} with SOLUTION={got_sol!r}; this row wants {want}")
sys.exit(1)
PY
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

# note_from_details <details.json>: one line for the table. The verifier's
# "reason" when it writes one, else the list of failed gates when it writes
# "gates", else empty.
note_from_details() {
    python - "$1" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1], encoding="utf-8"))
    reason = str(d.get("reason", "")).replace("\n", " ").strip()
    if not reason and isinstance(d.get("gates"), dict):
        failed = [k for k, v in d["gates"].items() if not v]
        reason = ("gates failed: " + ", ".join(failed)) if failed else "all gates passed"
        if "stray_kills" in d:
            reason += f"; stray_kills={d['stray_kills']}"
    print(reason[:90])
except Exception:
    print("")
PY
}

# mark_err <name> <reason>: the row is ERR. Stale evidence under this name is
# removed and an explicit marker file records why.
mark_err() {
    local name="$1" reason="$2"
    rm -f "$RESULTS_DIR/$name.reward.txt" "$RESULTS_DIR/$name.details.json"
    printf 'ERR\n%s\nrecorded %s\n' "$reason" "$(date +%Y-%m-%dT%H:%M:%S)" \
        > "$RESULTS_DIR/$name.ERR.txt"
    REWARD["$name"]="ERR"
    NOTE["$name"]="$reason"
    echo "qc.sh: $name -> ERR: $reason" >&2
}

declare -A REWARD
declare -A NOTE
declare -A SECS

for name in "${NAMES[@]}"; do
    stamp="$(date +%Y%m%d-%H%M%S)"
    job="qc-${name}-${stamp}"
    if [ "$name" = "nop" ]; then
        agent_args=(-a nop)
    else
        if [ ! -f "$VARIANTS_DIR/$name/apply.sh" ]; then
            mark_err "$name" "variant missing: no $VARIANTS_DIR/$name/apply.sh"
            SECS["$name"]=0
            continue
        fi
        agent_args=(-a oracle --ae "SOLUTION=$name")
    fi

    echo "=== $TASK / $name  (job $job) ==="
    # The marker is touched before Harbor starts, so only a job directory
    # created from here on can be this iteration's. The pause keeps the
    # comparison safe on filesystems with coarse timestamps.
    touch "$MARKER"
    sleep 1
    t0=$SECONDS
    "${HARBOR[@]}" run -p "$TASK_DIR" "${agent_args[@]}" -y -q -n 1 \
        -o "$JOBS_DIR" --job-name "$job"
    status=$?
    SECS["$name"]=$((SECONDS - t0))
    [ "$status" -ne 0 ] && echo "qc.sh: harbor exited with $status for $name (continuing)" >&2

    jobdir="$(iteration_job_dir "$job" || true)"
    if [ -z "$jobdir" ]; then
        mark_err "$name" "harbor created no job directory for this run (exit $status); see $JOBS_DIR"
        continue
    fi
    trial="$(trial_dir "$jobdir" || true)"
    if [ -z "$trial" ]; then
        mark_err "$name" "no trial with result.json under $(basename "$jobdir") (exit $status)"
        continue
    fi
    mismatch="$(trial_matches "$trial" "$name")"
    if [ -n "$mismatch" ]; then
        mark_err "$name" "$mismatch (job $(basename "$jobdir"), trial $(basename "$trial"))"
        continue
    fi

    reward_src="$(find "$trial" -name reward.txt -path '*verifier*' | head -n 1)"
    details_src="$(find "$trial" -name details.json -path '*verifier*' | head -n 1)"

    rm -f "$RESULTS_DIR/$name.ERR.txt"
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
    NOTE["$name"]="$(note_from_details "$RESULTS_DIR/$name.details.json")"
    echo "qc.sh: $name -> ${REWARD[$name]} (${SECS[$name]} s; trial $(basename "$trial"))"
done

echo
echo "QC results for $TASK  (files in $RESULTS_DIR)"
printf '%-20s %-8s %-8s %s\n' "variant" "reward" "seconds" "note"
printf '%-20s %-8s %-8s %s\n' "-------" "------" "-------" "----"
for name in "${NAMES[@]}"; do
    printf '%-20s %-8s %-8s %s\n' "$name" "${REWARD[$name]:-ERR}" "${SECS[$name]:-0}" "${NOTE[$name]:-}"
done
