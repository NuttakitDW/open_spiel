#!/bin/bash
set -euo pipefail

# lockf holds an OS lock across this entire run and releases it on crash.
if [[ ${1:-} != --locked ]]; then
  if [[ $# -ne 5 ]]; then
    echo "usage: run_hu_limit_bot.sh ABS_BINARY ABS_RUN_DIR DEADLINE_EPOCH TRAIN_SECONDS EVAL_HANDS" >&2
    exit 64
  fi
  mkdir -p "$2"
  exec /usr/bin/lockf -k -t 0 "$2/.runner.lock" /bin/bash "$0" --locked "$@"
fi
shift

if [[ $# -ne 5 ]]; then
  echo "usage: run_hu_limit_bot.sh ABS_BINARY ABS_RUN_DIR DEADLINE_EPOCH TRAIN_SECONDS EVAL_HANDS" >&2
  exit 64
fi

binary=$1
run_dir=$2
deadline=$3
train_seconds=$4
eval_hands=$5
mkdir -p "$run_dir" || exit 1

run_status="$run_dir/run.status"
checkpoint="$run_dir/current.chk"
trainer_status="$run_dir/trainer.status"
evaluation="$run_dir/evaluation.txt"
status() {
  phase=$1
  detail=$2
  {
    printf 'timestamp_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'phase=%s\n' "$phase"
    printf 'detail=%s\n' "$detail"
    printf 'checkpoint=%s\n' "$checkpoint"
    printf 'evaluation=%s\n' "$evaluation"
  } > "$run_status.tmp"
  mv "$run_status.tmp" "$run_status"
}

status training starting
train_rc=1
for attempt in 1 2 3; do
  if [[ $(date +%s) -ge $deadline ]]; then
    train_rc=2
    status incomplete deadline_before_training
    break
  fi
  status training "attempt_$attempt"
  if "$binary" train "--checkpoint=$checkpoint" "--status=$trainer_status" \
      "--seconds=$train_seconds" "--deadline_epoch=$deadline" \
      --checkpoint_interval=600 --hands=10 --final_eval=0; then
    train_rc=0
    break
  else
    train_rc=$?
  fi
  if [[ $train_rc -eq 2 ]]; then
    status incomplete training_deadline_or_resource_limit
    break
  fi
  if [[ $attempt -lt 3 ]]; then sleep 5; fi
done

if [[ ! -f "$checkpoint" ]]; then
  if [[ $train_rc -eq 2 ]]; then
    status incomplete no_checkpoint_before_deadline
    exit 2
  fi
  status failed "training_exit_$train_rc no_checkpoint"
  exit 1
fi

status evaluating "training_exit_$train_rc"
eval_budget=$((deadline + 900 - $(date +%s)))
if [[ $eval_budget -le 0 ]]; then
  status incomplete evaluation_deadline
  exit 2
fi
if /usr/bin/perl -e 'alarm(shift @ARGV); exec @ARGV' "$eval_budget" \
    "$binary" eval "--checkpoint=$checkpoint" "--hands=$eval_hands" \
    > "$evaluation.tmp"; then
  mv "$evaluation.tmp" "$evaluation"
else
  eval_rc=$?
  status failed "evaluation_exit_$eval_rc"
  exit "$eval_rc"
fi

if [[ $train_rc -eq 0 ]]; then
  status complete "trained_${train_seconds}_seconds_and_evaluated"
  exit 0
fi
status incomplete "training_exit_$train_rc checkpoint_evaluated"
exit "$train_rc"
