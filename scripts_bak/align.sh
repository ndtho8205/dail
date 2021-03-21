#!/usr/bin/env bash

set -o errexit
set -o errtrace
set -o nounset
set -o pipefail

trap _cleanup SIGINT SIGTERM ERR EXIT

HERE="$(
  cd "$(dirname "$0")/.."
  pwd -P
)"

command=

main() {
  _parse_params "$@"

  case "$command" in
  dynamics)
    _create_tmux "align_dynamics" 0 9 "_align_dynamics"
    ;;
  embodiment)
    date
    ;;
  viewpoint)
    date
    ;;
  esac
}

_print_usage() {
  cat <<EOF
usage: $(basename "$0") <subcommand> [arguments]

subcommands:
  dynamics
  embodiment
  viewpoint

optional arguments:
  -h, --help     Show this help and exit
EOF
}

_parse_params() {
  local param

  while [[ $# -gt 0 ]]; do
    param="$1"
    shift

    case $param in
    dynamics)
      command="dynamics"
      ;;
    embodiment)
      command="embodiment"
      ;;
    viewpoint)
      command="viewpoint"
      ;;
    -h | --help)
      _print_usage
      exit 0
      ;;
    -?*)
      echo "error: unrecognized argument: $param"
      exit 1
      ;;
    *)
      echo "error: unrecognized subcommand: $param"
      exit 1
      ;;
    esac
  done

  if [[ -z "$command" ]]; then
    echo "error: no subcommand specified"
    echo
    _print_usage
    exit 1
  fi
}

_cleanup() {
  :
}

_create_tmux() {
  local -r session="$1"

  typeset -i seed_begin seed_end seed
  local -r seed_begin=$2
  local -r seed_end=$3

  local command=

  tmux kill-session -t "$session" || true
  tmux new-session -d -s "$session"
  tmux set-window-option remain-on-exit on

  for ((seed = seed_begin; seed <= seed_end; ++seed)); do
    command=$($4 "$seed")
    tmux split-window -t "$session":1 "$command"
    tmux select-layout -t "$session":1 tiled
  done

  tmux attach -t "$session"
}

_align_dynamics() {
  echo "cd $HERE/dail && \
    poetry run dail \
    --algo ddpg \
    --agent_type gama \
    --load_dataset_dir $HERE/data/alignment_taskset/dynamics.pickle \
    --load_expert_dir $HERE/data/alignment_expert/reacher2_wall/12goals \
    --save_learner_dir $HERE/data/saved_alignments/dynamics/12goals/seed_${1} \
    --logdir $HERE/data/logs/dynamics/12goals/seed_${1} \
    --expert_domain reacher2_wall \
    --learner_domain reacher2_act_wall \
    --seed ${1} \
    --gpu -1"
}

main "$@"
