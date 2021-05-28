#!/usr/bin/env bash

HERE="$(
  cd "$(dirname "$0")/.." || exit 1
  pwd -P
)"

readonly seed=$1

cd "$HERE/dail" &&
  poetry run dail \
    --expert_domain reacher2_wall \
    --learner_domain reacher2_act_wall \
    --seed "$seed" \
    --agent_type gama \
    --algo ddpg \
    --logdir "$HERE/data/logs/dynamics/12goals/seed_${seed}" \
    #TODO: Check the below args
    --load_dataset_dir "$HERE/data/alignment_taskset/dynamics.pickle" \
    --load_expert_dir "$HERE/data/alignment_expert/reacher2_wall/12goals" \
    --save_learner_dir "$HERE/data/saved_alignments/dynamics/12goals/seed_${seed}" \
    --gpu -1
