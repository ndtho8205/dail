#!/usr/bin/env bash

# gen_weight = 0.01 and action_loss * 100 with learning rate 1e-5
# 7/10 worked! alignment is good 7/10

HERE="$(
  cd "$(dirname "$0")/.."
  pwd -P
)"

source "$HERE/scripts/set_env.sh"

# start new tmux sesson
SESS_NAME="eval_d_r2r"

tmux kill-session -t $SESS_NAME
tmux new-session -d -s $SESS_NAME

BEGIN=0
END=3

for ((i = BEGIN; i <= END; i++)); do
  gpu_num=$((i % TOTAL_GPU))

  PYTHON_CMD="\
        source '$VENV_DIR' && \
        cd $HERE/dail && \
        python train.py \
          --algo ddpg \
          --agent_type zeroshot \
          --load_expert_dir $HERE/data/target_expert/reacher2_corner/alldemo \
          --load_learner_dir $HERE/data/saved_alignments/dynamics/12goals/seed_${i} \
          --edomain reacher2_corner \
          --ldomain reacher2_act_corner \
          --seed 100${i} \
          --doc d_r2r"

  if [ $i -ne $BEGIN ]; then
    tmux selectp -t $SESS_NAME:1
    tmux split-window -h
    tmux send-keys -t $SESS_NAME:1 "$PYTHON_CMD" "C-m"
  else
    tmux selectp -t $SESS_NAME:1
    tmux send-keys -t $SESS_NAME:1 "$PYTHON_CMD" "C-m"
  fi

  sleep 0.1

  tmux select-layout tiled
done

tmux a -t $SESS_NAME
