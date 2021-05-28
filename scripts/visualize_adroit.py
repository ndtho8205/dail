#!/usr/bin/env python
import argparse

import gym

from dail.envs import register_adroit

# pen-human-v1         pen-cloned-v1         pen-expert-v1
# door-human-v1        door-cloned-v1        door-expert-v1
# hammer-human-v1      hammer-cloned-v1      hammer-expert-v1
# relocate-human-v1    relocate-cloned-v1    relocate-expert-v0v1

if __name__ == "__main__":
    register_adroit()

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="hammer-human-v1")
    args = parser.parse_args()

    env = gym.make(args.env_name)

    dataset = env.get_dataset()
    if "infos/qpos" not in dataset:
        raise ValueError("Only MuJoCo-based environments can be visualized")
    qpos = dataset["infos/qpos"]
    qvel = dataset["infos/qvel"]
    rewards = dataset["rewards"]
    actions = dataset["actions"]

    env.reset()
    env.set_state(qpos[0], qvel[0])
    for t in range(qpos.shape[0]):
        env.set_state(qpos[t], qvel[t])
        env.mj_render()
