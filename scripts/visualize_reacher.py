#!/usr/bin/env python3

import time

import gym

from dail.envs import register_reacher_envs

# Environments
#   Reacher2DOF-v0
#   Reacher2DOFWall-v0
#   Reacher2DOFCorner-v0
# Dynamics
#   Reacher2DOFAct-v0
#   Reacher2DOFActWall-v0
#   Reacher2DOFActCorner-v0
# Embodiment
#   Reacher3DOF-v0
#   Reacher3DOFWall-v0
#   Reacher3DOFCorner-v0
# Push
#   Reacher2DOFPush-v0
#   Reacher2DOFActPush-v0
#   Reacher3DOFPush-v0
# Viewpoint
#   TP_Reacher2DOF-v0
#   TP_WRITE_Reacher2DOF-v0
#   WRITE_Reacher2DOF-v0
# Longer reachers
#   Reacher4DOF-v0
#   Reacher5DOF-v0
#   Reacher6DOF-v0


def main() -> None:
    register_reacher_envs()

    env_name = "Reacher3DOFPush-v0"
    env = gym.make(env_name)

    while True:
        step = 0
        done = False

        env.reset()
        while not done:
            step += 1
            env.render()
            action = env.action_space.sample()
            observation, reward, done, _ = env.step(action)

            print(
                (
                    f"Step: {step}\n"
                    f"  Reward: {reward:0.4f} | Action: {action} | State: {observation}"
                ),
            )

            time.sleep(0.01)


if __name__ == "__main__":
    main()
