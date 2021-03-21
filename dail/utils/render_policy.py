from typing import Any, Dict

from pathlib import Path

import numpy as np

from dail.utils import save_mp4


def render_policy(
    sess: Any,
    graph: Any,
    ph: Any,
    env: Dict[str, Dict[str, Any]],
    domain: str,
    num_rollout: int = 20,
    save_video: bool = True,
    save_dir: Path = Path.cwd() / "temp",
) -> float:
    frames = []
    tot_reward = []

    if save_video:
        print("Saving video")
    else:
        print("Evaluating expert performance")

    for _ in range(num_rollout):
        done = False
        obs = env[domain]["env"].reset()
        steps = 0
        ep_reward = 0.0

        while not done:
            if save_video:
                frames.append(env[domain]["env"].render(mode="rgb_array"))

            (action,) = sess.run(
                graph[domain]["action"],
                feed_dict={
                    ph[domain]["state"]: obs[None],
                    ph[domain]["is_training"]: False,
                },
            )
            obs, reward, done, _ = env[domain]["env"].step(action)

            ep_reward += reward
            steps += 1

        tot_reward.append(ep_reward)

    if save_video:
        save_mp4(frames, save_dir / "temp.mp4")

    avg_reward = np.mean(tot_reward)
    print(f"Steps: {steps}")
    print(f"Avg Reward: {avg_reward}")

    return avg_reward
