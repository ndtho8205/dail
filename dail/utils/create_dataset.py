from typing import Any, Dict

from pathlib import Path

import numpy as np

from dail.utils import save_mp4


def create_dataset(
    sess: Any,
    graph: Any,
    ph: Any,
    env: Dict[str, Dict[str, Any]],
    save_dir: Path,
    num_rollout: int = 20,
    save_video: bool = True,
    vid_name: str = "demonstrations.mp4",
) -> None:

    frames = []
    tot_reward = []
    total_obs = []
    total_acs = []

    if save_video:
        print("Saving video")
    else:
        print("Creating transfer dataset")

    for _ in range(num_rollout):
        done = False
        obs = env["expert"]["env"].reset()

        steps = 0
        ep_reward = 0.0
        ep_obs = []
        ep_acs = []

        while not done:
            # Get next action
            (action,) = sess.run(
                graph["expert"]["action"],
                feed_dict={
                    ph["expert"]["state"]: obs[None],
                    ph["expert"]["is_training"]: False,
                },
            )
            ep_obs.append(np.squeeze(obs))
            ep_acs.append(np.squeeze(action))

            # Save dataset as video
            if save_video:
                eimg = env["expert"]["env"].render(mode="rgb_array")
                frames.append(eimg)

            # Step in environment
            # Add slight noise to the action space
            action += np.random.normal(0, 0.05)
            obs, reward, done, _ = env["expert"]["env"].step(action)

            ep_reward += reward
            steps += 1

        tot_reward.append(ep_reward)
        total_obs.append(np.array(ep_obs))
        total_acs.append(np.array(ep_acs))

    # Print metrics
    print(f"Steps: {steps}")
    print(f"Avg Reward: {np.mean(tot_reward)}")

    # Create a video of the dataset
    if save_video:
        save_mp4(frames, file_path=save_dir / vid_name)

    # Save into dataset
    print(f"Saved {num_rollout} demonstrations to {save_dir}")
    # shape [num_demo, ep_len, data_dim]
    total_obs = np.array(total_obs, dtype="object")
    total_acs = np.array(total_acs, dtype="object")
    np.savez(save_dir, obs=total_obs, acs=total_acs)
