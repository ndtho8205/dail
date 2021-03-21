from typing import Any, Dict, Deque

import shelve
from pathlib import Path
from collections import deque

import numpy as np

from dail.utils import save_mp4


def create_hybrid_dataset(
    sess: Any,
    graph: Any,
    ph: Any,
    env: Dict[str, Dict[str, Any]],
    save_dir: Path,
    num_transitions: int = 20,
    save_video: bool = True,
) -> None:

    expert_deque: Deque[Any] = deque(maxlen=num_transitions)
    learner_deque: Deque[Any] = deque(maxlen=num_transitions)

    # ================ EXPERT DATASET ================
    frames = []
    steps = 0

    tot_reward = []
    while steps < num_transitions:
        done = False
        ep_reward = 0.0
        obs = env["expert"]["env"].reset()

        while not done:
            # Save dataset as video
            if save_video:
                img = env["expert"]["env"].render(mode="rgb_array")
                frames.append(img)

            # Get next action
            raw_action = sess.run(
                graph["expert"]["action"],
                feed_dict={
                    ph["expert"]["state"]: obs[None],
                    ph["expert"]["is_training"]: False,
                },
            )

            raw_action = raw_action[0]

            # Step in environment
            ## Add slight noise to the action space
            noisy_action = raw_action + np.random.normal(0, 0.05, size=raw_action.shape)
            #            noisy_action = raw_action
            next_obs, reward, done, info = env["expert"]["env"].step(noisy_action)

            ep_reward += reward
            steps += 1

            # Fill up the expert deque
            expert_transition = (
                obs,
                noisy_action,
                reward,
                next_obs,
                0.0 if done else 1.0,
                raw_action,
                0.0,
                0.0,
                0.0,
            )
            expert_deque.append(expert_transition)

            obs = next_obs

        # Keep track of reward
        tot_reward.append(ep_reward)

    # Create a video of the dataset
    if save_video:
        print("Saving video")
        save_mp4(frames, file_path=save_dir / "expert_dataset.mp4")

    print("Expert")
    print(f"Num Transitions: {steps}")
    print(f"Avg Reward: {np.mean(tot_reward)}")

    #    #================ IDENTITY DATASET ================
    #    frames = []
    #    tot_reward = []
    #    steps = 0
    #
    #    while steps < num_transitions:
    #        done = False
    #        ep_reward = 0.
    #        obs = env['expert']['env'].reset()
    #
    #
    #        while not done:
    #            # Save dataset as video
    #            if save_video:
    #                img = env['expert']['env'].render(mode='rgb_array')
    #                frames.append(img)
    #
    #            # Get next action
    #            raw_action = sess.run(graph['expert']['action'], feed_dict={ph['expert']['state']: obs[None],
    #                                                                        ph['expert']['is_training']: False})
    #
    #            raw_action = raw_action[0]
    #
    #
    #            # Step in environment
    #            ## Add slight noise to the action space
    #            noisy_action = raw_action + np.random.normal(0, 0.05, size=raw_action.shape)
    ##            noisy_action = raw_action
    #            next_obs, reward, done, info = env['expert']['env'].step(noisy_action)
    #
    #
    #            ep_reward += reward
    #            steps += 1
    #
    #            # Fill up the expert deque
    #            learner_transition = (obs, noisy_action, reward, next_obs, 0.0 if done else 1.0, raw_action, 0., 0., 0.)
    #            learner_deque.append(learner_transition)
    #
    #            obs = next_obs
    #
    #        # Keep track of reward
    #        tot_reward.append(ep_reward)
    #
    #
    #    print("-----------------------")
    #    print("Learner")
    #    print("Num Transitions: {}".format(steps))
    #    print("Avg Reward: {}".format(np.mean(tot_reward)))
    #
    #    # Create a video of the dataset
    #    if save_video:
    #        print("Saving video")
    #        save_frames_as_video(frames, filename='learner_dataset')

    #    #================ D_R2R DATASET ================
    #    frames = []
    #    tot_reward = []
    #    steps = 0
    #
    #    while steps < num_transitions:
    #        done = False
    #        ep_reward = 0.
    #        obs = env['learner']['env'].reset()
    #
    #
    #        while not done:
    #            # Save dataset as video
    #            if save_video:
    #                img = env['learner']['env'].render(mode='rgb_array')
    #                frames.append(img)
    #
    #            # Get next action
    #            raw_action = sess.run(graph['expert']['action'], feed_dict={ph['expert']['state']: obs[None],
    #                                                                        ph['expert']['is_training']: False})
    #
    #            raw_action = raw_action[0] / 10.
    #
    #
    #            # Step in environment
    #            ## Add slight noise to the action space
    #            noisy_action = raw_action + np.random.normal(0, 0.05, size=raw_action.shape) / 10.
    ##            noisy_action = raw_action
    #            next_obs, reward, done, info = env['learner']['env'].step(noisy_action)
    #
    #
    #            ep_reward += reward
    #            steps += 1
    #
    #            # Fill up the expert deque
    #            learner_transition = (obs, noisy_action, reward, next_obs, 0.0 if done else 1.0, raw_action, 0., 0., 0.)
    #            learner_deque.append(learner_transition)
    #
    #            obs = next_obs
    #
    #        # Keep track of reward
    #        tot_reward.append(ep_reward)
    #
    #
    #    print("-----------------------")
    #    print("Learner")
    #    print("Num Transitions: {}".format(steps))
    #    print("Avg Reward: {}".format(np.mean(tot_reward)))
    #
    #    # Create a video of the dataset
    #    if save_video:
    #        print("Saving video")
    #        save_frames_as_video(frames, filename='learner_dataset')

    # ========================== LEARNER DATASET ==========================
    frames = []
    tot_reward = []
    steps = 0

    while steps < num_transitions:
        done = False
        ep_reward = 0
        env["expert"]["env"].reset()
        obs = env["learner"]["env"].reset()

        while not done:
            mapped_state_raw, raw_action = sess.run(
                [graph["learner"]["mapped_state"], graph["learner"]["action"]],
                feed_dict={
                    ph["learner"]["state"]: obs[None],
                    ph["learner"]["is_training"]: False,
                },
            )

            raw_action = raw_action[0]
            mapped_state_raw = mapped_state_raw[0]

            env["expert"]["env"].env.set_state_from_obs(mapped_state_raw)
            # 		self.env['expert']['env'].set_state_from_obs(mapped_state_raw)

            # Render
            if save_video:
                # Concatenate learner and expert images
                limg = env["learner"]["env"].render(mode="rgb_array")
                eimg = env["expert"]["env"].render(mode="rgb_array")
                img = np.concatenate([limg, eimg], axis=1)
                frames.append(img)

            # Step
            noisy_action = raw_action + np.random.normal(0, 0.05, size=raw_action.shape)
            #                noisy_action = raw_action
            next_obs, reward, done, info = env["learner"]["env"].step(noisy_action)

            ep_reward += reward
            steps += 1

            # Fill up the expert deque
            learner_transition = (
                obs,
                noisy_action,
                reward,
                next_obs,
                0.0 if done else 1.0,
                raw_action,
                0.0,
                0.0,
                0.0,
            )
            learner_deque.append(learner_transition)

            obs = next_obs

        tot_reward.append(ep_reward)

    print("-----------------------")
    print("Learner")
    print(f"Num Transitions: {steps}")
    print(f"Avg Reward: {np.mean(tot_reward)}")

    # Create a video of the dataset
    if save_video:
        print("Saving learner video")
        save_mp4(frames, file_path=save_dir / "learner_dataset.mp4")

    hybrid_dataset = shelve.open(save_dir, writeback=True)
    hybrid_dataset["expert"] = expert_deque
    hybrid_dataset["learner"] = learner_deque
    hybrid_dataset.close()
