from dataclasses import dataclass

import gym
import numpy as np


@dataclass
class DomainEnv:
    """Environment for a domain."""

    name: str
    env: gym.Env
    env_type: str
    seed: int
    state_dim: int
    action_dim: int


_ENVS = {
    # Unmodified gym environments
    "reacher2": ("Reacher2DOF-v0", "goal"),
    "reacher2_wall": ("Reacher2DOFWall-v0", "goal"),
    "reacher2_corner": ("Reacher2DOFCorner-v0", "goal"),
    # Dynamics
    "reacher2_act": ("Reacher2DOFAct-v0", "goal"),
    "reacher2_act_wall": ("Reacher2DOFActWall-v0", "goal"),
    "reacher2_act_corner": ("Reacher2DOFActCorner-v0", "goal"),
    # Embodiment
    "reacher3": ("Reacher3DOF-v0", "goal"),
    "reacher3_wall": ("Reacher3DOFWall-v0", "goal"),
    "reacher3_corner": ("Reacher3DOFCorner-v0", "goal"),
    # Push
    "reacher2_push": ("Reacher2DOFPush-v0", "goal"),
    "reacher2_act_push": ("Reacher2DOFActPush-v0", "goal"),
    "reacher3_push": ("Reacher3DOFPush-v0", "goal"),
    # Viewpoint
    "tp_reacher2": ("TP_Reacher2DOF-v0", "goal"),
    "tp_write_reacher2": ("TP_WRITE_Reacher2DOF-v0", "goal"),
    "write_reacher2": ("WRITE_Reacher2DOF-v0", "goal"),
    # Longer reachers
    "reacher4": ("Reacher4DOF-v0", "goal"),
    "reacher5": ("Reacher5DOF-v0", "goal"),
    "reacher6": ("Reacher6DOF-v0", "goal"),
}


def make_env(env_name: str, seed: int) -> DomainEnv:
    if env_name not in _ENVS:
        raise ValueError(f"Failed to recognize environment: {env_name}.")

    (env_id, env_type) = _ENVS[env_name]
    env = gym.make(env_id)

    # Seed the chosen env
    env.seed(seed)

    return DomainEnv(
        name=env_name,
        env=env,
        env_type=env_type,
        seed=seed,
        state_dim=np.prod(np.array(env.observation_space.shape)),
        action_dim=np.prod(np.array(env.action_space.shape)),
    )
