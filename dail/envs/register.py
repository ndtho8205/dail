from typing import Union

from gym.envs import register as register_gym_env
from gym.envs import registry


def register_reacher() -> None:
    # Mujoco environments
    _register(
        env_id="Reacher2DOF-v0",
        entry_point="dail.envs.reacher.reacher_2dof:Reacher2DOFEnv",
        max_episode_steps=60,
    )
    _register(
        env_id="Reacher2DOFWall-v0",
        entry_point="dail.envs.reacher.reacher_2dof_wall:Reacher2DOFWallEnv",
        max_episode_steps=60,
    )
    _register(
        env_id="Reacher2DOFCorner-v0",
        entry_point="dail.envs.reacher.reacher_2dof_corner:Reacher2DOFCornerEnv",
        max_episode_steps=60,
    )

    # Dynamics
    _register(
        env_id="Reacher2DOFSto-v0",
        entry_point="dail.envs.reacher.reacher_2dof_sto:Reacher2DOFStoEnv",
        max_episode_steps=60,
    )
    _register(
        env_id="Reacher2DOFAct-v0",
        entry_point="dail.envs.reacher.reacher_2dof_act:Reacher2DOFActEnv",
        max_episode_steps=60,
    )
    _register(
        env_id="Reacher2DOFActWall-v0",
        entry_point="dail.envs.reacher.reacher_2dof_act_wall:Reacher2DOFActWallEnv",
        max_episode_steps=60,
    )
    _register(
        env_id="Reacher2DOFActCorner-v0",
        entry_point="dail.envs.reacher.reacher_2dof_act_corner:Reacher2DOFActCornerEnv",
        max_episode_steps=60,
    )

    # embodiment altered
    _register(
        env_id="Reacher3DOF-v0",
        entry_point="dail.envs.reacher.reacher_3dof:Reacher3DOFEnv",
        max_episode_steps=60,
    )
    _register(
        env_id="Reacher3DOFWall-v0",
        entry_point="dail.envs.reacher.reacher_3dof_wall:Reacher3DOFWallEnv",
        max_episode_steps=60,
    )
    _register(
        env_id="Reacher3DOFCorner-v0",
        entry_point="dail.envs.reacher.reacher_3dof_corner:Reacher3DOFCornerEnv",
        max_episode_steps=60,
    )

    # Push
    _register(
        env_id="Reacher2DOFPush-v0",
        entry_point="dail.envs.reacher.reacher_2dof_push:Reacher2DOFPushEnv",
        max_episode_steps=500,
    )
    _register(
        env_id="Reacher2DOFActPush-v0",
        entry_point="dail.envs.reacher.reacher_2dof_act_push:Reacher2DOFActPushEnv",
        max_episode_steps=500,
    )
    _register(
        env_id="Reacher3DOFPush-v0",
        entry_point="dail.envs.reacher.reacher_3dof_push:Reacher3DOFPushEnv",
        max_episode_steps=500,
    )

    # Viewpoint
    _register(
        env_id="TP_Reacher2DOF-v0",
        entry_point="dail.envs.reacher.tp_reacher_2dof:TP_Reacher2DOFEnv",
        max_episode_steps=60,
    )
    _register(
        env_id="TP_WRITE_Reacher2DOF-v0",
        entry_point="dail.envs.reacher.tp_write_reacher_2dof:TP_WRITE_Reacher2DOFEnv",
        max_episode_steps=500,
    )
    _register(
        env_id="WRITE_Reacher2DOF-v0",
        entry_point="dail.envs.reacher.write_reacher_2dof:WRITE_Reacher2DOFEnv",
        max_episode_steps=500,
    )

    # Longer reachers
    _register(
        env_id="Reacher4DOF-v0",
        entry_point="dail.envs.reacher.reacher_4dof:Reacher4DOFEnv",
        max_episode_steps=60,
    )
    _register(
        env_id="Reacher5DOF-v0",
        entry_point="dail.envs.reacher.reacher_5dof:Reacher5DOFEnv",
        max_episode_steps=70,
    )
    _register(
        env_id="Reacher6DOF-v0",
        entry_point="dail.envs.reacher.reacher_6dof:Reacher6DOFEnv",
        max_episode_steps=80,
    )


def register_adroit() -> None:
    from .adroit import infos

    # V1 envs
    # pen-human-v1         pen-cloned-v1         pen-expert-v1
    # door-human-v1        door-cloned-v1        door-expert-v1
    # hammer-human-v1      hammer-cloned-v1      hammer-expert-v1
    # relocate-human-v1    relocate-cloned-v1    relocate-expert-v0v1
    max_steps = {
        "pen": 100,
        "door": 200,
        "hammer": 200,
        "relocate": 200,
    }
    env_mapping = {
        "pen": "PenEnvV0",
        "door": "DoorEnvV0",
        "hammer": "HammerEnvV0",
        "relocate": "RelocateEnvV0",
    }

    for env in ["pen", "door", "hammer", "relocate"]:
        for dataset in ["human", "cloned", "expert"]:
            env_name = f"{env}-{dataset}-v1"
            _register(
                env_id=env_name,
                entry_point="dail.envs.adroit:" + env_mapping[env],
                max_episode_steps=max_steps[env],
                dataset_url=infos["dataset_urls"][env_name],
                ref_min_score=infos["ref_min_score"][env_name],
                ref_max_score=infos["ref_max_score"][env_name],
            )


def _register(env_id: str, **kwargs: Union[str, float, int]) -> None:
    if env_id not in registry.env_specs.keys():
        register_gym_env(id=env_id, **kwargs)
