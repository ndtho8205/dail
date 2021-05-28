from gym.envs import registry

from dail.envs import register_reacher_envs


def test_should_work():
    register_reacher_envs()
