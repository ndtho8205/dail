import pytest

from dail.envs import register_reacher


@pytest.fixture(scope="session", autouse=True)
def _register_envs() -> None:
    register_reacher()
