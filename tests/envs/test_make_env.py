import pytest

from dail.envs import make_env


@pytest.mark.parametrize(
    ("domain_env_name", "expected_domain_id", "expected_env_type"),
    [
        ("reacher2", "Reacher2DOF-v0", "goal"),
        ("reacher2_corner", "Reacher2DOFCorner-v0", "goal"),
        ("reacher2_wall", "Reacher2DOFWall-v0", "goal"),
    ],
)
def test_should_work(
    domain_env_name: str,
    expected_domain_id: str,
    expected_env_type: str,
) -> None:
    seed = 3
    domain_env = make_env(domain_env_name, seed)
    print(domain_env)

    assert domain_env.name == domain_env_name
    assert domain_env.env_type == expected_env_type
    assert domain_env.seed == seed


def test_should_raise_an_error() -> None:
    expected_error_message = "Failed to recognize environment: unknown."

    with pytest.raises(ValueError, match=expected_error_message):
        make_env("unknown", 9)
