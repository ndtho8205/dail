import pytest

from dail.envs import make_envs


@pytest.mark.parametrize(
    ("expert_domain", "learner_domain"),
    [
        ("reacher2", "reacher2_corner"),
        ("reacher2", "reacher2_wall"),
        ("reacher2_wall", "reacher2_corner"),
    ],
)
def test_should_work(
    expert_domain: str,
    learner_domain: str,
) -> None:
    envs = make_envs(expert_domain, learner_domain, 3)

    assert "expert" in envs
    assert "learner" in envs
