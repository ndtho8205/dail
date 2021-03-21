import pytest

from dail.utils import create_replay_memory
from dail.params import SavedParameters


def test_should_work() -> None:
    env = {
        "expert": None,
        "learner": None,
    }
    params = SavedParameters(
        train={
            "memtype": "vanilla",
            "memsize": 1000,
        },
        expert={},
        learner={},
        behavior_cloning={},
    )

    memories = create_replay_memory(env, params)  # type: ignore

    assert "expert" in memories
    assert "learner" in memories
    assert "model" in memories


def test_should_raise_an_error() -> None:
    env = {
        "expert": None,
        "learner": None,
    }
    params = SavedParameters(
        train={
            "memtype": "unknown",
            "memsize": 1000,
        },
        expert={},
        learner={},
        behavior_cloning={},
    )

    expected_error_message = (
        "Failed to create replay memory with type: unknown. Only support vanilla type."
    )

    with pytest.raises(ValueError, match=expected_error_message):
        create_replay_memory(env, params)  # type: ignore
