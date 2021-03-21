from typing import Dict

from dail.envs import DomainEnv
from dail.params import SavedParameters
from dail.utils.replay_memory import ReplayMemory


def create_replay_memory(
    env: Dict[str, DomainEnv],
    params: SavedParameters,
) -> Dict[str, ReplayMemory]:
    """Creates separate replay memories for each domain."""
    replay_memory = {}
    memory_type = params.train["memtype"]
    memory_capacity = params.train["memsize"]

    if memory_type != "vanilla":
        raise ValueError(
            (
                f"Failed to create replay memory with type: {memory_type}. "
                "Only support vanilla type."
            ),
        )

    for domain in env.keys():
        replay_memory[domain] = ReplayMemory(capacity=memory_capacity)
    replay_memory["model"] = ReplayMemory(capacity=memory_capacity)

    return replay_memory
