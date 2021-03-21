from collections import deque

import numpy as np

from dail.utils import Transition, ReplayMemory


def test_should_work() -> None:
    memory = ReplayMemory(capacity=10)
    transition = Transition(np.array([]), 1, np.array([]), 0.0, False)

    assert len(memory) == 0

    memory.push(
        transition.state,
        transition.action,
        transition.next_state,
        transition.reward,
        transition.done,
    )
    assert len(memory) == 1

    batch = memory.sample(1)
    assert len(batch) == 1
    assert batch == [transition]

    memory.load(deque([transition, transition]))
    assert len(memory) == 2
