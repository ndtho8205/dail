from typing import List, Deque

import random
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class Transition:
    """A single transition in the environment."""

    state: np.ndarray
    action: int
    next_state: np.ndarray
    reward: float
    done: bool


class ReplayMemory:
    """A cyclic buffer to store the transitions observed recently."""

    def __init__(self, capacity: int):
        self.memory: Deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        """Return the current size of the memory."""
        return len(self.memory)

    def push(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
        reward: float,
        done: bool,
    ) -> None:
        """Save a transition."""
        self.memory.append(Transition(state, action, next_state, reward, done))

    def sample(self, batch_size: int) -> List[Transition]:
        """Randomly sample a batch of transitions from memory."""
        return random.sample(self.memory, batch_size)

    def load(self, loaded_deque: Deque[Transition]) -> None:
        """Load the replay memory from a saved file."""
        self.memory = loaded_deque
