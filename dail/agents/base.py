class Agent:
    """Abstract base class for the agents."""

    def __init__(self) -> None:
        pass

    def act(self) -> None:
        """Action."""
        raise NotImplementedError

    def train(self) -> None:
        """Train the agent."""
        raise NotImplementedError
