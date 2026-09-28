from abc import ABC, abstractmethod

from src.state import TicketContext


class Agent(ABC):
    """Common interface every pipeline agent implements."""

    name: str = "agent"

    @abstractmethod
    def run(self, context: TicketContext) -> TicketContext:
        raise NotImplementedError
