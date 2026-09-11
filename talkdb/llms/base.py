from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """Abstract base class for all LLM providers."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Send a prompt to the LLM and return the raw text response."""

    @abstractmethod
    def generate_with_history(
        self, messages: list[dict[str, str]]
    ) -> str:
        """Send a conversation history (list of role/content dicts) and return a response."""

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return a human-readable name for this LLM provider."""
