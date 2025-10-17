from abc import ABC, abstractmethod
from typing import Dict, Generator, List
from models import ChatCompletionChunk


class ChatClient(ABC):
    """Abstract base class for chat clients following SOLID principles."""

    @abstractmethod
    def chat_completion(
        self, messages: List[Dict[str, str]]
    ) -> Generator[ChatCompletionChunk, None, None]:
        """Generate streaming chat completion.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys

        Returns:
            Generator yielding ChatCompletionChunk objects
        """
        pass
