from typing import Optional
from chat_client import ChatClient
from ollama_client import OllamaClient
from openai_client import OpenAIClient


class ChatClientFactory:
    """Factory for creating chat clients based on configuration."""

    @staticmethod
    def create_client(
        provider: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_version: Optional[str] = None,
        host: Optional[str] = None,
    ) -> ChatClient:
        if provider.lower() == "ollama":
            return OllamaClient(
                model=model or "mistral", host=host or "http://localhost:11434"
            )
        elif provider.lower() == "openai":
            if not api_key:
                raise ValueError("API key is required for OpenAI client")

            return OpenAIClient(
                api_key=api_key,
                model=model or "gpt-3.5-turbo",
                base_url=base_url,
                api_version=api_version,
            )
        else:
            raise ValueError(
                f"Unsupported provider: {provider}. Use 'ollama' or 'openai'"
            )
