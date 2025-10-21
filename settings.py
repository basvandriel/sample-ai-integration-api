from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator, model_validator
from typing import Optional
import os


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env" if os.path.exists(".env") else None,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    CHAT_PROVIDER: str = Field(
        default="openai", description="Chat provider to use: 'openai' or 'ollama'"
    )
    CHAT_MODEL: str = Field(
        default="gpt-4.1", description="Model to use for chat completions"
    )

    # OpenAI configuration (optional when using Ollama)
    OPENAI_API_KEY: Optional[str] = Field(
        default=None, description="OpenAI API key (required when using OpenAI)"
    )
    OPENAI_API_ENDPOINT: Optional[str] = Field(
        default=None, description="OpenAI API endpoint (optional)"
    )
    OPENAI_API_VERSION: Optional[str] = Field(
        default=None, description="OpenAI API version (optional)"
    )

    # Ollama configuration (optional when using OpenAI)
    OLLAMA_HOST: str = Field(
        default="http://localhost:11434", description="Ollama server host"
    )

    @field_validator("CHAT_PROVIDER")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        if v.lower() not in ["openai", "ollama"]:
            raise ValueError("CHAT_PROVIDER must be either 'openai' or 'ollama'")
        return v.lower()

    @model_validator(mode="after")
    def validate_provider_requirements(self) -> "Settings":
        """Validate that required fields are present based on the provider."""
        if self.CHAT_PROVIDER == "openai" and not self.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is required when CHAT_PROVIDER is 'openai'"
            )
        return self
