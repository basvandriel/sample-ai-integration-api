from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    OPENAI_API_KEY: str = Field(
        description="OpenAI API key"
    )
    OPENAI_API_ENDPOINT: str = Field(
        description="OpenAI API endpoint"
    )
    OPENAI_API_VERSION: str = Field(
        description="OpenAI API version"
    )
