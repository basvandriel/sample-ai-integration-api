
import logging
from settings import Settings


def load_settings() -> Settings:
    """Load and log application settings."""
    logger = logging.getLogger("uvicorn")  # noqa: F821
    settings = Settings()
    
    # Log loaded environment variables in uvicorn style
    field_names = ", ".join(settings.model_fields.keys())
    logger.info(f"Loading environment variables: {field_names}")
    
    return settings