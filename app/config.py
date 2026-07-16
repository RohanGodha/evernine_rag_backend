"""Application configuration, loaded from environment / .env."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # LLM (Groq, OpenAI-compatible)
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-20b"
    groq_base_url: str = "https://api.groq.com/openai/v1"
    llm_max_retries: int = 2

    # Database
    database_url: str = (
        "postgresql+psycopg2://postgres:postgres@localhost:5432/aster_oak"
    )

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
