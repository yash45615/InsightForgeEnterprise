from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "InsightForge Enterprise"
    app_version: str = "0.1.0"
    environment: str = "development"

    # Start with a local SQLite database.
    # We will configure PostgreSQL in Phase 2.
    database_url: str = "sqlite:///./insightforge.db"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
