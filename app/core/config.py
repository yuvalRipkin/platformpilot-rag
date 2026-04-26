from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "platformpilot-rag"
    version: str = "0.1.0"


settings = Settings()
