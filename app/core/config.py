import os
from typing import List, Optional
from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Настройки приложения"""

    # Основные настройки
    PROJECT_NAME: str = "Circle Centers Detection API"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "API для обнаружения центров кругов на изображениях"
    DEBUG: bool = True

    # Настройки сервера
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = True

    # CORS настройки
    ALLOWED_HOSTS: List[str] = ["*"]

    # Настройки детекторов
    DETECTORS_ROOT: str = "detectors"
    MAX_IMAGE_SIZE: int = 50 * 1024 * 1024  # 50MB
    SUPPORTED_IMAGE_TYPES: List[str] = ["image/jpeg", "image/png", "image/bmp", "image/tiff"]

    # Настройки безопасности
    API_PREFIX: str = "/api"
    SECRET_KEY: str = "your-secret-key-here"

    # Настройки логирования
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Временные файлы
    UPLOAD_DIR: str = "uploads"
    MAX_UPLOAD_SIZE: int = 50 * 1024 * 1024  # 50MB

    @field_validator("ALLOWED_HOSTS", mode="before")
    @classmethod
    def parse_allowed_hosts(cls, v):
        if isinstance(v, str):
            return [host.strip() for host in v.split(",")]
        return v

    @field_validator("SUPPORTED_IMAGE_TYPES", mode="before")
    @classmethod
    def parse_supported_image_types(cls, v):
        if isinstance(v, str):
            return [img_type.strip() for img_type in v.split(",")]
        return v

    model_config = {
        "env_file": ".env",
        "case_sensitive": True,
        "env_file_encoding": 'utf-8'
    }


# Глобальный экземпляр настроек
settings = Settings()


def get_settings() -> Settings:
    """Получение экземпляра настроек (для dependency injection)"""
    return settings