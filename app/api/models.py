from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

class DetectionResult(BaseModel):
    """Модель результата обнаружения центра круга"""
    center_x: float = Field(..., description="X координата центра", ge=0)
    center_y: float = Field(..., description="Y координата центра", ge=0)
    normal_x: float = Field(..., description="X компонент нормали")
    normal_y: float = Field(..., description="Y компонент нормали")
    normal_z: float = Field(..., description="Z компонент нормали")
    radius: Optional[float] = Field(None, description="Радиус круга", ge=0)
    confidence: float = Field(1.0, description="Уверенность детекции", ge=0, le=1)
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Дополнительные метаданные")

class ImageSize(BaseModel):
    """Модель размеров изображения"""
    width: int = Field(..., description="Ширина изображения в пикселях", ge=1)
    height: int = Field(..., description="Высота изображения в пикселях", ge=1)

class DetectionResponse(BaseModel):
    """Модель ответа на запрос детекции"""
    detector_name: str = Field(..., description="Название использованного детектора")
    detector_version: str = Field(..., description="Версия детектора")
    centers: List[DetectionResult] = Field(..., description="Обнаруженные центры")
    image_size: ImageSize = Field(..., description="Размеры обработанного изображения")
    processing_time: Optional[float] = Field(None, description="Время обработки в секундах")
    warnings: List[str] = Field(default_factory=list, description="Предупреждения при обработке")
    used_cached: bool = Field(False, description="Был ли использован кэшированный экземпляр")

    model_config = {
        "json_schema_extra": {
            "example": {
                "detector_name": "Hough Circles Detector",
                "detector_version": "2.1.0",
                "centers": [
                    {
                        "center_x": 100.5,
                        "center_y": 150.2,
                        "normal_x": 0.0,
                        "normal_y": 0.0,
                        "normal_z": 1.0,
                        "radius": 25.0,
                        "confidence": 0.9,
                        "metadata": {}
                    }
                ],
                "image_size": {
                    "width": 800,
                    "height": 600
                },
                "processing_time": 0.15,
                "warnings": ["Unknown parameter: unknown_param"],
                "used_cached": True
            }
        }
    }

class DetectorParameter(BaseModel):
    """Модель параметра детектора"""
    name: str = Field(..., description="Название параметра")
    type: str = Field(..., description="Тип параметра (int, float, str, bool)")
    default_value: Any = Field(..., description="Значение по умолчанию")
    description: Optional[str] = Field(None, description="Описание параметра")
    min_value: Optional[float] = Field(None, description="Минимальное значение")
    max_value: Optional[float] = Field(None, description="Максимальное значение")

class DetectorInfo(BaseModel):
    """Модель информации о детекторе из meta.json"""
    name: str = Field(..., description="Название детектора")
    version: str = Field(..., description="Версия детектора")
    class_name: str = Field(..., description="Имя класса детектора")
    module_path: str = Field(..., description="Путь к модулю детектора")
    full_module_path: str = Field(..., description="Полный путь к модулю для импорта")
    description: str = Field(..., description="Описание детектора")
    required_parameters: List[str] = Field(..., description="Список обязательных параметров")
    optional_parameters: Dict[str, Any] = Field(..., description="Опциональные параметры с значениями по умолчанию")

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "Hough Circles Detector",
                "version": "2.1.0",
                "class_name": "HoughCirclesDetector",
                "module_path": "detectors/hough_circles",
                "full_module_path": "detectors.hough_circles.detector",
                "description": "Detects circles using Hough Transform with subpixel accuracy",
                "required_parameters": [],
                "optional_parameters": {
                    "dp": 1.2,
                    "min_dist": 50,
                    "param1": 100,
                    "param2": 30
                }
            }
        }
    }

class ValidationResult(BaseModel):
    """Модель результата валидации конфигурации"""
    valid: bool = Field(..., description="Результат валидации")
    errors: List[str] = Field(default_factory=list, description="Список ошибок")
    warnings: List[str] = Field(default_factory=list, description="Список предупреждений")
    meta_info: Optional[Dict[str, Any]] = Field(None, description="Мета-информация о детекторе")

    model_config = {
        "json_schema_extra": {
            "example": {
                "valid": True,
                "errors": [],
                "warnings": ["Unknown parameter: test_param"],
                "meta_info": {
                    "name": "Hough Circles Detector",
                    "version": "2.1.0"
                }
            }
        }
    }

class HealthCheck(BaseModel):
    """Модель ответа проверки здоровья"""
    status: str = Field(..., description="Статус сервиса")
    project_name: str = Field(..., description="Название проекта")
    version: str = Field(..., description="Версия API")
    timestamp: Optional[str] = Field(None, description="Время проверки")

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "healthy",
                "project_name": "Circle Centers Detection API",
                "version": "1.0.0",
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }
    }

class ErrorResponse(BaseModel):
    """Модель ответа об ошибке"""
    error: str = Field(..., description="Тип ошибки")
    message: str = Field(..., description="Сообщение об ошибке")
    details: Optional[Dict[str, Any]] = Field(None, description="Детали ошибки")

    model_config = {
        "json_schema_extra": {
            "example": {
                "error": "ValidationError",
                "message": "Invalid detector configuration",
                "details": {"missing_field": "module_path"}
            }
        }
    }

class CacheInfo(BaseModel):
    """Модель информации о кэше"""
    loaded_modules: List[str] = Field(..., description="Загруженные модули")
    cached_meta: List[str] = Field(..., description="Закэшированные мета-данные")
    cache_size: int = Field(..., description="Размер кэша")

# Модели для запросов
class DetectionRequest(BaseModel):
    """Модель запроса на детекцию"""
    detector_config: Dict[str, Any] = Field(..., description="Конфигурация детектора")

    model_config = {
        "json_schema_extra": {
            "example": {
                "detector_config": {
                    "module_path": "detectors/hough_circles",
                    "class_name": "HoughCirclesDetector",
                    "parameters": {
                        "min_dist": 50,
                        "param1": 100
                    }
                }
            }
        }
    }

class ValidationRequest(BaseModel):
    """Модель запроса на валидацию"""
    detector_config: Dict[str, Any] = Field(..., description="Конфигурация для валидации")