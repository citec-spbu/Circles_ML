from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import json
import logging
from typing import List, Dict, Any

from app.core.detector_factory import DetectorFactory, DetectorMeta
from app.api.models import (
    DetectionRequest,
    DetectionResponse,
    DetectorInfo,
    HealthCheck,
    ValidationResult
)

# Настройка логирования
logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/detect",
    response_model=DetectionResponse,
    summary="Обнаружение центров кругов на изображении",
    description="""
    Принимает изображение и конфигурацию детектора, возвращает координаты центров кругов.

    - **file**: Изображение в формате JPEG, PNG
    - **detector_config**: JSON строка с конфигурацией детектора
    """
)
async def detect_circles(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(..., description="Изображение для анализа"),
        detector_config: str = Form(..., description="Конфигурация детектора в JSON формате"),
        validate_only: bool = Query(False, description="Только валидация без выполнения"),
        use_cache: bool = Query(True, description="Использовать кэшированные экземпляры детекторов")
):
    """
    Основной эндпоинт для обнаружения центров кругов
    """
    try:
        # Валидация файла
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="Файл должен быть изображением"
            )

        logger.info(f"Processing image: {file.filename}")

        # Чтение и декодирование изображения
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(
                status_code=400,
                detail="Не удалось декодировать изображение"
            )

        # Парсинг конфигурации детектора
        try:
            config = json.loads(detector_config)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Неверный формат JSON конфигурации: {e}"
            )

        # Валидация конфигурации
        validation_result = DetectorFactory.validate_detector_config(config)

        if not validation_result["valid"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid detector configuration: {', '.join(validation_result['errors'])}"
            )

        # Если только валидация - возвращаем результат
        if validate_only:
            return JSONResponse({
                "valid": True,
                "message": "Configuration is valid",
                "warnings": validation_result["warnings"],
                "meta_info": validation_result["meta_info"]
            })

        # Предупреждения (если есть)
        if validation_result["warnings"]:
            logger.warning(f"Configuration warnings: {validation_result['warnings']}")

        # Создание или получение детектора
        try:
            if use_cache:
                # Используем кэшированный экземпляр
                detector = DetectorFactory.get_or_create_detector(config)
                logger.info("Using cached detector instance")
            else:
                # Создаем новый экземпляр
                detector = DetectorFactory.create_detector(config)
                logger.info("Created new detector instance")
        except Exception as e:
            logger.error(f"Detector creation failed: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Ошибка создания детектора: {str(e)}"
            )

        # Обнаружение центров
        try:
            results = detector.detect(image)
            logger.info(f"Detected {len(results)} circles")
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Ошибка во время детекции: {str(e)}"
            )

        # Очистка ресурсов только для НЕкэшированных экземпляров
        if not use_cache and hasattr(detector, 'cleanup'):
            background_tasks.add_task(detector.cleanup)

        return DetectionResponse(
            detector_name=DetectorFactory.get_detector_name(config['module_path'], config['class_name']),
            detector_version=DetectorFactory.get_detector_version(config['module_path'], config['class_name']),
            centers=[result.dict() for result in results],
            image_size={"width": image.shape[1], "height": image.shape[0]},
            warnings=validation_result["warnings"],
            used_cached=use_cache
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Внутренняя ошибка сервера: {str(e)}"
        )


@router.get(
    "/detectors",
    response_model=List[DetectorInfo],
    summary="Список доступных детекторов",
    description="Возвращает информацию о всех автоматически обнаруженных детекторах (без загрузки модулей)"
)
async def get_available_detectors(
        force_refresh: bool = Query(False, description="Принудительное обновление списка")
):
    """Получение списка доступных детекторов"""
    try:
        if force_refresh:
            DetectorFactory.clear_cache()

        detectors = DetectorFactory.discover_available_detectors()
        return detectors
    except Exception as e:
        logger.error(f"Failed to discover detectors: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при получении списка детекторов: {str(e)}"
        )


@router.get(
    "/detectors/{detector_path}/{class_name}",
    response_model=DetectorInfo,
    summary="Информация о конкретном детекторе",
    description="Возвращает детальную информацию о конкретном детекторе"
)
async def get_detector_info(
        detector_path: str,
        class_name: str
):
    """Получение информации о конкретном детекторе"""
    try:
        meta_info = DetectorFactory.get_detector_meta(detector_path, class_name)

        if not meta_info:
            raise HTTPException(
                status_code=404,
                detail=f"Detector not found: {detector_path}.{class_name}"
            )

        return meta_info
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get detector info: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при получении информации о детекторе: {str(e)}"
        )


@router.post(
    "/validate",
    response_model=ValidationResult,
    summary="Валидация конфигурации детектора",
    description="Проверяет конфигурацию детектора без выполнения детекции"
)
async def validate_detector_config(
        detector_config: str = Form(..., description="Конфигурация детектора в JSON формате")
):
    """Валидация конфигурации детектора"""
    try:
        config = json.loads(detector_config)
        validation_result = DetectorFactory.validate_detector_config(config)

        return ValidationResult(
            valid=validation_result["valid"],
            errors=validation_result["errors"],
            warnings=validation_result["warnings"],
            meta_info=validation_result.get("meta_info")
        )

    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid JSON format: {e}"
        )
    except Exception as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Validation error: {str(e)}"
        )


@router.get(
    "/debug/cache",
    summary="Отладочная информация о кэше",
    description="Возвращает информацию о загруженных модулях и кэше (только для отладки)"
)
async def get_cache_info():
    """Отладочная информация о кэше"""
    return {
        "cached_instances": DetectorFactory.get_cached_instances(),
        "cached_classes": DetectorFactory.get_cached_classes(),
        "cached_meta": DetectorFactory.get_cached_meta(),
        "cache_size": len(DetectorFactory.get_cached_instances())
    }


@router.delete(
    "/debug/cache",
    summary="Очистка кэша",
    description="Очищает все кэши детекторов (только для отладки)"
)
async def clear_cache():
    """Очистка кэша"""
    DetectorFactory.clear_cache()
    return {"message": "Cache cleared successfully"}


@router.get(
    "/health",
    response_model=HealthCheck,
    summary="Проверка здоровья сервиса"
)
async def health_check():
    """Проверка работоспособности сервиса"""
    return HealthCheck(
        status="healthy",
        project_name="Circle Centers Detection API",
        version="1.0.0"
    )