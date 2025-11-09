import importlib
import importlib.util
import sys
import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Type
from datetime import datetime

from detectors import BaseDetector

# Настройка логирования
logger = logging.getLogger(__name__)


class DetectorMeta:
    """Класс для работы с мета-информацией детекторов"""

    def __init__(self, meta_data: Dict[str, Any], detector_path: Path):
        self.meta_data = meta_data
        self.detector_path = detector_path
        self._validate_meta()

    def _validate_meta(self):
        """Валидация обязательных полей в мета-данных"""
        required_fields = ["name", "version", "class_name", "module_path"]
        for field in required_fields:
            if field not in self.meta_data:
                raise ValueError(f"Missing required field in meta.json: {field}")

    @property
    def name(self) -> str:
        return self.meta_data["name"]

    @property
    def version(self) -> str:
        return self.meta_data["version"]

    @property
    def class_name(self) -> str:
        return self.meta_data["class_name"]

    @property
    def module_path(self) -> str:
        return self.meta_data["module_path"]

    @property
    def description(self) -> str:
        return self.meta_data.get("description", "")

    @property
    def required_parameters(self) -> List[str]:
        return self.meta_data.get("required_parameters", [])

    @property
    def optional_parameters(self) -> Dict[str, Any]:
        return self.meta_data.get("optional_parameters", {})

    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует мета-данные в словарь для API"""
        # Строим полный путь к модулю
        package_name = self.detector_path.name
        full_module_path = f"detectors.{package_name}.{self.module_path}"

        return {
            "name": self.name,
            "version": self.version,
            "class_name": self.class_name,
            "module_path": str(self.detector_path),
            "full_module_path": full_module_path,
            "description": self.description,
            "required_parameters": self.required_parameters,
            "optional_parameters": self.optional_parameters
        }


class DetectorFactory:
    """Фабрика для загрузки и управления внешними детекторами"""

    # Кэш для экземпляров детекторов (основной кэш)
    _instance_cache: Dict[str, BaseDetector] = {}
    # Кэш для классов детекторов (для создания новых экземпляров при разных параметрах)
    _class_cache: Dict[str, Type[BaseDetector]] = {}
    # Кэш для мета-данных детекторов
    _meta_cache: Dict[str, DetectorMeta] = {}

    @staticmethod
    def get_detectors_root() -> Path:
        """Возвращает абсолютный путь к корневой директории детекторов"""
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent
        detectors_root = project_root / "detectors"
        return detectors_root

    @staticmethod
    def discover_available_detectors(detectors_root: Optional[str] = None) -> List[Dict]:
        """
        Обнаруживает доступные детекторы через meta.json файлы
        БЕЗ загрузки самих модулей!
        """
        if detectors_root is None:
            detectors_root = DetectorFactory.get_detectors_root()
        else:
            detectors_root = Path(detectors_root)

        available_detectors = []

        logger.info(f"Discovering detectors in: {detectors_root.absolute()}")

        if not detectors_root.exists():
            logger.warning(f"Detectors directory does not exist: {detectors_root}")
            return available_detectors

        # Очищаем кэш мета-данных при каждом обнаружении
        DetectorFactory._meta_cache.clear()

        for item in detectors_root.iterdir():
            if item.is_dir() and not item.name.startswith('__'):
                try:
                    meta_info = DetectorFactory._load_meta_info(item)
                    if meta_info:
                        detector_info = meta_info.to_dict()
                        available_detectors.append(detector_info)

                        # Кэшируем мета-информацию
                        cache_key = f"{meta_info.module_path}.{meta_info.class_name}"
                        DetectorFactory._meta_cache[cache_key] = meta_info

                        logger.info(f"Found detector: {meta_info.name} v{meta_info.version}")
                except Exception as e:
                    logger.warning(f"Failed to load detector meta from {item}: {e}")
                    continue

        logger.info(f"Found {len(available_detectors)} available detectors")
        return available_detectors

    @staticmethod
    def _load_meta_info(detector_path: Path) -> Optional[DetectorMeta]:
        """
        Загружает мета-информацию из meta.json файла
        """
        meta_file = detector_path / "meta.json"

        if not meta_file.exists():
            logger.warning(f"No meta.json found in {detector_path}")
            return None

        try:
            with open(meta_file, 'r', encoding='utf-8') as f:
                meta_data = json.load(f)

            return DetectorMeta(meta_data, detector_path)

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {meta_file}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error reading meta.json from {detector_path}: {e}")
            return None

    @staticmethod
    def get_or_create_detector(detector_config: Dict[str, Any]) -> BaseDetector:
        """
        Возвращает существующий экземпляр детектора ИЛИ создает новый
        Экземпляры кэшируются по комбинации module_path + class_name + параметры
        """
        module_path = detector_config['module_path']
        class_name = detector_config['class_name']
        parameters = detector_config.get('parameters', {})

        # Создаем ключ для кэша экземпляров
        # Учитываем все параметры для уникальности конфигурации
        params_key = str(sorted(parameters.items()))
        instance_key = f"{module_path}.{class_name}.{params_key}"

        # Проверяем кэш экземпляров
        if instance_key in DetectorFactory._instance_cache:
            logger.debug(f"Using cached detector instance: {instance_key}")
            detector = DetectorFactory._instance_cache[instance_key]

            # Обновляем конфигурацию если нужно
            if hasattr(detector, 'update_config'):
                detector.update_config(**parameters)
            elif hasattr(detector, 'config'):
                detector.config.update(parameters)

            return detector

        # Создаем новый экземпляр
        start_time = datetime.now()

        try:
            # Получаем класс детектора (кэшируется)
            detector_class = DetectorFactory._get_detector_class(module_path, class_name)

            # Создаем экземпляр
            detector = detector_class(**parameters)

            # Кэшируем экземпляр
            DetectorFactory._instance_cache[instance_key] = detector

            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Detector created and cached in {processing_time:.3f}s: {instance_key}")

            return detector

        except Exception as e:
            logger.error(f"Failed to create detector: {e}")
            raise ValueError(f"Ошибка создания детектора: {str(e)}")

    @staticmethod
    def _get_detector_class(module_path: str, class_name: str) -> Type[BaseDetector]:
        """
        Возвращает класс детектора (кэшируется)
        """
        class_cache_key = f"{module_path}.{class_name}"

        # Проверяем кэш классов
        if class_cache_key in DetectorFactory._class_cache:
            logger.debug(f"Using cached detector class: {class_cache_key}")
            return DetectorFactory._class_cache[class_cache_key]

        # Ищем в кэше мета-данных
        meta_info = DetectorFactory._meta_cache.get(class_cache_key)

        if meta_info:
            # Используем полный путь из мета-данных
            full_module_path = meta_info.to_dict()["full_module_path"]
        else:
            # Fallback: строим путь из конфигурации
            detectors_root = DetectorFactory.get_detectors_root()
            detector_abs_path = detectors_root / module_path
            package_name = detector_abs_path.name
            full_module_path = f"detectors.{package_name}.detector"

        print(f"🔍 DEBUG: Loading detector class: {full_module_path}.{class_name}")

        # Загрузка модуля
        module = DetectorFactory._load_module(full_module_path)

        # Получение класса детектора
        detector_class = getattr(module, class_name)

        # Валидация класса
        if not issubclass(detector_class, BaseDetector):
            raise TypeError(f"Class {class_name} must inherit from BaseDetector")

        # Кэшируем класс
        DetectorFactory._class_cache[class_cache_key] = detector_class
        logger.debug(f"Cached detector class: {class_cache_key}")

        return detector_class

    @staticmethod
    def create_detector(detector_config: Dict[str, Any]) -> BaseDetector:
        """
        Создает новый экземпляр детектора (без кэширования)
        Для обратной совместимости
        """
        module_path = detector_config['module_path']
        class_name = detector_config['class_name']
        parameters = detector_config.get('parameters', {})

        detector_class = DetectorFactory._get_detector_class(module_path, class_name)
        return detector_class(**parameters)

    @staticmethod
    def _load_module(full_module_path: str) -> Any:
        """
        Загружает модуль по полному пути модуля
        """
        print(f"🔍 DEBUG: Loading module: {full_module_path}")

        try:
            # Загрузка через importlib (для пакетов)
            module = importlib.import_module(full_module_path)

            logger.debug(f"Successfully loaded module: {full_module_path}")

            return module

        except ImportError as e:
            logger.warning(f"Import failed for {full_module_path}: {e}")
            # Используем fallback
            return DetectorFactory._load_module_fallback(full_module_path)

    @staticmethod
    def _load_module_fallback(full_module_path: str) -> Any:
        """
        Fallback метод для загрузки модуля из файла
        """
        try:
            # Парсим путь из full_module_path
            parts = full_module_path.split('.')
            if len(parts) >= 3 and parts[0] == 'detectors':
                package_name = parts[1]
                module_name = parts[2] if len(parts) > 2 else 'detector'

                # Строим путь к файлу
                detectors_root = DetectorFactory.get_detectors_root()
                detector_path = detectors_root / package_name
                module_file = detector_path / f"{module_name}.py"

                if not module_file.exists():
                    raise ValueError(f"Module file not found: {module_file}")

                print(f"🔍 DEBUG: Fallback loading from: {module_file}")

                # Создание спецификации из файла
                spec = importlib.util.spec_from_file_location(full_module_path, module_file)
                if spec is None or spec.loader is None:
                    raise ValueError(f"Failed to create spec for module: {module_file}")

                module = importlib.util.module_from_spec(spec)

                # Добавляем в sys.modules для поддержки относительных импортов
                sys.modules[full_module_path] = module

                # Выполняем загрузку
                spec.loader.exec_module(module)

                logger.debug(f"Successfully loaded module via fallback: {full_module_path}")

                return module
            else:
                raise ValueError(f"Invalid module path format: {full_module_path}")

        except Exception as e:
            logger.error(f"Fallback loading failed for {full_module_path}: {e}")
            raise ValueError(f"Failed to load detector module: {e}")

    @staticmethod
    def get_detector_meta(module_path: str, class_name: str) -> Optional[Dict[str, Any]]:
        """
        Возвращает мета-информацию о конкретном детекторе
        """
        cache_key = f"{module_path}.{class_name}"
        meta_info = DetectorFactory._meta_cache.get(cache_key)

        if meta_info:
            return meta_info.to_dict()

        # Если нет в кэше, ищем в файловой системе
        try:
            detectors_root = DetectorFactory.get_detectors_root()
            detector_path = detectors_root / module_path

            if detector_path.exists() and detector_path.is_dir():
                meta_info = DetectorFactory._load_meta_info(detector_path)
                if meta_info:
                    DetectorFactory._meta_cache[cache_key] = meta_info
                    return meta_info.to_dict()
        except Exception as e:
            logger.warning(f"Failed to get meta for {module_path}.{class_name}: {e}")

        return None

    @staticmethod
    def validate_detector_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Расширенная валидация конфигурации детектора с использованием мета-данных
        """
        validation_result = {
            "valid": False,
            "errors": [],
            "warnings": [],
            "meta_info": None
        }

        try:
            required_fields = ["module_path", "class_name"]
            for field in required_fields:
                if field not in config:
                    validation_result["errors"].append(f"Missing required field: {field}")

            if validation_result["errors"]:
                return validation_result

            module_path = config["module_path"]
            class_name = config["class_name"]

            # Получаем мета-информацию
            meta_info = DetectorFactory.get_detector_meta(module_path, class_name)

            if not meta_info:
                validation_result["errors"].append(
                    f"Detector not found: {module_path}.{class_name}"
                )
                return validation_result

            validation_result["meta_info"] = meta_info

            # Валидация параметров
            parameters = config.get("parameters", {})
            required_params = meta_info.get("required_parameters", [])
            optional_params = meta_info.get("optional_parameters", {})

            # Проверка обязательных параметров
            for param in required_params:
                if param not in parameters:
                    validation_result["errors"].append(
                        f"Missing required parameter: {param}"
                    )

            # Проверка неизвестных параметров
            all_valid_params = set(required_params) | set(optional_params.keys())
            for param in parameters.keys():
                if param not in all_valid_params:
                    validation_result["warnings"].append(
                        f"Unknown parameter: {param}"
                    )

            validation_result["valid"] = len(validation_result["errors"]) == 0

        except Exception as e:
            validation_result["errors"].append(f"Validation error: {str(e)}")

        return validation_result

    @staticmethod
    def clear_cache():
        """Очищает все кэши"""
        DetectorFactory._instance_cache.clear()
        DetectorFactory._class_cache.clear()
        DetectorFactory._meta_cache.clear()
        logger.info("All detector caches cleared")

    @staticmethod
    def get_cached_instances() -> List[str]:
        """Возвращает список закэшированных экземпляров (для отладки)"""
        return list(DetectorFactory._instance_cache.keys())

    @staticmethod
    def get_cached_classes() -> List[str]:
        """Возвращает список закэшированных классов (для отладки)"""
        return list(DetectorFactory._class_cache.keys())

    @staticmethod
    def get_cached_meta() -> List[str]:
        """Возвращает список закэшированных мета-данных (для отладки)"""
        return list(DetectorFactory._meta_cache.keys())# Добавим в класс DetectorFactory:

    @staticmethod
    def get_detector_name(module_path: str, class_name: str) -> str:
        """Возвращает имя детектора из meta.json"""
        meta_info = DetectorFactory.get_detector_meta(module_path, class_name)
        return meta_info["name"] if meta_info else "Unknown Detector"

    @staticmethod
    def get_detector_version(module_path: str, class_name: str) -> str:
        """Возвращает версию детектора из meta.json"""
        meta_info = DetectorFactory.get_detector_meta(module_path, class_name)
        return meta_info["version"] if meta_info else "1.0.0"

    @staticmethod
    def get_detector_description(module_path: str, class_name: str) -> str:
        """Возвращает описание детектора из meta.json"""
        meta_info = DetectorFactory.get_detector_meta(module_path, class_name)
        return meta_info["description"] if meta_info else ""