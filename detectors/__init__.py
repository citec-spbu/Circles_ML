from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import numpy as np


class DetectionResult:
    """Модель результата обнаружения центра круга"""

    def __init__(self, center_x: float, center_y: float, normal_x: float,
                 normal_y: float, normal_z: float, radius: Optional[float] = None,
                 confidence: float = 1.0, metadata: Dict[str, Any] = None):
        self.center_x = center_x
        self.center_y = center_y
        self.normal_x = normal_x
        self.normal_y = normal_y
        self.normal_z = normal_z
        self.radius = radius
        self.confidence = confidence
        self.metadata = metadata or {}

    def dict(self):
        """Конвертирует в словарь для JSON"""
        return {
            "center_x": self.center_x,
            "center_y": self.center_y,
            "normal_x": self.normal_x,
            "normal_y": self.normal_y,
            "normal_z": self.normal_z,
            "radius": self.radius,
            "confidence": self.confidence,
            "metadata": self.metadata
        }


class BaseDetector(ABC):
    """Абстрактный базовый класс для ВСЕХ детекторов"""

    def __init__(self, **kwargs):
        self.config = kwargs

    def update_config(self, **kwargs):
        """
        Обновляет конфигурацию детектора на лету
        Для поддержки кэширования экземпляров
        """
        self.config.update(kwargs)
        # Дополнительная логика при обновлении конфигурации
        self._on_config_update()

    def _on_config_update(self):
        """
        Вызывается после обновления конфигурации
        Можно переопределить в дочерних классах для дополнительной логики
        """
        pass

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        """Основной метод обнаружения центров"""
        pass

    def cleanup(self):
        """
        Очистка ресурсов детектора
        Вызывается при уничтожении некэшированного экземпляра
        """
        pass