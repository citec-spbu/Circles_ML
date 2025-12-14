# Интеграция программы

В этой директории описаны способы интеграции нашей программы.

## 1. Использование UI 

**Описание:** Запуск веб-интерфейса через полную версию приложения.

**Шаги:**
1. Установите зависимости:
```bash
pip install -r requirements.txt
```

2. Запустите сервер:
```bash
python -m app.main
```

или

```bash
uvicorn app.main:app --port 8000 --host 0.0.0.0
```

3. Откройте в браузере: `http://localhost:8000`

## 2. Использование API

**Описание:** API для интеграции с другими системами.

**Шаги:**
1. Установите зависимости:
```bash
pip install -r requirements-slim.txt
```

2. Запустите сервер:
```bash
python -m app-slim.main
```

или

```bash
uvicorn app-slim.main:app --port 8000 --host 0.0.0.0
```

**Доступные эндпоинты:**
- `POST /api/detect` - детекция объектов

**Пример запроса:**
Изображение отправленное как `form-data` с именем file
```bash
curl -X POST http://localhost:8000/api/detect \
  -F "file=@/путь/к/изображению.jpg" \
  -H "Content-Type: multipart/form-data"
```

**Пример ответа:**
```json
[
    {
        "center_x": 448.5064392089844,
        "center_y": 487.8146667480469,
        "normal_x": 0.0,
        "normal_y": 0.0,
        "normal_z": 1.0,
        "radius": 28.476208228154373,
        "confidence": 0.9644081519750358,
        "metadata": {}
    }
]
```


## 3. Прямая интеграция Python кода

**Описание:** Встраивание детектора напрямую в ваш Python-код.

**Шаги:**
1. Из репозитория возьмите пакет `detectors`
2. Оставьте только:
   - `__init__.py`
   - `contour_ellipse_detector`

**Структура:**
```
detectors/
├── __init__.py
└── contour_ellipse_detector
```

**Пример использования:**
```python
from detectors.contour_ellipse_detector import ContourEllipseDetector

# Инициализация детектора
detector = ContourEllipseDetector()

# Использование
results = detector.detect(image)
```


## 4. Сборка в Docker

**Описание:** Создание Docker-образа для развертывания.

---

**Примечание:** Выберите способ интеграции, наиболее подходящий для вашего сценария использования.
