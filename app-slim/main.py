from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
import argparse
import numpy as np
import cv2
from detectors.contour_ellipse_detector import ContourEllipseDetector

app = FastAPI()
detector = ContourEllipseDetector()


@app.post("/api/detect")
async def detect(file: UploadFile = File(..., description="Изображение для анализа")):
    # Валидация файла
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="Файл должен быть изображением"
        )

    # Чтение и декодирование изображения
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Не удалось декодировать изображение"
        )

    result = detector.detect(image)
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Запуск FastAPI приложения")
    parser.add_argument('--host', default='0.0.0.0', help='Хост для запуска сервера')
    parser.add_argument('--port', type=int, default=8000, help='Порт для запуска сервера')

    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
