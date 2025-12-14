FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-slim.txt .

RUN pip install --no-cache-dir -r requirements-slim.txt

COPY app-slim app-slim
COPY detectors/contour_ellipse_detector detectors/contour_ellipse_detector
COPY detectors/__init__.py detectors/__init__.py

EXPOSE 8000

CMD ["uvicorn", "app-slim.main:app", "--host", "0.0.0.0", "--port", "8000"]