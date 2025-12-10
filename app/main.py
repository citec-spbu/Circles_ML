from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.api.endpoints import router as api_router
from app.core.config import settings


def create_application() -> FastAPI:
    """Создание и настройка FastAPI приложения"""

    application = FastAPI(
        title=settings.PROJECT_NAME,
        description=settings.DESCRIPTION,
        version=settings.VERSION,
        debug=settings.DEBUG
    )

    # Настройка CORS
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_HOSTS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Подключение статических файлов и шаблонов
    application.mount("/static", StaticFiles(directory="app/static"), name="static")

    # Подключение API роутеров
    application.include_router(api_router, prefix="/api")

    # Глобальные обработчики
    @application.get("/")
    async def root(request: Request):
        """Главная страница с веб-интерфейсом"""
        templates = Jinja2Templates(directory="app/templates")
        return templates.TemplateResponse("index.html", {"request": request})

    @application.get("/health")
    async def health_check():
        """Эндпоинт для проверки работоспособности"""
        return {
            "status": "healthy",
            "project": settings.PROJECT_NAME,
            "version": settings.VERSION
        }

    return application


app = create_application()

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )