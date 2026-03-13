from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.chat import router as chat_router
from app.api.rag import router as rag_router
from app.api.tools import router as tools_router
from app.core.config import get_settings
from app.core.exceptions import AppBaseException


def register_routers(app: FastAPI) -> None:
    router = APIRouter()
    settings = get_settings()

    @router.get("/", summary="Root")
    async def root() -> dict[str, str]:
        return {
            "app": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "status": "running",
        }

    @router.get("/health", summary="Health Check")
    async def health_check() -> dict[str, str]:
        return {
            "status": "ok",
            "environment": settings.APP_ENV,
        }

    app.include_router(router)
    app.include_router(chat_router)
    app.include_router(rag_router)
    app.include_router(tools_router)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppBaseException)
    async def handle_app_exception(_: Request, exc: AppBaseException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.error_code,
                    "message": exc.message,
                    "details": exc.details,
                }
            },
        )


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        debug=settings.APP_DEBUG,
    )
    register_exception_handlers(app)
    register_routers(app)
    return app


app = create_app()
