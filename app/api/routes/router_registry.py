
from fastapi import APIRouter

from app.api.routes.health import router as health_router
from app.api.routes.ask import router as ask_router
from app.api.routes.classify import router as classify_router

router = APIRouter()

# Register each route
router.include_router(health_router)
router.include_router(ask_router)
router.include_router(classify_router)

def get_all_routers() -> list[APIRouter]:
    return [
        health_router,
        ask_router,
        classify_router,
    ]

