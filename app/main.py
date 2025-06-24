import os
from app.config.env_loader import load_env
load_env()  

from fastapi import FastAPI
from app.api.routes.router_registry import router

app = FastAPI()
app.include_router(router)