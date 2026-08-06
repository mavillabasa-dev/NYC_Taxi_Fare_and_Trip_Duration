"""Punto de entrada de la aplicación FastAPI."""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from app.model.router import router
from app.model.services import model_service
from settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
	model_service.load(settings.model_path)
	app.state.model_service = model_service
	yield


app = FastAPI(
	title="NYC Taxi Fare and Trip Duration API",
	version="0.1.0",
	lifespan=lifespan,
)
app.include_router(router)
