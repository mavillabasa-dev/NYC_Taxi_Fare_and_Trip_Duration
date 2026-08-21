"""Prediction and model health endpoints."""

from fastapi import APIRouter, HTTPException, Request, status

from app.model.schema import HealthResponse, PredictionRequest, PredictionResponse
from app.model.services import ModelService

router = APIRouter()


def get_model_service(request: Request) -> ModelService:
	return request.app.state.model_service


@router.get("/health")
def health(request: Request) -> HealthResponse:
	service = get_model_service(request)
	return HealthResponse(
		status="ok" if service.is_loaded else "degraded",
		model_loaded=service.is_loaded,
		model_version=service.version,
		detail=service.load_error,
	)


@router.post("/predict")
def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
	service = get_model_service(request)
	if not service.is_loaded:
		raise HTTPException(
			status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
			detail=service.load_error or "Model is not loaded",
		)

	try:
		return service.predict(payload)
	except RuntimeError as exc:
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail=str(exc),
		) from exc
