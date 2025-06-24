"""
classify.py

FastAPI route for classifying the intent of a user input.
Implements structured logging, traceability, request validation,
and clean separation of concerns for production readiness.
"""

from uuid import uuid4
from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional

from app.classification.intent_classifier import BERTIntentClassifier
from app.utils.logger_utils import get_logger

# Initialize router and logger
router = APIRouter()
logger = get_logger("ClassifyRoute")

# Initialize classifier once (thread-safe if model is stateless)
classifier = BERTIntentClassifier()

# Request model
class ClassifyRequest(BaseModel):
    text: str = Field(..., description="User input to be classified into an intent.")

# Response model
class ClassifyResponse(BaseModel):
    intent: str
    trace_id: str
    error: Optional[str] = None

@router.post("/classify", summary="Classify user intent", response_model=ClassifyResponse)
async def classify_intent(request: Request, body: ClassifyRequest):
    """
    Classifies the intent of the provided text input using a BERT-based classifier.
    Adds structured logging and returns a trace ID for observability.

    Args:
        request: FastAPI request object for metadata and headers
        body: JSON input with a 'text' field

    Returns:
        ClassifyResponse with intent, trace ID, and optional error
    """
    # Extract or generate trace ID
    trace_id = request.headers.get("X-Trace-ID") or str(uuid4())

    try:
        logger.info(
            "Received intent classification request",
            extra={
                "trace_id": trace_id,
                "method": request.method,
                "path": str(request.url),
                "text": body.text
            }
        )

        predicted_intent = classifier.classify(body.text)

        return ClassifyResponse(
            intent=predicted_intent,
            trace_id=trace_id,
            error=None
        )

    except Exception as e:
        logger.exception("Intent classification failed", extra={"trace_id": trace_id})

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ClassifyResponse(
                intent="unknown",
                trace_id=trace_id,
                error=str(e)
            ).model_dump()
        )
