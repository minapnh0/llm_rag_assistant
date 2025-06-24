from uuid import uuid4
from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List

from app.orchestrator.intent_router import LLMOrchestrator
from app.utils.logger_utils import get_logger

router = APIRouter()
logger = get_logger("AskRoute")

orchestrator = LLMOrchestrator()

class AskRequest(BaseModel):
    question: str = Field(..., description="The question to be answered.")

class AskResponse(BaseModel):
    response: Optional[str]
    intent: str
    source_docs: Optional[List[str]] = None
    trace_id: str
    error: Optional[str] = None

@router.post("/", response_model=AskResponse)
async def ask(request: Request, body: AskRequest):
    trace_id = request.headers.get("X-Trace-ID") or str(uuid4())

    try:
        result = orchestrator.handle_query(body.question, trace_id=trace_id)

        return AskResponse(
            response=result.get("response"),
            intent=result.get("intent", "unknown"),
            source_docs=result.get("source_docs"),
            trace_id=trace_id,
            error=result.get("error")
        )

    except Exception as e:
        logger.exception("LLM query failed", extra={"trace_id": trace_id})
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=AskResponse(
                response=None,
                intent="error",
                source_docs=None,
                trace_id=trace_id,
                error=str(e)
            ).model_dump()
        )
