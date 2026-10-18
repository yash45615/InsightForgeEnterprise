from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from insightforge.agents.retrieval_tool import KnowledgeTool
from insightforge.evaluation.safety import inspect_user_input

router = APIRouter(prefix="/knowledge", tags=["Knowledge"])
knowledge = KnowledgeTool()


class QuestionRequest(BaseModel):
    question: str = Field(max_length=500)
    top_k: int = Field(default=3, ge=1, le=5)


@router.post("/ask")
def ask_question(request: QuestionRequest) -> dict[str, Any]:
    """Answer a question using retrieved evidence and source citations."""
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question must not be empty.",
        )

    safety = inspect_user_input(question)
    if not safety["allowed"]:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Question rejected by the input safety check.",
                "flags": safety["flags"],
            },
        )

    try:
        return knowledge.search(question, top_k=request.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
