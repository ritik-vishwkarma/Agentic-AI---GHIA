"""API routes for doctor feedback and learning"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import logging

from app.services.feedback_learning import get_learning_engine, FeedbackType

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/feedback", tags=["Learning"])


class FeedbackRequest(BaseModel):
    intake_id: int
    doctor_id: str
    feedback_type: str
    original: Dict[str, Any]
    corrected: Dict[str, Any]
    reasoning: Optional[str] = None


@router.post("/submit")
async def submit_feedback(request: FeedbackRequest):
    """Doctor submits correction or confirmation"""
    try:
        feedback_type = FeedbackType(request.feedback_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid type. Use: {[t.value for t in FeedbackType]}")
    
    engine = get_learning_engine()
    feedback = engine.capture_feedback(
        intake_id=request.intake_id,
        doctor_id=request.doctor_id,
        feedback_type=feedback_type,
        original=request.original,
        corrected=request.corrected,
        reasoning=request.reasoning
    )
    
    return {"success": True, "feedback_id": feedback.feedback_id, "severity": feedback.severity_score}


@router.get("/performance")
async def get_performance(days: int = 30):
    """Get system performance metrics"""
    return get_learning_engine().get_performance_metrics(days)


@router.get("/learned-mappings")
async def get_learned_mappings():
    """Get all learned symptom mappings"""
    engine = get_learning_engine()
    return {"symptom_mappings": engine._learned_symptoms, "hints_count": len(engine._diagnosis_hints)}