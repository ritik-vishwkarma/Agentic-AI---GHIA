"""API routes for patient intake - Enhanced version with clinical decision support"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid
import logging
import os
import tempfile

from app.config import get_settings
from app.agents.base_agent import AgentContext
from app.agents.clinical_decision import get_clinical_decision_agent, UrgencyLevel
from app.agents.orchestrator import run_agent_pipeline
from app.services.security import (
    get_consent_manager,
    get_audit_logger,
    get_encryption,
    AuditLogger,
    DataClassification,
)
from app.services.asr import transcribe_audio, get_supported_languages
from app.db.database import save_intake_record

logger = logging.getLogger(__name__)
router = APIRouter(tags=["intake"])


class PatientIntakeRequest(BaseModel):
    """Request model for patient intake"""

    transcript: Optional[str] = None
    symptoms: List[str] = Field(default_factory=list)
    patient_id: Optional[str] = None
    age: Optional[str] = None
    gender: Optional[str] = None
    language: str = "hi"
    duration: Optional[str] = None
    chief_complaint: Optional[str] = None
    known_conditions: Optional[str] = None
    medications: Optional[str] = None
    is_pregnant: bool = False
    consent_given: bool = False
    phc_id: Optional[str] = None


class IntakeResponse(BaseModel):
    """Response model for patient intake"""

    intake_id: str
    session_id: str
    urgency: Dict[str, Any]
    symptoms: Dict[str, Any]
    differential_diagnoses: List[Dict[str, Any]]
    treatment_plan: Dict[str, Any]
    flags: List[str]
    follow_up_questions: List[str] = []
    summary_hindi: Optional[str] = None
    summary_english: Optional[str] = None
    requires_review: bool = False
    created_at: str


class SimpleIntakeResponse(BaseModel):
    """Simplified response compatible with existing frontend"""

    id: int
    session_id: str
    risk_level: str
    summary_english: str
    summary_hindi: str
    symptoms: List[Dict[str, Any]]
    recommended_action: str
    follow_up_questions: List[str]
    created_at: str


@router.get("/languages")
def get_languages():
    """Get list of supported languages"""
    return {"languages": get_supported_languages(), "default": "hi"}


@router.post("/process", response_model=IntakeResponse)
async def process_intake(
    request: PatientIntakeRequest, background_tasks: BackgroundTasks
):
    """Process patient intake with clinical decision support."""
    settings = get_settings()
    audit = get_audit_logger()
    consent_manager = get_consent_manager()

    intake_id = str(uuid.uuid4())
    session_id = f"session-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{intake_id[:8]}"

    # Check consent if patient_id provided
    if request.patient_id and not request.consent_given:
        has_consent, _ = consent_manager.check_consent(request.patient_id, "treatment")
        if not has_consent:
            raise HTTPException(
                status_code=403,
                detail="Patient consent required for processing health data",
            )

    context = AgentContext(
        session_id=session_id,
        patient_id=request.patient_id,
        phc_id=request.phc_id or settings.default_phc_id,
        language=request.language,
    )

    patient_data = {
        "age": request.age,
        "gender": request.gender,
        "duration": request.duration,
        "chief_complaint": request.chief_complaint or request.transcript,
        "known_conditions": request.known_conditions,
        "medications": request.medications,
        "is_pregnant": request.is_pregnant,
    }

    symptoms = request.symptoms
    if not symptoms and request.transcript:
        symptoms = [s.strip() for s in request.transcript.split(",") if s.strip()]

    try:
        clinical_agent = get_clinical_decision_agent()
        result = await clinical_agent(
            context=context, symptoms=symptoms, patient_data=patient_data
        )

        if not result.success:
            raise HTTPException(
                status_code=500, detail=f"Clinical analysis failed: {result.error}"
            )

        data = result.data

        # Encrypt for storage
        encryption = get_encryption()
        encrypted_data, metadata = encryption.encrypt_phi(
            {"patient_data": patient_data, "symptoms": symptoms, "result": data},
            classification=DataClassification.CONFIDENTIAL,
        )

        # Audit
        audit.log(
            user_id=context.user_id,
            action=AuditLogger.AuditAction.CREATE,
            resource_type="intake",
            resource_id=intake_id,
            patient_id=request.patient_id,
            phc_id=context.phc_id,
            details={
                "symptoms_count": len(symptoms),
                "urgency": data.get("urgency", {}).get("level"),
                "has_red_flags": len(
                    data.get("symptom_analysis", {}).get("red_flags", [])
                )
                > 0,
            },
        )

        # Background save
        background_tasks.add_task(
            _save_intake_background,
            intake_id=intake_id,
            encrypted_data=encrypted_data,
            metadata=metadata,
            result=data,
            patient_data=patient_data,
        )

        return IntakeResponse(
            intake_id=intake_id,
            session_id=session_id,
            urgency=data.get("urgency", {}),
            symptoms=data.get("symptom_analysis", {}),
            differential_diagnoses=data.get("differential_diagnoses", []),
            treatment_plan=data.get("treatment_plan", {}),
            flags=data.get("flags", []),
            follow_up_questions=_generate_follow_up_questions(data),
            summary_hindi=_generate_summary_hindi(data),
            summary_english=_generate_summary_english(data),
            requires_review=data.get("urgency", {}).get("level")
            in ["emergency", "urgent"],
            created_at=datetime.utcnow().isoformat(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Intake processing error: {e}")

        audit.log(
            user_id=context.user_id,
            action=AuditLogger.AuditAction.CREATE,
            resource_type="intake",
            resource_id=intake_id,
            patient_id=request.patient_id,
            success=False,
            error_message=str(e),
        )

        raise HTTPException(
            status_code=500, detail=f"Failed to process intake: {str(e)}"
        )


@router.post("/audio")
async def process_audio_intake(
    audio: UploadFile = File(...),
    patient_id: Optional[str] = Form(None),
    age: Optional[str] = Form(None),
    gender: Optional[str] = Form(None),
    language: str = Form("hi"),
    phc_id: Optional[str] = Form(None),
    consent_given: bool = Form(False),
    background_tasks: BackgroundTasks = None,
):
    """Process audio recording for patient intake."""
    settings = get_settings()

    # Read audio bytes
    try:
        audio_bytes = await audio.read()
        logger.info(
            f"Received audio: {len(audio_bytes)} bytes, content_type: {audio.content_type}"
        )
    except Exception as e:
        logger.error(f"Failed to read audio: {e}")
        raise HTTPException(status_code=400, detail="Failed to read audio file")

    if len(audio_bytes) < 1000:
        raise HTTPException(status_code=400, detail="Audio file too small or empty")

    # Validate size
    max_size = settings.max_upload_size_mb * 1024 * 1024
    if len(audio_bytes) > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"Audio file too large. Max size: {settings.max_upload_size_mb}MB",
        )

    try:
        # Transcribe
        logger.info(
            f"Transcribing audio: {len(audio_bytes)} bytes, language: {language}"
        )
        transcript, detected_lang = await transcribe_audio(audio_bytes, language)

        if not transcript:
            logger.warning("Transcription returned empty result")
            raise HTTPException(
                status_code=400,
                detail="Could not transcribe audio. Please speak clearly or use text input.",
            )

        logger.info(f"Transcription result: {transcript[:100]}...")

        # Use existing orchestrator pipeline for full processing
        result = await run_agent_pipeline(
            raw_transcript=transcript,
            language=detected_lang or language,
            patient_id=patient_id,
            phc_id=phc_id or settings.default_phc_id,
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Audio intake processing failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Audio processing failed: {str(e)}"
        )


@router.post("/text")
async def process_text_intake(
    text: str = Form(...),
    language: Optional[str] = Form("hi"),
    patient_id: Optional[str] = Form(None),
    phc_id: Optional[str] = Form(None),
):
    """Process text intake using full agent pipeline."""
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    settings = get_settings()

    try:
        result = await run_agent_pipeline(
            raw_transcript=text.strip(),
            language=language,
            patient_id=patient_id,
            phc_id=phc_id or settings.default_phc_id,
        )
        return result

    except Exception as e:
        logger.exception(f"Text intake processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/demo")
async def demo_intake():
    """Demo endpoint with sample Hindi input."""
    demo_text = "मुझे तीन दिन से बुखार है, सिर में दर्द और शरीर में कमजोरी है। खाँसी भी है।"

    try:
        result = await run_agent_pipeline(raw_transcript=demo_text, language="hi")

        return {
            "message": "Demo intake processed successfully",
            "demo_input": demo_text,
            "result": result,
        }

    except Exception as e:
        logger.exception(f"Demo failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Helper functions
def _generate_follow_up_questions(data: Dict[str, Any]) -> List[str]:
    """Generate follow-up questions based on analysis."""
    questions = []

    symptom_analysis = data.get("symptom_analysis", {})
    associated = symptom_analysis.get("associated_symptoms", [])

    for symptom in associated[:3]:
        questions.append(f"क्या आपको {symptom} भी है? (Do you also have {symptom}?)")

    if not symptom_analysis.get("duration"):
        questions.append("यह समस्या कब से है? (How long have you had this problem?)")

    if data.get("urgency", {}).get("level") in ["moderate", "urgent"]:
        questions.append("क्या पहले भी ऐसा हुआ है? (Has this happened before?)")

    return questions[:5]


def _generate_summary_hindi(data: Dict[str, Any]) -> str:
    """Generate Hindi summary."""
    urgency = data.get("urgency", {}).get("level", "routine")
    symptoms = data.get("symptom_analysis", {}).get("symptoms", [])
    treatment = data.get("treatment_plan", {})

    urgency_hindi = {
        "emergency": "आपातकालीन",
        "urgent": "तत्काल",
        "moderate": "मध्यम",
        "routine": "सामान्य",
    }.get(urgency, "सामान्य")

    symptom_names = [
        s.get("name", str(s)) if isinstance(s, dict) else str(s) for s in symptoms[:3]
    ]

    summary = f"स्थिति: {urgency_hindi}। "
    if symptom_names:
        summary += f"लक्षण: {', '.join(symptom_names)}। "

    if treatment.get("referral_needed"):
        summary += "रेफरल आवश्यक। "
    elif treatment.get("follow_up_days"):
        summary += f"{treatment['follow_up_days']} दिन में फॉलो-अप करें।"

    return summary


def _generate_summary_english(data: Dict[str, Any]) -> str:
    """Generate English summary."""
    urgency = data.get("urgency", {}).get("level", "routine")
    symptoms = data.get("symptom_analysis", {}).get("symptoms", [])
    differentials = data.get("differential_diagnoses", [])
    treatment = data.get("treatment_plan", {})

    summary = f"Urgency: {urgency.upper()}. "

    symptom_names = [
        s.get("name", str(s)) if isinstance(s, dict) else str(s) for s in symptoms[:3]
    ]
    if symptom_names:
        summary += f"Symptoms: {', '.join(symptom_names)}. "

    if differentials:
        top = differentials[0]
        summary += f"Likely: {top.get('condition', 'Unknown')} ({int(top.get('confidence', 0) * 100)}% confidence). "

    if treatment.get("referral_needed"):
        summary += (
            f"REFERRAL REQUIRED: {treatment.get('referral_type', 'specialist')}. "
        )
    else:
        summary += f"Follow-up in {treatment.get('follow_up_days', 7)} days."

    return summary


async def _save_intake_background(
    intake_id: str,
    encrypted_data: str,
    metadata: Dict,
    result: Dict,
    patient_data: Dict,
):
    """Background task to save intake record."""
    try:
        intake_data = {
            "chief_complaint": patient_data.get("chief_complaint"),
            "symptoms": result.get("symptom_analysis", {}).get("symptoms", []),
            "duration": patient_data.get("duration"),
            "severity": result.get("symptom_analysis", {}).get("severity"),
            "associated_symptoms": result.get("symptom_analysis", {}).get(
                "associated_symptoms", []
            ),
        }

        urgency = result.get("urgency", {}).get("level", "routine")

        save_intake_record(
            raw_transcript=patient_data.get("chief_complaint", ""),
            language_detected="hi",
            intake_data=intake_data,
            risk_level=urgency,
            agent_decisions=[{"agent": "clinical_decision", "result": "success"}],
            summary_english=_generate_summary_english(result),
            summary_hindi=_generate_summary_hindi(result),
            follow_up_questions=_generate_follow_up_questions(result),
            recommended_action=result.get("treatment_plan", {}).get(
                "referral_type", "follow_up"
            ),
        )

        logger.info(f"Saved intake record: {intake_id}")

    except Exception as e:
        logger.error(f"Failed to save intake record {intake_id}: {e}")
