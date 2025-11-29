"""
GHIA Multi-Agent Orchestrator using LangGraph

Enhanced with:
- True agent autonomy with decision routing
- Conversation memory
- Agent reflection and self-correction
- Confidence scoring
- Parallel agent execution where possible
"""

from typing import TypedDict, List, Optional, Dict, Any, Annotated, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import json
import logging
import asyncio
from datetime import datetime
from enum import Enum

from app.config import get_settings
from app.services.llm import get_llm_service
from app.services.medical_kb import (
    normalize_hinglish_text,
    check_red_flags,
    extract_duration,
    get_symptom_severity_indicators,
)
from app.services.outbreak_detection import (
    get_outbreak_analyzer,
    enrich_state_with_outbreak_context,
)

logger = logging.getLogger(__name__)


class AgentConfidence(str, Enum):
    HIGH = "high"  # > 0.8
    MEDIUM = "medium"  # 0.5 - 0.8
    LOW = "low"  # < 0.5


class RiskLevel(str, Enum):
    CRITICAL = "critical"  # Life-threatening, immediate action
    URGENT = "urgent"  # Needs attention within hours
    MODERATE = "moderate"  # Needs attention within 24-48 hours
    ROUTINE = "routine"  # Regular appointment


class GHIAState(TypedDict):
    """Enhanced state with conversation history and agent metadata"""

    # Session info
    session_id: str
    patient_id: Optional[str]
    phc_id: str
    created_at: str

    # Input
    raw_transcript: str
    language_detected: str
    conversation_history: List[Dict[str, str]]  # Multi-turn support

    # Normalized data
    normalized_text: str
    mapped_phrases: List[Dict]

    # Medical extraction results
    chief_complaint: Optional[str]
    symptoms: List[Dict]
    duration: Optional[str]
    severity: Optional[str]
    associated_symptoms: List[str]
    vital_signs: Optional[Dict]  # If provided
    medical_history: Optional[str]
    current_medications: List[str]
    allergies: List[str]

    # Demographics (optional, for better triage)
    age_group: Optional[str]
    gender: Optional[str]
    is_pregnant: Optional[bool]

    # Agent decisions and trace
    agent_decisions: List[Dict]
    agent_reflections: List[Dict]  # Self-correction notes

    # Triage result
    risk_level: RiskLevel
    risk_confidence: float
    red_flags: List[str]
    differential_diagnosis: List[Dict]

    # Clinical decision support
    icd11_codes: List[Dict]
    treatment_suggestions: List[Dict]
    referral_recommendation: Optional[Dict]

    # Follow-up
    needs_followup: bool
    followup_questions: List[str]
    missing_information: List[str]

    # Final outputs
    summary_english: str
    summary_hindi: str
    recommended_action: str
    urgency_explanation: str

    # Processing flags
    extraction_complete: bool
    extraction_confidence: float
    triage_complete: bool
    requires_human_review: bool

    # Outbreak awareness (ADD THESE)
    community_health_alerts: List[Dict]
    outbreak_aware_recommendations: List[str]

    # Error handling
    errors: List[Dict]
    warnings: List[str]


def create_initial_state(
    raw_transcript: str,
    language: str = "hi",
    session_id: str = None,
    patient_id: str = None,
    phc_id: str = None,
    conversation_history: List[Dict] = None,
) -> GHIAState:
    """Create initial state for the pipeline"""
    settings = get_settings()
    normalized_text, mapped_phrases = normalize_hinglish_text(raw_transcript)

    return GHIAState(
        session_id=session_id
        or f"session-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
        patient_id=patient_id,
        phc_id=phc_id or settings.default_phc_id,
        created_at=datetime.utcnow().isoformat(),
        raw_transcript=raw_transcript,
        language_detected=language,
        conversation_history=conversation_history or [],
        normalized_text=normalized_text,
        mapped_phrases=mapped_phrases,
        chief_complaint=None,
        symptoms=[],
        duration=None,
        severity=None,
        associated_symptoms=[],
        vital_signs=None,
        medical_history=None,
        current_medications=[],
        allergies=[],
        age_group=None,
        gender=None,
        is_pregnant=None,
        agent_decisions=[],
        agent_reflections=[],
        risk_level=RiskLevel.ROUTINE,
        risk_confidence=0.0,
        red_flags=[],
        differential_diagnosis=[],
        icd11_codes=[],
        treatment_suggestions=[],
        referral_recommendation=None,
        needs_followup=False,
        followup_questions=[],
        missing_information=[],
        summary_english="",
        summary_hindi="",
        recommended_action="",
        urgency_explanation="",
        extraction_complete=False,
        extraction_confidence=0.0,
        triage_complete=False,
        requires_human_review=False,
        community_health_alerts=[],
        outbreak_aware_recommendations=[],
        errors=[],
        warnings=[],
    )


# ============= AGENT 1: Medical Extractor =============
async def medical_extractor_agent(state: GHIAState) -> GHIAState:
    """
    Extracts structured medical information with confidence scoring.
    Uses few-shot prompting for better accuracy.
    """
    logger.info("🔍 Medical Extractor Agent running...")

    llm_service = get_llm_service()
    settings = get_settings()

    # Build context from conversation history
    history_context = ""
    if state["conversation_history"]:
        history_context = "\n".join(
            [
                f"{turn['role']}: {turn['content']}"
                for turn in state["conversation_history"][-5:]  # Last 5 turns
            ]
        )

    prompt = f"""You are an expert medical intake assistant specializing in rural Indian healthcare.
Extract structured medical information from the patient statement.

IMPORTANT GUIDELINES:
1. Preserve medical terms in both Hindi and English
2. Be conservative with severity - only mark severe if clearly indicated
3. Extract specific durations (e.g., "3 din se" → "3 days")
4. Note any red flag symptoms separately
5. Provide confidence scores for your extractions

{"CONVERSATION HISTORY:" + chr(10) + history_context + chr(10) if history_context else ""}

CURRENT PATIENT STATEMENT: "{state["normalized_text"]}"
ORIGINAL (may contain Hinglish): "{state["raw_transcript"]}"
MAPPED MEDICAL PHRASES: {json.dumps(state["mapped_phrases"], ensure_ascii=False)}

FEW-SHOT EXAMPLES:
Input: "मुझे तीन दिन से बुखार है और सिर में दर्द भी है"
Output: {{"chief_complaint": "Fever with headache for 3 days", "symptoms": [{{"symptom": "fever", "hindi": "बुखार", "severity": "moderate", "duration": "3 days"}}, {{"symptom": "headache", "hindi": "सिर दर्द", "severity": "moderate"}}], "duration": "3 days", "severity_overall": "moderate", "associated_symptoms": [], "confidence": 0.9}}

Input: "pet mein bahut dard hai, ulti bhi ho rahi hai kal se"
Output: {{"chief_complaint": "Severe abdominal pain with vomiting since yesterday", "symptoms": [{{"symptom": "abdominal pain", "hindi": "पेट दर्द", "severity": "severe", "body_part": "abdomen"}}, {{"symptom": "vomiting", "hindi": "उल्टी", "severity": "moderate"}}], "duration": "1 day", "severity_overall": "severe", "associated_symptoms": ["nausea"], "confidence": 0.85}}

NOW EXTRACT FROM THE CURRENT STATEMENT. Return ONLY valid JSON:
{{
    "chief_complaint": "main issue in 5-10 words (English)",
    "chief_complaint_hindi": "मुख्य समस्या हिंदी में",
    "symptoms": [
        {{
            "symptom": "symptom name",
            "hindi": "हिंदी में",
            "severity": "mild/moderate/severe",
            "duration": "if mentioned",
            "body_part": "if mentioned"
        }}
    ],
    "duration": "overall duration",
    "severity_overall": "mild/moderate/severe",
    "associated_symptoms": ["list of secondary symptoms"],
    "vital_concerns": ["any concerning vital sign mentions"],
    "confidence": 0.0-1.0
}}"""

    try:
        response = await llm_service.invoke(
            messages=[HumanMessage(content=prompt)],
            system_prompt="You are a medical NLP system. Extract structured data from patient statements. Always respond with valid JSON only.",
        )

        # Parse response
        response_text = response.strip()
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
        response_text = response_text.strip()

        extracted = json.loads(response_text)

        # Update state
        state["chief_complaint"] = extracted.get("chief_complaint")
        state["symptoms"] = extracted.get("symptoms", [])
        state["duration"] = extracted.get("duration") or extract_duration(
            state["normalized_text"]
        )
        state["severity"] = extracted.get("severity_overall")
        state["associated_symptoms"] = extracted.get("associated_symptoms", [])
        state["extraction_complete"] = True
        state["extraction_confidence"] = extracted.get("confidence", 0.7)

        # Log decision
        state["agent_decisions"].append(
            {
                "agent_name": "Medical Extractor",
                "timestamp": datetime.utcnow().isoformat(),
                "action": "extracted_symptoms",
                "details": {
                    "chief_complaint": state["chief_complaint"],
                    "symptom_count": len(state["symptoms"]),
                    "confidence": state["extraction_confidence"],
                },
                "reasoning": f"Identified chief complaint: {state['chief_complaint']}",
                "confidence": state["extraction_confidence"],
            }
        )

        logger.info(
            f"✅ Extracted: {state['chief_complaint']} (confidence: {state['extraction_confidence']:.2f})"
        )

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse extraction response: {e}")
        state["errors"].append(
            {
                "agent": "Medical Extractor",
                "error": "JSON parsing failed",
                "details": str(e),
            }
        )
        state["extraction_complete"] = True
        state["extraction_confidence"] = 0.3
        state["requires_human_review"] = True

    except Exception as e:
        logger.error(f"Medical extractor error: {e}")
        state["errors"].append({"agent": "Medical Extractor", "error": str(e)})
        state["chief_complaint"] = "Unable to extract - manual review needed"
        state["extraction_complete"] = True
        state["requires_human_review"] = True

    return state


# ============= AGENT 2: Quality Checker =============
async def quality_checker_agent(state: GHIAState) -> GHIAState:
    """
    Reviews extraction quality and identifies missing information.
    Implements agent self-reflection pattern.
    """
    logger.info("🔎 Quality Checker Agent running...")

    missing_info = []
    warnings = []

    # Check for required information
    if not state.get("chief_complaint"):
        missing_info.append("chief complaint")
    if not state.get("duration"):
        missing_info.append("duration of symptoms")
    if not state.get("severity"):
        warnings.append("Severity not explicitly stated")

    # Check symptom quality
    for symptom in state.get("symptoms", []):
        if not symptom.get("severity"):
            warnings.append(f"Severity missing for symptom: {symptom.get('symptom')}")

    # Check if extraction confidence is low
    if state.get("extraction_confidence", 0) < 0.6:
        warnings.append("Low extraction confidence - consider follow-up questions")
        state["requires_human_review"] = True

    state["missing_information"] = missing_info
    state["warnings"].extend(warnings)
    state["needs_followup"] = len(missing_info) > 0

    # Agent reflection
    if missing_info or len(warnings) > 2:
        state["agent_reflections"].append(
            {
                "agent": "Quality Checker",
                "timestamp": datetime.utcnow().isoformat(),
                "reflection": f"Extraction quality concerns: missing {missing_info}, warnings: {len(warnings)}",
                "recommendation": "Generate follow-up questions"
                if missing_info
                else "Proceed with caution",
            }
        )

    state["agent_decisions"].append(
        {
            "agent_name": "Quality Checker",
            "timestamp": datetime.utcnow().isoformat(),
            "action": "quality_assessment",
            "details": {
                "missing_info": missing_info,
                "warning_count": len(warnings),
                "needs_followup": state["needs_followup"],
            },
            "reasoning": f"Found {len(missing_info)} missing fields, {len(warnings)} warnings",
            "confidence": 0.9,
        }
    )

    logger.info(
        f"✅ Quality check complete: {len(missing_info)} missing, {len(warnings)} warnings"
    )

    return state


# ============= AGENT 3: Interrogator =============
async def interrogator_agent(state: GHIAState) -> GHIAState:
    """
    Generates contextual follow-up questions based on missing information.
    Supports multi-turn conversation.
    """
    logger.info("❓ Interrogator Agent running...")

    if not state.get("needs_followup") and not state.get("missing_information"):
        state["agent_decisions"].append(
            {
                "agent_name": "Interrogator",
                "timestamp": datetime.utcnow().isoformat(),
                "action": "skipped",
                "reasoning": "No follow-up needed - sufficient information available",
                "confidence": 0.95,
            }
        )
        return state

    llm_service = get_llm_service()
    settings = get_settings()

    missing = state.get("missing_information", [])
    symptoms = state.get("symptoms", [])

    prompt = f"""Based on this patient intake, generate follow-up questions.

Patient's Chief Complaint: "{state.get("chief_complaint", "Not yet identified")}"
Current Symptoms: {json.dumps(symptoms, ensure_ascii=False)}
Duration: {state.get("duration", "Not specified")}

MISSING INFORMATION: {", ".join(missing) if missing else "None"}

GUIDELINES:
1. Generate questions in simple Hindi (Devanagari script)
2. Also provide English translation
3. Prioritize clinically important questions
4. Ask about red flag symptoms if not mentioned
5. Keep questions simple for rural patients

Red flags to check if not mentioned:
- Chest pain, difficulty breathing
- High fever, stiff neck
- Sudden severe headache
- Blood in stool/urine
- Pregnancy-related concerns

Return JSON:
{{
    "questions": [
        {{
            "hindi": "हिंदी में सवाल?",
            "english": "Question in English?",
            "priority": "high/medium/low",
            "targets": "what information this question aims to collect"
        }}
    ],
    "max_questions": 3
}}"""

    try:
        response = await llm_service.invoke(
            messages=[HumanMessage(content=prompt)],
            system_prompt="You are a helpful medical assistant generating follow-up questions in Hindi.",
        )

        response_text = response.strip()
        if "```" in response_text:
            response_text = response_text.split("```")[1].replace("json", "").strip()

        result = json.loads(response_text)
        questions = result.get("questions", [])

        # Extract Hindi questions for patient-facing output
        state["followup_questions"] = [q["hindi"] for q in questions[:3]]

        state["agent_decisions"].append(
            {
                "agent_name": "Interrogator",
                "timestamp": datetime.utcnow().isoformat(),
                "action": "generated_questions",
                "details": {
                    "question_count": len(state["followup_questions"]),
                    "missing_info_addressed": missing,
                },
                "reasoning": f"Generated {len(state['followup_questions'])} follow-up questions for missing: {missing}",
                "confidence": 0.85,
            }
        )

        logger.info(
            f"✅ Generated {len(state['followup_questions'])} follow-up questions"
        )

    except Exception as e:
        logger.error(f"Interrogator agent error: {e}")
        # Fallback questions
        state["followup_questions"] = [
            "यह तकलीफ़ कब से है? (How long have you had this problem?)",
            "क्या दर्द बढ़ रहा है? (Is the pain increasing?)",
            "क्या कोई दवाई ले रहे हैं? (Are you taking any medicine?)",
        ]

    return state


async def outbreak_awareness_agent(state: GHIAState) -> GHIAState:
    """Enriches state with community outbreak context."""
    logger.info("🌍 Outbreak Awareness Agent running...")

    analyzer = get_outbreak_analyzer()

    # Check for new outbreaks
    alerts = analyzer.analyze_recent_patterns(phc_id=state.get("phc_id"), hours=24)

    # Enrich state
    state = enrich_state_with_outbreak_context(state, analyzer)

    return state


# ============= AGENT 4: Triage Agent =============
async def triage_agent(state: GHIAState) -> GHIAState:
    """
    Determines urgency level with confidence scoring.
    Uses rule-based + LLM hybrid approach for safety.
    """
    logger.info("🚨 Triage Agent running...")

    # First, check rule-based red flags (safety-critical)
    all_symptoms = [state.get("chief_complaint", "")]
    all_symptoms.extend(state.get("associated_symptoms", []))
    for s in state.get("symptoms", []):
        all_symptoms.append(s.get("symptom", ""))
        all_symptoms.append(s.get("hindi", ""))

    red_flags = check_red_flags(all_symptoms)
    state["red_flags"] = red_flags

    # Critical red flags override LLM decision
    critical_flags = [
        "chest pain",
        "difficulty breathing",
        "unconscious",
        "severe bleeding",
        "seizure",
        "stroke symptoms",
    ]

    has_critical = any(
        flag.lower() in rf.lower() for rf in red_flags for flag in critical_flags
    )

    if has_critical:
        state["risk_level"] = RiskLevel.CRITICAL
        state["risk_confidence"] = 0.98
        state["triage_complete"] = True
        state["requires_human_review"] = True

        state["agent_decisions"].append(
            {
                "agent_name": "Triage",
                "timestamp": datetime.utcnow().isoformat(),
                "action": "marked_critical",
                "details": {"red_flags": red_flags},
                "reasoning": f"Critical red flags detected: {red_flags}",
                "confidence": 0.98,
            }
        )

        logger.warning(f"🚨 CRITICAL case detected: {red_flags}")
        return state

    if red_flags:
        state["risk_level"] = RiskLevel.URGENT
        state["risk_confidence"] = 0.9
        state["triage_complete"] = True

        state["agent_decisions"].append(
            {
                "agent_name": "Triage",
                "timestamp": datetime.utcnow().isoformat(),
                "action": "marked_urgent",
                "details": {"red_flags": red_flags},
                "reasoning": f"Red flags detected: {red_flags}",
                "confidence": 0.9,
            }
        )

        logger.info(f"⚠️ Urgent case: {red_flags}")
        return state

    # Use LLM for nuanced triage
    llm_service = get_llm_service()

    symptoms_text = ", ".join(all_symptoms)
    severity = state.get("severity", "unknown")
    duration = state.get("duration", "unknown")

    prompt = f"""Assess the urgency of this patient case for a rural PHC in India.

Chief Complaint: {state.get("chief_complaint", "Unknown")}
Symptoms: {symptoms_text}
Duration: {duration}
Severity Reported: {severity}
Patient Age Group: {state.get("age_group", "Unknown")}
Is Pregnant: {state.get("is_pregnant", "Unknown")}

Consider:
1. Symptom progression risk
2. Common conditions in rural India (malaria, dengue, typhoid, etc.)
3. Access to healthcare (may take hours to reach hospital)
4. Patient's vulnerability (age, pregnancy)

Classify as:
- "critical": Life-threatening, needs immediate referral
- "urgent": Needs medical attention within hours
- "moderate": Needs attention within 24-48 hours  
- "routine": Can wait for regular appointment

Return JSON:
{{
    "risk_level": "critical/urgent/moderate/routine",
    "confidence": 0.0-1.0,
    "reasoning": "brief clinical reasoning",
    "differential_diagnosis": [
        {{"condition": "name", "probability": "high/medium/low", "icd11": "code if known"}}
    ],
    "recommended_timeframe": "when patient should be seen"
}}"""

    try:
        response = await llm_service.invoke(
            messages=[HumanMessage(content=prompt)],
            system_prompt="You are an experienced medical triage assistant familiar with rural Indian healthcare.",
        )

        response_text = response.strip()
        if "```" in response_text:
            response_text = response_text.split("```")[1].replace("json", "").strip()

        result = json.loads(response_text)

        risk_level_str = result.get("risk_level", "moderate").lower()
        state["risk_level"] = (
            RiskLevel(risk_level_str)
            if risk_level_str in [r.value for r in RiskLevel]
            else RiskLevel.MODERATE
        )
        state["risk_confidence"] = result.get("confidence", 0.7)
        state["differential_diagnosis"] = result.get("differential_diagnosis", [])
        state["urgency_explanation"] = result.get("reasoning", "")

        state["agent_decisions"].append(
            {
                "agent_name": "Triage",
                "timestamp": datetime.utcnow().isoformat(),
                "action": f"classified_{state['risk_level'].value}",
                "details": {
                    "confidence": state["risk_confidence"],
                    "differential_count": len(state["differential_diagnosis"]),
                },
                "reasoning": result.get("reasoning", "LLM triage assessment"),
                "confidence": state["risk_confidence"],
            }
        )

    except Exception as e:
        logger.error(f"Triage LLM error: {e}")
        state["risk_level"] = RiskLevel.MODERATE
        state["risk_confidence"] = 0.5
        state["warnings"].append("Triage LLM failed - defaulting to moderate")

    state["triage_complete"] = True
    logger.info(
        f"✅ Triage complete: {state['risk_level'].value} (confidence: {state['risk_confidence']:.2f})"
    )

    return state


# ============= AGENT 5: Clinical Decision Support =============
async def clinical_decision_agent(state: GHIAState) -> GHIAState:
    """
    Provides clinical decision support including:
    - ICD-11 coding
    - Treatment suggestions
    - Referral recommendations
    """
    logger.info("🏥 Clinical Decision Agent running...")

    settings = get_settings()

    # Skip if feature disabled
    if (
        not settings.enable_drug_interaction_check
        and not settings.enable_differential_diagnosis
    ):
        state["agent_decisions"].append(
            {
                "agent_name": "Clinical Decision",
                "timestamp": datetime.utcnow().isoformat(),
                "action": "skipped",
                "reasoning": "Clinical decision features disabled",
                "confidence": 1.0,
            }
        )
        return state

    from app.agents.clinical_decision import ClinicalDecisionSupport

    cds = ClinicalDecisionSupport()

    # Get ICD-11 codes
    symptoms = [s.get("symptom", "") for s in state.get("symptoms", [])]
    chief_complaint = state.get("chief_complaint", "")

    icd_codes = cds.get_icd11_codes(symptoms, chief_complaint)
    state["icd11_codes"] = icd_codes

    # Get treatment recommendations
    age_group = state.get("age_group", "adult")
    is_pregnant = state.get("is_pregnant", False)

    treatment = cds.get_treatment_recommendation(
        chief_complaint=chief_complaint,
        symptoms=symptoms,
        age_group=age_group,
        is_pregnant=is_pregnant,
    )
    state["treatment_suggestions"] = treatment.get("medications", [])

    # Check for referral need
    referral = cds.determine_referral_need(
        risk_level=state.get("risk_level", RiskLevel.ROUTINE).value,
        red_flags=state.get("red_flags", []),
        icd_codes=icd_codes,
    )
    state["referral_recommendation"] = referral

    # Update risk level if referral is urgent
    if referral.get("needed") and referral.get("urgency") == "urgent":
        if state["risk_level"] not in [RiskLevel.CRITICAL, RiskLevel.URGENT]:
            state["risk_level"] = RiskLevel.URGENT
            state["warnings"].append(
                "Risk level upgraded based on referral recommendation"
            )

    state["agent_decisions"].append(
        {
            "agent_name": "Clinical Decision",
            "timestamp": datetime.utcnow().isoformat(),
            "action": "clinical_analysis",
            "details": {
                "icd_codes": len(icd_codes),
                "treatment_suggestions": len(state["treatment_suggestions"]),
                "referral_needed": referral.get("needed", False),
            },
            "reasoning": f"Identified {len(icd_codes)} ICD codes, referral: {referral.get('needed')}",
            "confidence": 0.85,
        }
    )

    logger.info(
        f"✅ Clinical analysis: {len(icd_codes)} ICD codes, Referral: {referral.get('needed')}"
    )

    return state


# ============= AGENT 6: Output Generator =============
async def output_agent(state: GHIAState) -> GHIAState:
    """
    Generates bilingual summaries and recommended actions.
    Ensures output is appropriate for both patients and healthcare workers.
    """
    logger.info("📝 Output Agent running...")

    llm_service = get_llm_service()

    prompt = f"""Create comprehensive medical summaries for a patient intake.

CASE DETAILS:
- Chief Complaint: {state.get("chief_complaint", "Not specified")}
- Symptoms: {json.dumps(state.get("symptoms", []), ensure_ascii=False)}
- Duration: {state.get("duration", "Not specified")}
- Risk Level: {state.get("risk_level", RiskLevel.MODERATE).value}
- Red Flags: {state.get("red_flags", [])}
- Differential Diagnosis: {json.dumps(state.get("differential_diagnosis", []), ensure_ascii=False)}

REQUIREMENTS:
1. English summary: Clinical, professional, for doctor dashboard (2-3 sentences)
2. Hindi summary: Patient-friendly, in Devanagari script (2-3 sentences)
3. Recommended action: Specific, actionable guidance
4. Urgency explanation: Why this risk level was assigned

Return JSON:
{{
    "summary_english": "Clinical summary for healthcare worker...",
    "summary_hindi": "मरीज़ के लिए सारांश...",
    "recommended_action": "Specific next steps...",
    "urgency_explanation": "This case is {state.get("risk_level", "moderate").value} because...",
    "patient_instructions_hindi": "मरीज़ के लिए निर्देश..."
}}"""

    try:
        response = await llm_service.invoke(
            messages=[HumanMessage(content=prompt)],
            system_prompt="You are a medical documentation specialist creating bilingual summaries.",
        )

        response_text = response.strip()
        if "```" in response_text:
            response_text = response_text.split("```")[1].replace("json", "").strip()

        result = json.loads(response_text)

        state["summary_english"] = result.get("summary_english", "Summary unavailable")
        state["summary_hindi"] = result.get("summary_hindi", "सारांश उपलब्ध नहीं है")
        state["recommended_action"] = result.get(
            "recommended_action", "Consult physician"
        )
        state["urgency_explanation"] = result.get("urgency_explanation", "")

    except Exception as e:
        logger.error(f"Output agent error: {e}")
        # Fallback summaries
        state["summary_english"] = (
            f"Patient reports {state.get('chief_complaint', 'health issues')}. Risk level: {state.get('risk_level', RiskLevel.MODERATE).value}."
        )
        state["summary_hindi"] = (
            f"मरीज़ को {state.get('chief_complaint', 'स्वास्थ्य समस्या')} की शिकायत है।"
        )
        state["recommended_action"] = "Medical consultation recommended"

    state["agent_decisions"].append(
        {
            "agent_name": "Output Generator",
            "timestamp": datetime.utcnow().isoformat(),
            "action": "generated_summaries",
            "details": {
                "has_english": bool(state["summary_english"]),
                "has_hindi": bool(state["summary_hindi"]),
            },
            "reasoning": "Generated bilingual summaries for dashboard and patient",
            "confidence": 0.9,
        }
    )

    logger.info("✅ Output generation complete")

    return state


# ============= ROUTING LOGIC =============
def should_run_interrogator(state: GHIAState) -> Literal["interrogator", "triage"]:
    """Decide whether to run interrogator based on extraction quality"""
    if state.get("needs_followup") or len(state.get("missing_information", [])) > 0:
        return "interrogator"
    if state.get("extraction_confidence", 0) < 0.6:
        return "interrogator"
    return "triage"


def should_run_clinical_decision(
    state: GHIAState,
) -> Literal["clinical_decision", "output"]:
    """Decide whether to run clinical decision agent"""
    settings = get_settings()
    if not settings.enable_differential_diagnosis:
        return "output"
    # Skip for routine cases to save processing
    if state.get("risk_level") == RiskLevel.ROUTINE and not state.get("red_flags"):
        return "output"
    return "clinical_decision"


# ============= BUILD THE GRAPH =============
def create_ghia_graph():
    """Creates the LangGraph workflow with conditional routing"""

    graph = StateGraph(GHIAState)

    # Add nodes
    graph.add_node("extractor", medical_extractor_agent)
    graph.add_node("quality_check", quality_checker_agent)
    graph.add_node("interrogator", interrogator_agent)
    graph.add_node("outbreak_check", outbreak_awareness_agent)  
    graph.add_node("triage", triage_agent)
    graph.add_node("clinical_decision", clinical_decision_agent)
    graph.add_node("output", output_agent)

    # Set entry point
    graph.set_entry_point("extractor")

    # Add edges with conditional routing
    graph.add_edge("extractor", "quality_check")
    graph.add_conditional_edges(
        "quality_check",
        should_run_interrogator,
        {"interrogator": "interrogator", "triage": "outbreak_check"},
    )
    graph.add_edge("interrogator", "outbreak_check")
    graph.add_edge("outbreak_check", "triage")
    graph.add_conditional_edges(
        "triage",
        should_run_clinical_decision,
        {"clinical_decision": "clinical_decision", "output": "output"},
    )
    graph.add_edge("clinical_decision", "output")
    graph.add_edge("output", END)

    # Compile with memory for multi-turn conversations
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


# Global graph instance
_graph = None


def get_graph():
    """Get or create the agent graph"""
    global _graph
    if _graph is None:
        _graph = create_ghia_graph()
    return _graph


async def run_agent_pipeline(
    raw_transcript: str,
    language: str = "hi",
    session_id: str = None,
    patient_id: str = None,
    phc_id: str = None,
    conversation_history: List[Dict] = None,
) -> Dict[str, Any]:
    """
    Run the full multi-agent pipeline on a transcript.

    Args:
        raw_transcript: Patient's raw input text
        language: Detected or specified language
        session_id: Session identifier for multi-turn
        patient_id: Optional patient identifier
        phc_id: PHC identifier
        conversation_history: Previous conversation turns

    Returns:
        Complete intake result dictionary
    """
    initial_state = create_initial_state(
        raw_transcript=raw_transcript,
        language=language,
        session_id=session_id,
        patient_id=patient_id,
        phc_id=phc_id,
        conversation_history=conversation_history,
    )

    graph = get_graph()

    # Run with session config for memory
    config = {"configurable": {"thread_id": session_id or "default"}}
    # final_state = await asyncio.to_thread(graph.invoke, initial_state, config)
    final_state = await graph.ainvoke(initial_state, config)

    # Save to database
    from app.db.repository import save_intake

    record_id = save_intake(final_state)

    return {
        "id": record_id,
        "session_id": final_state.get("session_id"),
        "raw_transcript": raw_transcript,
        "language_detected": language,
        "chief_complaint": final_state.get("chief_complaint"),
        "symptoms": final_state.get("symptoms"),
        "duration": final_state.get("duration"),
        "severity": final_state.get("severity"),
        "risk_level": final_state.get("risk_level", RiskLevel.ROUTINE).value,
        "risk_confidence": final_state.get("risk_confidence", 0),
        "red_flags": final_state.get("red_flags"),
        "differential_diagnosis": final_state.get("differential_diagnosis"),
        "summary_english": final_state.get("summary_english"),
        "summary_hindi": final_state.get("summary_hindi"),
        "recommended_action": final_state.get("recommended_action"),
        "urgency_explanation": final_state.get("urgency_explanation"),
        "followup_questions": final_state.get("followup_questions"),
        "icd11_codes": final_state.get("icd11_codes"),
        "treatment_suggestions": final_state.get("treatment_suggestions"),
        "referral_recommendation": final_state.get("referral_recommendation"),
        "agent_decisions": final_state.get("agent_decisions"),
        "requires_human_review": final_state.get("requires_human_review", False),
        "errors": final_state.get("errors"),
        "warnings": final_state.get("warnings"),
    }
