"""Enhanced Clinical Decision Support Agent"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import logging
import re

from app.agents.base_agent import LLMAgent, AgentContext, AgentResult
from app.config import get_settings
from app.services.security import get_audit_logger, AuditLogger

logger = logging.getLogger(__name__)


class UrgencyLevel(str, Enum):
    EMERGENCY = "emergency"
    URGENT = "urgent"
    MODERATE = "moderate"
    ROUTINE = "routine"
    SELF_CARE = "self_care"


class ClinicalFlag(str, Enum):
    RED_FLAG = "red_flag"
    YELLOW_FLAG = "yellow_flag"
    PREGNANCY = "pregnancy"
    PEDIATRIC = "pediatric"
    ELDERLY = "elderly"
    CHRONIC = "chronic"


@dataclass
class SymptomAnalysis:
    symptoms: List[Dict[str, Any]]
    duration: Optional[str]
    severity: str
    progression: str
    associated_symptoms: List[str]
    red_flags: List[str]

    def has_red_flags(self) -> bool:
        return len(self.red_flags) > 0


@dataclass
class DiagnosisResult:
    condition: str
    icd10_code: Optional[str]
    confidence: float
    supporting_evidence: List[str]
    ruling_out: List[str]
    tests_recommended: List[str]


@dataclass
class TreatmentPlan:
    medications: List[Dict[str, Any]]
    non_pharmacological: List[str]
    lifestyle_modifications: List[str]
    follow_up_days: int
    referral_needed: bool
    referral_type: Optional[str]
    referral_facility: Optional[str]
    patient_education: List[str]
    warning_signs: List[str] = field(default_factory=list)


class ClinicalDecisionAgent(LLMAgent):
    """Advanced clinical decision support agent for PHC-level care."""

    SYSTEM_PROMPT = """You are a Clinical Decision Support System for Primary Health Centers (PHCs) in rural India.

Your role:
1. Systematic symptom assessment
2. Identify danger signs (red flags)
3. Suggest differential diagnoses
4. Recommend treatment per WHO/national guidelines
5. Determine urgency and referral needs

CRITICAL RULES:
- Always prioritize patient safety
- Flag danger signs immediately
- Consider common conditions in rural India (TB, malaria, dengue, typhoid)
- Account for limited PHC resources
- Provide clear, actionable guidance
- Use simple language for health workers
- Recommend referral when uncertain

Output analysis in valid JSON format."""

    RED_FLAGS = {
        "general": [
            "unconsciousness",
            "difficulty breathing",
            "severe chest pain",
            "sudden severe headache",
            "seizures",
            "high fever > 104F",
            "severe dehydration",
            "severe bleeding",
        ],
        "pediatric": [
            "not feeding",
            "lethargy",
            "convulsions",
            "severe malnutrition",
            "bulging fontanelle",
            "fast breathing",
        ],
        "pregnancy": [
            "severe headache with blurred vision",
            "vaginal bleeding",
            "severe abdominal pain",
            "no fetal movement",
            "convulsions",
        ],
    }

    def __init__(self):
        super().__init__(
            name="clinical_decision_agent", system_prompt=self.SYSTEM_PROMPT
        )
        self.audit = get_audit_logger()

    async def execute(
        self,
        context: AgentContext,
        symptoms: List[str] = None,
        patient_data: Dict[str, Any] = None,
        **kwargs,
    ) -> AgentResult[Dict[str, Any]]:
        if not symptoms and not patient_data:
            return AgentResult(
                success=False, error="No symptoms or patient data provided"
            )

        try:
            symptom_analysis = await self._analyze_symptoms(
                context, symptoms or [], patient_data or {}
            )
            urgency = self._assess_urgency(symptom_analysis, patient_data or {})
            differentials = await self._generate_differentials(
                context, symptom_analysis, patient_data or {}
            )
            treatment = await self._create_treatment_plan(
                context, symptom_analysis, differentials, urgency
            )

            result = {
                "symptom_analysis": {
                    "symptoms": symptom_analysis.symptoms,
                    "severity": symptom_analysis.severity,
                    "red_flags": symptom_analysis.red_flags,
                    "progression": symptom_analysis.progression,
                },
                "urgency": {
                    "level": urgency.value,
                    "requires_immediate_action": urgency == UrgencyLevel.EMERGENCY,
                    "recommended_timeframe": self._get_timeframe(urgency),
                },
                "differential_diagnoses": [
                    {
                        "condition": d.condition,
                        "icd10": d.icd10_code,
                        "confidence": d.confidence,
                        "evidence": d.supporting_evidence,
                    }
                    for d in differentials[:5]
                ],
                "treatment_plan": {
                    "medications": treatment.medications,
                    "advice": treatment.non_pharmacological,
                    "follow_up_days": treatment.follow_up_days,
                    "referral_needed": treatment.referral_needed,
                    "referral_type": treatment.referral_type,
                    "patient_education": treatment.patient_education,
                },
                "flags": self._get_clinical_flags(patient_data or {}, symptom_analysis),
            }

            self.audit.log(
                user_id=context.user_id,
                action=AuditLogger.AuditAction.READ,
                resource_type="clinical_decision",
                patient_id=context.patient_id,
                phc_id=context.phc_id,
                details={
                    "urgency": urgency.value,
                    "red_flags_count": len(symptom_analysis.red_flags),
                },
            )

            return AgentResult(
                success=True,
                data=result,
                confidence=differentials[0].confidence if differentials else 0.0,
                next_actions=self._determine_next_actions(urgency, treatment),
            )
        except Exception as e:
            logger.exception(f"Clinical decision error: {e}")
            return AgentResult(success=False, error=str(e))

    async def _analyze_symptoms(
        self, context: AgentContext, symptoms: List[str], patient_data: Dict
    ) -> SymptomAnalysis:
        prompt = f"""Analyze symptoms:
Patient: Age {patient_data.get("age", "Unknown")}, {patient_data.get("gender", "Unknown")}
Symptoms: {", ".join(symptoms) if symptoms else patient_data.get("chief_complaint", "Not specified")}
Duration: {patient_data.get("duration", "Not specified")}

Provide JSON: {{"symptoms": [{{"name": "x", "severity": "mild/moderate/severe"}}], "overall_severity": "mild/moderate/severe", "progression": "improving/stable/worsening", "associated_symptoms": [], "red_flags": []}}"""

        try:
            response = await self.generate(prompt, context)
            data = self._parse_json(response)
            detected_red_flags = self._detect_red_flags(symptoms, patient_data)
            return SymptomAnalysis(
                symptoms=data.get("symptoms", [{"name": s} for s in symptoms]),
                duration=patient_data.get("duration"),
                severity=data.get("overall_severity", "moderate"),
                progression=data.get("progression", "unknown"),
                associated_symptoms=data.get("associated_symptoms", []),
                red_flags=list(set(data.get("red_flags", []) + detected_red_flags)),
            )
        except:
            return SymptomAnalysis(
                symptoms=[{"name": s} for s in symptoms],
                duration=patient_data.get("duration"),
                severity="moderate",
                progression="unknown",
                associated_symptoms=[],
                red_flags=self._detect_red_flags(symptoms, patient_data),
            )

    def _detect_red_flags(self, symptoms: List[str], patient_data: Dict) -> List[str]:
        detected = []
        text = " ".join(symptoms).lower()
        for flag in self.RED_FLAGS["general"]:
            if flag.lower() in text:
                detected.append(flag)
        return detected

    def _assess_urgency(
        self, analysis: SymptomAnalysis, patient_data: Dict
    ) -> UrgencyLevel:
        if analysis.has_red_flags():
            return UrgencyLevel.EMERGENCY
        if analysis.severity == "severe":
            return UrgencyLevel.URGENT
        if analysis.severity == "moderate" or analysis.progression == "worsening":
            return UrgencyLevel.MODERATE
        return UrgencyLevel.ROUTINE

    async def _generate_differentials(
        self, context: AgentContext, analysis: SymptomAnalysis, patient_data: Dict
    ) -> List[DiagnosisResult]:
        symptoms_text = ", ".join(
            [
                s.get("name", str(s)) if isinstance(s, dict) else str(s)
                for s in analysis.symptoms
            ]
        )
        prompt = f"""Differential diagnoses for: {symptoms_text}
Patient: {patient_data.get("age", "Unknown")} y/o {patient_data.get("gender", "Unknown")}

JSON: {{"differentials": [{{"condition": "x", "icd10_code": "x", "confidence": 0.8, "supporting_evidence": [], "ruling_out": [], "tests_recommended": []}}]}}"""

        try:
            response = await self.generate(prompt, context)
            data = self._parse_json(response)
            return [DiagnosisResult(**d) for d in data.get("differentials", [])[:5]]
        except:
            return [
                DiagnosisResult(
                    condition="Requires evaluation",
                    icd10_code=None,
                    confidence=0.3,
                    supporting_evidence=[],
                    ruling_out=[],
                    tests_recommended=["Physical exam"],
                )
            ]

    async def _create_treatment_plan(
        self,
        context: AgentContext,
        analysis: SymptomAnalysis,
        differentials: List[DiagnosisResult],
        urgency: UrgencyLevel,
    ) -> TreatmentPlan:
        if urgency == UrgencyLevel.EMERGENCY:
            return TreatmentPlan(
                medications=[],
                non_pharmacological=["Stabilize", "Prepare transfer"],
                lifestyle_modifications=[],
                follow_up_days=0,
                referral_needed=True,
                referral_type="emergency",
                referral_facility="District Hospital",
                patient_education=["Emergency - immediate attention needed"],
                warning_signs=analysis.red_flags,
            )

        diagnosis = differentials[0].condition if differentials else "Unknown"
        prompt = f"""Treatment plan for PHC:
Diagnosis: {diagnosis}, Severity: {analysis.severity}

JSON: {{"medications": [{{"name": "x", "dose": "x", "frequency": "x", "duration": "x"}}], "non_pharmacological": [], "lifestyle_modifications": [], "follow_up_days": 7, "referral_needed": false, "referral_type": null, "patient_education": [], "warning_signs": []}}"""

        try:
            response = await self.generate(prompt, context)
            data = self._parse_json(response)
            return TreatmentPlan(
                **{
                    k: data.get(k, v)
                    for k, v in TreatmentPlan(
                        [], [], [], 7, False, None, None, []
                    ).__dict__.items()
                }
            )
        except:
            return TreatmentPlan(
                medications=[],
                non_pharmacological=["Consult physician"],
                lifestyle_modifications=["Rest", "Hydration"],
                follow_up_days=3,
                referral_needed=True,
                referral_type="consultation",
                referral_facility="PHC Medical Officer",
                patient_education=["Please consult doctor"],
                warning_signs=["If symptoms worsen"],
            )

    def _get_timeframe(self, urgency: UrgencyLevel) -> str:
        return {
            "emergency": "Immediate",
            "urgent": "Within 2-4 hours",
            "moderate": "Within 24-48 hours",
            "routine": "Within 1 week",
            "self_care": "As needed",
        }.get(urgency, "ASAP")

    def _get_clinical_flags(
        self, patient_data: Dict, analysis: SymptomAnalysis
    ) -> List[str]:
        flags = []
        if analysis.has_red_flags():
            flags.append(ClinicalFlag.RED_FLAG.value)
        if patient_data.get("is_pregnant"):
            flags.append(ClinicalFlag.PREGNANCY.value)
        return flags

    def _determine_next_actions(
        self, urgency: UrgencyLevel, treatment: TreatmentPlan
    ) -> List[str]:
        actions = []
        if urgency == UrgencyLevel.EMERGENCY:
            actions.append("IMMEDIATE: Arrange emergency transport")
        if treatment.referral_needed:
            actions.append(f"Refer to: {treatment.referral_facility}")
        if treatment.follow_up_days > 0:
            actions.append(f"Follow-up in {treatment.follow_up_days} days")
        return actions

    def _parse_json(self, response: str) -> Dict:
        response = re.sub(r"^```\w*\n?", "", response.strip())
        response = re.sub(r"\n?```$", "", response)
        try:
            return json.loads(response)
        except:
            match = re.search(r"\{.*\}", response, re.DOTALL)
            return json.loads(match.group()) if match else {}


class ClinicalDecisionSupport:
    """
    Clinical decision support utilities for ICD coding,
    treatment recommendations, and referral decisions.
    Used by the orchestrator's clinical_decision_agent.
    """

    # Common ICD-11 codes for PHC conditions
    ICD11_MAPPINGS = {
        "fever": ("MG26", "Fever, unspecified"),
        "headache": ("8A80", "Headache disorder"),
        "cough": ("MD12", "Cough"),
        "diarrhea": ("ME05", "Diarrhoea"),
        "abdominal pain": ("MD81", "Abdominal pain"),
        "chest pain": ("MD30", "Chest pain"),
        "vomiting": ("MD90", "Nausea or vomiting"),
        "weakness": ("MG22", "Fatigue"),
        "cold": ("CA00", "Common cold"),
        "body pain": ("ME84", "Pain in limb"),
        "breathing difficulty": ("MD11", "Dyspnoea"),
        "skin rash": ("ME66", "Rash"),
        "sore throat": ("CA02", "Acute pharyngitis"),
        "bukhar": ("MG26", "Fever, unspecified"),
        "sir dard": ("8A80", "Headache disorder"),
        "khansi": ("MD12", "Cough"),
        "dast": ("ME05", "Diarrhoea"),
        "pet dard": ("MD81", "Abdominal pain"),
        "seene mein dard": ("MD30", "Chest pain"),
        "ulti": ("MD90", "Nausea or vomiting"),
        "kamzori": ("MG22", "Fatigue"),
    }

    # Basic treatment templates
    TREATMENT_TEMPLATES = {
        "fever": {
            "medications": [
                {
                    "name": "Paracetamol",
                    "dose": "500mg",
                    "frequency": "TID",
                    "duration": "3 days",
                }
            ],
            "advice": ["Rest", "Increase fluid intake", "Monitor temperature"],
        },
        "cough": {
            "medications": [
                {
                    "name": "Cough syrup",
                    "dose": "10ml",
                    "frequency": "TID",
                    "duration": "5 days",
                }
            ],
            "advice": ["Warm fluids", "Steam inhalation", "Avoid cold items"],
        },
        "diarrhea": {
            "medications": [
                {
                    "name": "ORS",
                    "dose": "1 packet in 1L water",
                    "frequency": "After each loose stool",
                    "duration": "Until recovery",
                }
            ],
            "advice": ["Stay hydrated", "BRAT diet", "Avoid dairy"],
        },
        "headache": {
            "medications": [
                {
                    "name": "Paracetamol",
                    "dose": "500mg",
                    "frequency": "SOS",
                    "duration": "As needed",
                }
            ],
            "advice": ["Rest in dark room", "Stay hydrated", "Reduce screen time"],
        },
        "body pain": {
            "medications": [
                {
                    "name": "Paracetamol",
                    "dose": "500mg",
                    "frequency": "TID",
                    "duration": "3 days",
                }
            ],
            "advice": ["Rest", "Warm compress", "Gentle stretching"],
        },
    }

    def get_icd11_codes(
        self, symptoms: List[str], chief_complaint: str
    ) -> List[Dict[str, str]]:
        """Map symptoms to ICD-11 codes."""
        codes = []
        all_text = " ".join(symptoms + [chief_complaint]).lower()

        for keyword, (code, description) in self.ICD11_MAPPINGS.items():
            if keyword in all_text:
                codes.append(
                    {
                        "code": code,
                        "description": description,
                        "matched_keyword": keyword,
                    }
                )

        # Remove duplicates by code
        seen_codes = set()
        unique_codes = []
        for c in codes:
            if c["code"] not in seen_codes:
                seen_codes.add(c["code"])
                unique_codes.append(c)

        return unique_codes[:5]  # Return top 5 matches

    def get_treatment_recommendation(
        self,
        chief_complaint: str,
        symptoms: List[str],
        age_group: str = "adult",
        is_pregnant: bool = False,
    ) -> Dict[str, Any]:
        """Get treatment recommendations based on symptoms."""
        all_text = " ".join([chief_complaint] + symptoms).lower()

        medications = []
        advice = []

        for condition, treatment in self.TREATMENT_TEMPLATES.items():
            if condition in all_text:
                # Adjust dosage for age/pregnancy
                for med in treatment["medications"]:
                    adjusted_med = med.copy()
                    if age_group == "pediatric":
                        adjusted_med["note"] = "Adjust dose for child's weight"
                    if is_pregnant:
                        adjusted_med["note"] = (
                            "Consult doctor before use during pregnancy"
                        )
                    medications.append(adjusted_med)
                advice.extend(treatment["advice"])

        # Default if nothing matched
        if not medications and not advice:
            advice = ["Rest", "Stay hydrated", "Monitor symptoms"]

        return {
            "medications": medications[:5],
            "advice": list(set(advice))[:5],
            "general": ["Follow up if symptoms persist or worsen"],
        }

    def determine_referral_need(
        self, risk_level: str, red_flags: List[str], icd_codes: List[Dict]
    ) -> Dict[str, Any]:
        """Determine if referral is needed."""
        if risk_level in ["critical", "urgent"] or red_flags:
            return {
                "needed": True,
                "urgency": "urgent" if risk_level == "critical" else "routine",
                "facility": "District Hospital" if risk_level == "critical" else "CHC",
                "reason": f"Red flags detected: {', '.join(red_flags)}"
                if red_flags
                else "High risk assessment",
            }

        return {
            "needed": False,
            "urgency": None,
            "facility": None,
            "reason": "Can be managed at PHC level",
        }


_clinical_agent: Optional[ClinicalDecisionAgent] = None


def get_clinical_decision_agent() -> ClinicalDecisionAgent:
    global _clinical_agent
    if _clinical_agent is None:
        _clinical_agent = ClinicalDecisionAgent()
    return _clinical_agent
