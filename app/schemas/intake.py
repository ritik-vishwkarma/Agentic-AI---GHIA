"""Pydantic schemas for API requests/responses"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


class RiskLevel(str, Enum):
    URGENT = "urgent"
    MODERATE = "moderate"
    ROUTINE = "routine"


class SymptomInfo(BaseModel):
    """Extracted symptom information"""
    symptom: str
    severity: Optional[str] = None
    duration: Optional[str] = None
    body_part: Optional[str] = None


class PatientIntake(BaseModel):
    """Complete patient intake data"""
    chief_complaint: Optional[str] = None
    symptoms: List[SymptomInfo] = []
    duration: Optional[str] = None
    severity: Optional[str] = None
    associated_symptoms: List[str] = []
    medical_history: Optional[str] = None
    age_group: Optional[str] = None
    gender: Optional[str] = None


class AgentDecision(BaseModel):
    """Decision made by an agent"""
    agent_name: str
    action: str
    reasoning: str
    confidence: float = Field(ge=0, le=1)


class IntakeResult(BaseModel):
    """Complete result from the multi-agent pipeline"""
    id: int
    created_at: datetime
    
    # Raw data
    raw_transcript: str
    language_detected: str
    
    # Processed data
    intake: PatientIntake
    risk_level: RiskLevel
    
    # Agent trace (for transparency)
    agent_decisions: List[AgentDecision] = []
    
    # Summaries
    summary_english: str
    summary_hindi: str
    
    # Follow-up
    follow_up_questions: List[str] = []
    recommended_action: str


class IntakeListItem(BaseModel):
    """Summary item for dashboard list"""
    id: int
    created_at: datetime
    chief_complaint: Optional[str]
    risk_level: RiskLevel
    summary_english: str
    summary_hindi: str


class ConversationTurn(BaseModel):
    """A single turn in the conversation"""
    role: str  # "patient" or "agent"
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
