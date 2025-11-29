"""
Continuous Learning System (Simplified)
- Captures doctor corrections and feedback
- Stores learnings in database
- Enriches agent prompts with learned patterns
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import json
import sqlite3
import logging

logger = logging.getLogger(__name__)

DB_PATH = "ghia.db"


class FeedbackType(str, Enum):
    SYMPTOM_CORRECTION = "symptom_correction"
    DIAGNOSIS_CORRECTION = "diagnosis_correction"
    TREATMENT_MODIFICATION = "treatment_modification"
    TRIAGE_ADJUSTMENT = "triage_adjustment"
    POSITIVE_CONFIRMATION = "positive_confirmation"


@dataclass
class DoctorFeedback:
    """Structured feedback from healthcare provider"""
    feedback_id: str
    intake_id: int
    doctor_id: str
    feedback_type: FeedbackType
    original_prediction: Dict[str, Any]
    corrected_values: Dict[str, Any]
    reasoning: Optional[str]
    severity_score: float
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "feedback_id": self.feedback_id,
            "intake_id": self.intake_id,
            "doctor_id": self.doctor_id,
            "feedback_type": self.feedback_type.value,
            "original": self.original_prediction,
            "corrected": self.corrected_values,
            "reasoning": self.reasoning,
            "severity_score": self.severity_score,
            "timestamp": self.timestamp.isoformat()
        }


class FeedbackLearningEngine:
    """
    Captures doctor corrections and provides learned context to agents.
    """
    
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_tables()
        
        # In-memory cache for quick access
        self._learned_symptoms: Dict[str, str] = {}
        self._diagnosis_hints: List[Dict] = []
        self._load_learnings()
    
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_tables(self):
        """Create feedback tracking tables"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS doctor_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feedback_id TEXT UNIQUE NOT NULL,
                intake_id INTEGER NOT NULL,
                doctor_id TEXT NOT NULL,
                feedback_type TEXT NOT NULL,
                original_prediction TEXT,
                corrected_values TEXT,
                reasoning TEXT,
                severity_score REAL,
                created_at TEXT NOT NULL,
                processed INTEGER DEFAULT 0
            )
            """)
            
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS learned_mappings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mapping_type TEXT NOT NULL,
                input_text TEXT NOT NULL,
                output_text TEXT NOT NULL,
                confidence REAL DEFAULT 0.5,
                usage_count INTEGER DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(mapping_type, input_text)
            )
            """)
            
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                total_cases INTEGER DEFAULT 0,
                corrections_needed INTEGER DEFAULT 0,
                accuracy REAL,
                avg_severity REAL,
                created_at TEXT NOT NULL
            )
            """)
            
            conn.commit()
            conn.close()
            logger.info("✅ Feedback learning tables initialized")
        except Exception as e:
            logger.error(f"Failed to init feedback tables: {e}")
    
    def _load_learnings(self):
        """Load learned mappings into memory"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Load symptom mappings
            cursor.execute("""
            SELECT input_text, output_text 
            FROM learned_mappings 
            WHERE mapping_type = 'symptom' AND confidence > 0.6
            """)
            
            for row in cursor.fetchall():
                self._learned_symptoms[row["input_text"].lower()] = row["output_text"]
            
            # Load recent diagnosis hints
            cursor.execute("""
            SELECT corrected_values, reasoning
            FROM doctor_feedback
            WHERE feedback_type = 'diagnosis_correction'
            AND created_at > ?
            ORDER BY created_at DESC
            LIMIT 20
            """, ((datetime.utcnow() - timedelta(days=30)).isoformat(),))
            
            for row in cursor.fetchall():
                try:
                    self._diagnosis_hints.append({
                        "correction": json.loads(row["corrected_values"]),
                        "reasoning": row["reasoning"]
                    })
                except:
                    pass
            
            conn.close()
            logger.info(f"📚 Loaded {len(self._learned_symptoms)} symptom mappings, {len(self._diagnosis_hints)} diagnosis hints")
        except Exception as e:
            logger.warning(f"Failed to load learnings: {e}")
    
    def capture_feedback(
        self,
        intake_id: int,
        doctor_id: str,
        feedback_type: FeedbackType,
        original: Dict[str, Any],
        corrected: Dict[str, Any],
        reasoning: str = None
    ) -> DoctorFeedback:
        """Capture and store doctor's correction"""
        
        severity = self._calculate_severity(original, corrected, feedback_type)
        
        feedback = DoctorFeedback(
            feedback_id=f"FB-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{intake_id}",
            intake_id=intake_id,
            doctor_id=doctor_id,
            feedback_type=feedback_type,
            original_prediction=original,
            corrected_values=corrected,
            reasoning=reasoning,
            severity_score=severity,
            timestamp=datetime.utcnow()
        )
        
        # Store in database
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
            INSERT INTO doctor_feedback 
            (feedback_id, intake_id, doctor_id, feedback_type, 
             original_prediction, corrected_values, reasoning, severity_score, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                feedback.feedback_id,
                feedback.intake_id,
                feedback.doctor_id,
                feedback.feedback_type.value,
                json.dumps(feedback.original_prediction),
                json.dumps(feedback.corrected_values),
                feedback.reasoning,
                feedback.severity_score,
                feedback.timestamp.isoformat()
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to save feedback: {e}")
        
        # Apply learning
        self._apply_learning(feedback)
        
        logger.info(f"📝 Feedback captured: {feedback_type.value} (severity: {severity:.2f})")
        
        return feedback
    
    def _calculate_severity(
        self,
        original: Dict,
        corrected: Dict,
        feedback_type: FeedbackType
    ) -> float:
        """Calculate error severity (0-1)"""
        
        if feedback_type == FeedbackType.POSITIVE_CONFIRMATION:
            return 0.0
        
        if feedback_type == FeedbackType.TRIAGE_ADJUSTMENT:
            risk_levels = {"routine": 0, "moderate": 1, "urgent": 2, "critical": 3}
            orig = risk_levels.get(original.get("risk_level", "routine"), 0)
            corr = risk_levels.get(corrected.get("risk_level", "routine"), 0)
            return min(abs(orig - corr) / 3.0, 1.0)
        
        if feedback_type == FeedbackType.DIAGNOSIS_CORRECTION:
            # Complete change = high severity
            return 0.8
        
        return 0.5
    
    def _apply_learning(self, feedback: DoctorFeedback):
        """Apply immediate learning from feedback"""
        
        if feedback.feedback_type == FeedbackType.SYMPTOM_CORRECTION:
            raw_input = feedback.corrected_values.get("raw_input", "").lower()
            correct_term = feedback.corrected_values.get("symptom", "")
            
            if raw_input and correct_term:
                self._save_mapping("symptom", raw_input, correct_term)
                self._learned_symptoms[raw_input] = correct_term
        
        elif feedback.feedback_type == FeedbackType.DIAGNOSIS_CORRECTION:
            self._diagnosis_hints.append({
                "correction": feedback.corrected_values,
                "reasoning": feedback.reasoning,
                "symptoms": feedback.original_prediction.get("symptoms", [])
            })
    
    def _save_mapping(self, mapping_type: str, input_text: str, output_text: str):
        """Save or update a learned mapping"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            now = datetime.utcnow().isoformat()
            
            cursor.execute("""
            INSERT INTO learned_mappings (mapping_type, input_text, output_text, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(mapping_type, input_text) DO UPDATE SET
                output_text = excluded.output_text,
                usage_count = usage_count + 1,
                confidence = MIN(confidence + 0.1, 1.0),
                updated_at = excluded.updated_at
            """, (mapping_type, input_text.lower(), output_text, now, now))
            
            conn.commit()
            conn.close()
            
            logger.info(f"📚 Learned mapping: '{input_text}' → '{output_text}'")
        except Exception as e:
            logger.error(f"Failed to save mapping: {e}")
    
    def get_learned_symptom(self, text: str) -> Optional[str]:
        """Get learned symptom mapping"""
        return self._learned_symptoms.get(text.lower())
    
    def get_diagnosis_context(self, symptoms: List[str]) -> str:
        """
        Get context from past corrections to enrich agent prompts.
        Returns a string that can be added to LLM prompts.
        """
        if not self._diagnosis_hints:
            return ""
        
        relevant_hints = []
        symptoms_lower = [s.lower() for s in symptoms]
        
        for hint in self._diagnosis_hints[-10:]:  # Last 10
            hint_symptoms = [s.lower() for s in hint.get("symptoms", [])]
            
            # Check for symptom overlap
            if any(s in hint_symptoms for s in symptoms_lower):
                correction = hint.get("correction", {})
                relevant_hints.append(
                    f"- Similar case: {correction.get('diagnosis', 'unknown')} "
                    f"(Reason: {hint.get('reasoning', 'N/A')})"
                )
        
        if relevant_hints:
            return "\n\nLEARNED FROM PAST CASES:\n" + "\n".join(relevant_hints[:3])
        
        return ""
    
    def get_performance_metrics(self, days: int = 30) -> Dict[str, Any]:
        """Get system performance based on feedback"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            since = (datetime.utcnow() - timedelta(days=days)).isoformat()
            
            cursor.execute("""
            SELECT 
                feedback_type,
                AVG(severity_score) as avg_severity,
                COUNT(*) as count
            FROM doctor_feedback
            WHERE created_at > ?
            GROUP BY feedback_type
            """, (since,))
            
            results = [dict(row) for row in cursor.fetchall()]
            conn.close()
            
            total = sum(r["count"] for r in results)
            confirmations = sum(r["count"] for r in results if r["feedback_type"] == "positive_confirmation")
            
            return {
                "total_feedback": total,
                "positive_confirmations": confirmations,
                "corrections_needed": total - confirmations,
                "accuracy": confirmations / total if total > 0 else 0.5,
                "avg_error_severity": sum(r["avg_severity"] * r["count"] for r in results) / total if total > 0 else 0,
                "by_type": results,
                "period_days": days
            }
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            return {"error": str(e)}


# Singleton
_learning_engine: Optional[FeedbackLearningEngine] = None


def get_learning_engine() -> FeedbackLearningEngine:
    global _learning_engine
    if _learning_engine is None:
        _learning_engine = FeedbackLearningEngine()
    return _learning_engine